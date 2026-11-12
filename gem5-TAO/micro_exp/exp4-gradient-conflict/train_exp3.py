"""
Experiment 3: Ablation Study
  Variant A: Concorde features → MSCS output, L_task + L_const + L_mono
  Variant B: Concorde features → MSCS output, L_task only (no physics)
  Variant C: Concorde features → scalar CPI (original Concorde baseline)

  All three use identical: input features (3873), hidden dims [256,128].
  Only difference: output format and loss function.

  Runs BOTH in-distribution (15:1 split) and OOD (leave-one-out).

Usage:
    python train_exp3.py --gpu 0                # full run
    python train_exp3.py --gpu 0 --ood-only     # skip in-distribution
    python train_exp3.py --gpu 0 --id-only      # skip OOD
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader

# --- Import from exp1 ---
sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")
from config import (
    CONCORDE_INPUT_DIM, MSCS_DIM,
    CONCORDE_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE,
    CONCORDE_LR, CONCORDE_WEIGHT_DECAY, CONCORDE_BATCH_SIZE, CONCORDE_EPOCHS,
    LAMBDA_CONST, LAMBDA_MONO,
    BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    TRAIN_RATIO, SEED,
)
from model import (
    GEMModel, ConcordeModel,
    compute_cpi_mean, loss_task, loss_const, loss_mono,
)
from dataset import (
    load_all_benchmarks, FeatureNormalizer, GEMDataset,
)


EXP3_OUTPUT_DIR = "/scratch/gem5-TAO/micro_exp/exp3/results"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


# ============================================================
# Data Splits
# ============================================================

def create_id_split(all_data, train_ratio=TRAIN_RATIO, batch_size=BATCH_SIZE, seed=SEED):
    """In-distribution split: 15:1 within each benchmark."""
    rng = np.random.RandomState(seed)
    train_arrays = {k: [] for k in ["con", "mscs", "cpi", "wmin"]}
    test_arrays  = {k: [] for k in ["con", "mscs", "cpi", "wmin"]}
    per_bm_test = {}
    test_offset = 0

    for name, data in sorted(all_data.items()):
        n = data["n_samples"]
        idx = rng.permutation(n)
        split = int(n * train_ratio)
        tr, te = idx[:split], idx[split:]

        for arr_dict, indices in [(train_arrays, tr), (test_arrays, te)]:
            arr_dict["con"].append(data["concord_features"][indices])
            arr_dict["mscs"].append(data["mscs_target"][indices])
            arr_dict["cpi"].append(data["cpi_golden"][indices])
            arr_dict["wmin"].append(data["w_min"][indices])

        n_te = len(te)
        per_bm_test[name] = (test_offset, test_offset + n_te)
        test_offset += n_te

    # Concatenate
    for d in [train_arrays, test_arrays]:
        for k in d:
            d[k] = np.concatenate(d[k])

    # Normalize (fit on train)
    norm = FeatureNormalizer()
    train_arrays["con"] = norm.fit_transform(train_arrays["con"])
    test_arrays["con"]  = norm.transform(test_arrays["con"])

    # Build datasets — use concord_features for BOTH gem_features and concord_features
    train_ds = GEMDataset(
        train_arrays["con"], train_arrays["con"],
        train_arrays["mscs"], train_arrays["cpi"], train_arrays["wmin"])
    test_ds = GEMDataset(
        test_arrays["con"], test_arrays["con"],
        test_arrays["mscs"], test_arrays["cpi"], test_arrays["wmin"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, test_loader, per_bm_test


def create_loo_split(all_data, held_out, batch_size=BATCH_SIZE):
    """Leave-one-out: train on all except held_out, test on held_out."""
    train_con, train_mscs, train_cpi, train_wmin = [], [], [], []

    for name, data in sorted(all_data.items()):
        if name == held_out:
            continue
        train_con.append(data["concord_features"])
        train_mscs.append(data["mscs_target"])
        train_cpi.append(data["cpi_golden"])
        train_wmin.append(data["w_min"])

    train_con  = np.concatenate(train_con)
    train_mscs = np.concatenate(train_mscs)
    train_cpi  = np.concatenate(train_cpi)
    train_wmin = np.concatenate(train_wmin)

    test_data = all_data[held_out]
    test_con  = test_data["concord_features"]
    test_mscs = test_data["mscs_target"]
    test_cpi  = test_data["cpi_golden"]
    test_wmin = test_data["w_min"]

    norm = FeatureNormalizer()
    train_con = norm.fit_transform(train_con)
    test_con  = norm.transform(test_con)

    train_ds = GEMDataset(train_con, train_con, train_mscs, train_cpi, train_wmin)
    test_ds  = GEMDataset(test_con, test_con, test_mscs, test_cpi, test_wmin)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, test_loader


# ============================================================
# Evaluate
# ============================================================

@torch.no_grad()
def evaluate_mscs_model(model, loader, device):
    """Evaluate Variant A or B (MSCS output)."""
    model.eval()
    all_pred, all_gold = [], []
    for batch in loader:
        feat     = batch["gem_features"].to(device)  # actually concord features
        w_min    = batch["w_min"].to(device)
        cpi_gold = batch["cpi_golden"].to(device)
        mscs_pred = model(feat)
        cpi_pred  = compute_cpi_mean(mscs_pred, w_min)
        all_pred.append(cpi_pred.cpu())
        all_gold.append(cpi_gold.cpu())
    all_pred = torch.cat(all_pred)
    all_gold = torch.cat(all_gold)
    return compute_mape(all_pred, all_gold)


@torch.no_grad()
def evaluate_scalar_model(model, loader, device):
    """Evaluate Variant C (scalar CPI output)."""
    model.eval()
    all_pred, all_gold = [], []
    for batch in loader:
        feat     = batch["concord_features"].to(device)
        cpi_gold = batch["cpi_golden"].to(device)
        cpi_pred = model(feat)
        all_pred.append(cpi_pred.cpu())
        all_gold.append(cpi_gold.cpu())
    all_pred = torch.cat(all_pred)
    all_gold = torch.cat(all_gold)
    return compute_mape(all_pred, all_gold)


# ============================================================
# Training Functions
# ============================================================

def train_variant_a(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS):
    """Variant A: MSCS + L_task + L_const + L_mono (full physics)."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_mape, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            feat     = batch["gem_features"].to(device)
            mscs_tgt = batch["mscs_target"].to(device)
            cpi_gold = batch["cpi_golden"].to(device)
            w_min    = batch["w_min"].to(device)

            mscs_pred = model(feat)
            l_t = loss_task(mscs_pred, mscs_tgt)
            l_c = loss_const(mscs_pred, w_min, cpi_gold)
            l_m = loss_mono(mscs_pred)
            loss = l_t + LAMBDA_CONST * l_c + LAMBDA_MONO * l_m

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        mape = evaluate_mscs_model(model, test_loader, device)
        if mape < best_mape:
            best_mape = mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
        if epoch % 50 == 0:
            print(f"        A epoch {epoch:3d} | MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    return model, best_mape


def train_variant_b(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS):
    """Variant B: MSCS + L_task only (no physics constraints)."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_mape, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            feat     = batch["gem_features"].to(device)
            mscs_tgt = batch["mscs_target"].to(device)

            mscs_pred = model(feat)
            loss = loss_task(mscs_pred, mscs_tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        mape = evaluate_mscs_model(model, test_loader, device)
        if mape < best_mape:
            best_mape = mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
        if epoch % 50 == 0:
            print(f"        B epoch {epoch:3d} | MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    return model, best_mape


def train_variant_c(model, train_loader, test_loader, device, num_epochs=CONCORDE_EPOCHS):
    """Variant C: Scalar CPI with relative error loss (Concorde baseline)."""
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR,
                            weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    def relative_loss(pred, target, eps=1e-8):
        return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()

    best_mape, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            feat     = batch["concord_features"].to(device)
            cpi_gold = batch["cpi_golden"].to(device)
            cpi_pred = model(feat)
            loss = relative_loss(cpi_pred, cpi_gold)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        mape = evaluate_scalar_model(model, test_loader, device)
        if mape < best_mape:
            best_mape = mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
        if epoch % 50 == 0:
            print(f"        C epoch {epoch:3d} | MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    return model, best_mape


# ============================================================
# Run One Configuration
# ============================================================

def make_mscs_model(device):
    """Create MSCS model (Variant A or B): concord_features → MSCS."""
    return GEMModel(
        input_dim=CONCORDE_INPUT_DIM,
        hidden_dims=CONCORDE_HIDDEN_DIMS,
        dropout=DROPOUT,
    ).to(device)


def make_scalar_model(device):
    """Create scalar CPI model (Variant C): concord_features → CPI."""
    return ConcordeModel(
        input_dim=CONCORDE_INPUT_DIM,
        hidden_dims=CONCORDE_HIDDEN_DIMS,
        dropout=0.0,
    ).to(device)


def run_all_variants(train_loader, test_loader, device, label=""):
    """Train and evaluate all three variants. Returns dict of MAPEs."""
    results = {}

    # Variant A: MSCS + physics
    print(f"      [A] MSCS + L_task + L_const + L_mono")
    model_a = make_mscs_model(device)
    _, mape_a = train_variant_a(model_a, train_loader, test_loader, device)
    results["A_physics"] = mape_a
    print(f"      [A] best MAPE: {mape_a:.2f}%")

    # Variant B: MSCS only
    print(f"      [B] MSCS + L_task only")
    model_b = make_mscs_model(device)
    _, mape_b = train_variant_b(model_b, train_loader, test_loader, device)
    results["B_no_physics"] = mape_b
    print(f"      [B] best MAPE: {mape_b:.2f}%")

    # Variant C: scalar CPI (Concorde)
    print(f"      [C] Scalar CPI (Concorde)")
    # Large batch for Concorde
    con_batch = min(CONCORDE_BATCH_SIZE, len(train_loader.dataset))
    con_loader = DataLoader(train_loader.dataset, batch_size=con_batch,
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    model_c = make_scalar_model(device)
    _, mape_c = train_variant_c(model_c, con_loader, test_loader, device)
    results["C_scalar"] = mape_c
    print(f"      [C] best MAPE: {mape_c:.2f}%")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Exp3: Ablation (A vs B vs C)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--id-only", action="store_true", help="Only in-distribution")
    parser.add_argument("--ood-only", action="store_true", help="Only OOD (leave-one-out)")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(EXP3_OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\nLoading benchmarks...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    benchmark_names = sorted(all_data.keys())
    all_results = {}

    # ============================================
    # Part 1: In-Distribution
    # ============================================
    if not args.ood_only:
        print("\n" + "=" * 70)
        print("  PART 1: In-Distribution (15:1 split)")
        print("=" * 70)

        train_loader, test_loader, per_bm_test = create_id_split(all_data)
        print(f"  Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")

        id_results = run_all_variants(train_loader, test_loader, device, "ID")

        print(f"\n  --- In-Distribution Summary ---")
        print(f"  Variant A (MSCS+physics): {id_results['A_physics']:.2f}%")
        print(f"  Variant B (MSCS only):    {id_results['B_no_physics']:.2f}%")
        print(f"  Variant C (scalar CPI):   {id_results['C_scalar']:.2f}%")

        all_results["in_distribution"] = id_results

        # Save incrementally
        with open(os.path.join(EXP3_OUTPUT_DIR, "exp3_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    # ============================================
    # Part 2: OOD (Leave-One-Out)
    # ============================================
    if not args.id_only:
        print("\n" + "=" * 70)
        print("  PART 2: OOD (Leave-One-Out)")
        print("=" * 70)

        ood_per_bm = []
        for i, held_out in enumerate(benchmark_names):
            n_test = all_data[held_out]["n_samples"]
            print(f"\n  [{i+1}/{len(benchmark_names)}] Holding out: {held_out} ({n_test} samples)")

            train_loader, test_loader = create_loo_split(all_data, held_out)
            bm_results = run_all_variants(train_loader, test_loader, device, held_out)
            bm_results["held_out"] = held_out
            bm_results["n_samples"] = n_test
            ood_per_bm.append(bm_results)

            # Save incrementally
            all_results["ood_per_benchmark"] = ood_per_bm
            with open(os.path.join(EXP3_OUTPUT_DIR, "exp3_results.json"), "w") as f:
                json.dump(all_results, f, indent=2)

        # OOD Summary
        avg_a = np.mean([r["A_physics"] for r in ood_per_bm])
        avg_b = np.mean([r["B_no_physics"] for r in ood_per_bm])
        avg_c = np.mean([r["C_scalar"] for r in ood_per_bm])

        print(f"\n  --- OOD Summary ---")
        print(f"  {'Benchmark':<20} {'A(physics)':>12} {'B(no phys)':>12} {'C(scalar)':>12} {'Best':>8}")
        print(f"  {'-'*66}")
        for r in ood_per_bm:
            a, b, c = r["A_physics"], r["B_no_physics"], r["C_scalar"]
            best = "A" if a <= b and a <= c else ("B" if b <= c else "C")
            print(f"  {r['held_out']:<20} {a:>11.2f}% {b:>11.2f}% {c:>11.2f}%  {best:>6}")
        print(f"  {'-'*66}")
        print(f"  {'Average':<20} {avg_a:>11.2f}% {avg_b:>11.2f}% {avg_c:>11.2f}%")

        a_wins = sum(1 for r in ood_per_bm
                     if r["A_physics"] <= r["B_no_physics"] and r["A_physics"] <= r["C_scalar"])
        print(f"  A wins: {a_wins}/{len(ood_per_bm)}")

        all_results["ood_summary"] = {
            "avg_A": avg_a, "avg_B": avg_b, "avg_C": avg_c,
            "A_wins": a_wins, "total": len(ood_per_bm),
        }

    # Save final
    result_path = os.path.join(EXP3_OUTPUT_DIR, "exp3_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()