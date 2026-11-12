"""
Experiment 3 Supplement: Variant A+D3
  Full GEM features (4681 = Concorde 3873 + D3 808) + MSCS output + physics constraints

  Runs both in-distribution (15:1 split) and OOD (leave-one-out).

Usage:
    python train_exp3_d3.py --gpu 0
    python train_exp3_d3.py --gpu 0 --id-only
    python train_exp3_d3.py --gpu 0 --ood-only
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")
from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    TRAIN_RATIO, SEED,
)
from model import (
    GEMModel, compute_cpi_mean, loss_task, loss_const, loss_mono,
)
from dataset import (
    load_all_benchmarks, FeatureNormalizer, GEMDataset,
)


EXP3_OUTPUT_DIR = "/path/to/gem5/micro_exp/exp3/results"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


# ============================================================
# Data Splits (with full GEM features)
# ============================================================

def create_id_split(all_data, train_ratio=TRAIN_RATIO, batch_size=BATCH_SIZE, seed=SEED):
    """In-distribution split using full GEM features (4681 dims)."""
    rng = np.random.RandomState(seed)
    train_gem, train_mscs, train_cpi, train_wmin = [], [], [], []
    test_gem, test_mscs, test_cpi, test_wmin = [], [], [], []
    per_bm_test = {}
    test_offset = 0

    for name, data in sorted(all_data.items()):
        n = data["n_samples"]
        idx = rng.permutation(n)
        split = int(n * train_ratio)
        tr, te = idx[:split], idx[split:]

        for arrays, indices in [
            ((train_gem, train_mscs, train_cpi, train_wmin), tr),
            ((test_gem, test_mscs, test_cpi, test_wmin), te),
        ]:
            arrays[0].append(data["gem_features"][indices])
            arrays[1].append(data["mscs_target"][indices])
            arrays[2].append(data["cpi_golden"][indices])
            arrays[3].append(data["w_min"][indices])

        n_te = len(te)
        per_bm_test[name] = (test_offset, test_offset + n_te)
        test_offset += n_te

    train_gem  = np.concatenate(train_gem)
    train_mscs = np.concatenate(train_mscs)
    train_cpi  = np.concatenate(train_cpi)
    train_wmin = np.concatenate(train_wmin)
    test_gem   = np.concatenate(test_gem)
    test_mscs  = np.concatenate(test_mscs)
    test_cpi   = np.concatenate(test_cpi)
    test_wmin  = np.concatenate(test_wmin)

    norm = FeatureNormalizer()
    train_gem = norm.fit_transform(train_gem)
    test_gem  = norm.transform(test_gem)

    # GEMDataset: gem_features=full, concord_features=full (not used separately)
    train_ds = GEMDataset(train_gem, train_gem, train_mscs, train_cpi, train_wmin)
    test_ds  = GEMDataset(test_gem, test_gem, test_mscs, test_cpi, test_wmin)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, test_loader, per_bm_test


def create_loo_split(all_data, held_out, batch_size=BATCH_SIZE):
    """Leave-one-out using full GEM features."""
    train_gem, train_mscs, train_cpi, train_wmin = [], [], [], []

    for name, data in sorted(all_data.items()):
        if name == held_out:
            continue
        train_gem.append(data["gem_features"])
        train_mscs.append(data["mscs_target"])
        train_cpi.append(data["cpi_golden"])
        train_wmin.append(data["w_min"])

    train_gem  = np.concatenate(train_gem)
    train_mscs = np.concatenate(train_mscs)
    train_cpi  = np.concatenate(train_cpi)
    train_wmin = np.concatenate(train_wmin)

    test_data = all_data[held_out]
    test_gem  = test_data["gem_features"]
    test_mscs = test_data["mscs_target"]
    test_cpi  = test_data["cpi_golden"]
    test_wmin = test_data["w_min"]

    norm = FeatureNormalizer()
    train_gem = norm.fit_transform(train_gem)
    test_gem  = norm.transform(test_gem)

    train_ds = GEMDataset(train_gem, train_gem, train_mscs, train_cpi, train_wmin)
    test_ds  = GEMDataset(test_gem, test_gem, test_mscs, test_cpi, test_wmin)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, test_loader


# ============================================================
# Evaluate & Train
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_gold = [], []
    for batch in loader:
        feat  = batch["gem_features"].to(device)
        w_min = batch["w_min"].to(device)
        cpi_gold = batch["cpi_golden"].to(device)
        mscs_pred = model(feat)
        cpi_pred = compute_cpi_mean(mscs_pred, w_min)
        all_pred.append(cpi_pred.cpu())
        all_gold.append(cpi_gold.cpu())
    return compute_mape(torch.cat(all_pred), torch.cat(all_gold))


def train_a_d3(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS):
    """Variant A+D3: full GEM features + MSCS + physics constraints."""
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

        mape = evaluate(model, test_loader, device)
        if mape < best_mape:
            best_mape = mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
        if epoch % 50 == 0:
            print(f"      epoch {epoch:3d} | MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    return model, best_mape


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Exp3: Variant A+D3")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--id-only", action="store_true")
    parser.add_argument("--ood-only", action="store_true")
    parser.add_argument("--target", type=str, default=None,
                        help="Run OOD for single benchmark (for parallel runs)")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(EXP3_OUTPUT_DIR, exist_ok=True)

    print("\nLoading benchmarks...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    benchmark_names = sorted(all_data.keys())
    results = {}

    # ============================
    # In-Distribution
    # ============================
    if not args.ood_only:
        print("\n" + "=" * 60)
        print("  Variant A+D3: In-Distribution")
        print(f"  Input: {GEM_INPUT_DIM} dims (Concorde {CONCORDE_INPUT_DIM} + D3 {GEM_INPUT_DIM - CONCORDE_INPUT_DIM})")
        print("=" * 60)

        train_loader, test_loader, _ = create_id_split(all_data)
        model = GEMModel(
            input_dim=GEM_INPUT_DIM,
            hidden_dims=CONCORDE_HIDDEN_DIMS,
            dropout=DROPOUT,
        ).to(device)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        _, id_mape = train_a_d3(model, train_loader, test_loader, device)
        results["id_mape"] = id_mape
        print(f"\n  A+D3 In-Distribution MAPE: {id_mape:.2f}%")

        with open(os.path.join(EXP3_OUTPUT_DIR, "exp3_d3_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # ============================
    # OOD (Leave-One-Out)
    # ============================
    if not args.id_only:
        print("\n" + "=" * 60)
        print("  Variant A+D3: OOD (Leave-One-Out)")
        print("=" * 60)

        # Determine which benchmarks to run
        if args.target:
            assert args.target in benchmark_names, \
                f"Unknown benchmark '{args.target}'. Available: {benchmark_names}"
            targets = [args.target]
        else:
            targets = benchmark_names

        ood_results = []
        for i, held_out in enumerate(targets):
            n_test = all_data[held_out]["n_samples"]
            print(f"\n  [{i+1}/{len(targets)}] Holding out: {held_out} ({n_test} samples)")

            train_loader, test_loader = create_loo_split(all_data, held_out)
            model = GEMModel(
                input_dim=GEM_INPUT_DIM,
                hidden_dims=CONCORDE_HIDDEN_DIMS,
                dropout=DROPOUT,
            ).to(device)

            _, mape = train_a_d3(model, train_loader, test_loader, device)
            print(f"    A+D3 MAPE: {mape:.2f}%")

            ood_results.append({"held_out": held_out, "mape": mape, "n_samples": n_test})

            # Save per-benchmark result (for parallel runs)
            per_bm_path = os.path.join(EXP3_OUTPUT_DIR, f"exp3_d3_ood_{held_out}.json")
            with open(per_bm_path, "w") as f:
                json.dump(ood_results[-1], f, indent=2)

        # Summary (only if running all)
        if not args.target:
            avg = np.mean([r["mape"] for r in ood_results])
            print(f"\n  --- A+D3 OOD Summary ---")
            print(f"  {'Benchmark':<20} {'A+D3 MAPE':>12}")
            print(f"  {'-'*34}")
            for r in ood_results:
                print(f"  {r['held_out']:<20} {r['mape']:>11.2f}%")
            print(f"  {'-'*34}")
            print(f"  {'Average':<20} {avg:>11.2f}%")
            results["ood"] = ood_results
            results["ood_avg"] = avg

    result_path = os.path.join(EXP3_OUTPUT_DIR, "exp3_d3_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()