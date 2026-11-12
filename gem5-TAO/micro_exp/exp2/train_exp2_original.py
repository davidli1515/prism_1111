"""
Experiment 2: OOD Generalization (Low-Data Setting)
  For each target benchmark:
    - Other benchmarks: ALL data in training set
    - Target benchmark: 1/16 in training, 15/16 in test
  Three variants: GEM+D3, GEM-noD3, Concorde

Usage:
    python train_exp2.py --gpu 0                     # all benchmarks
    python train_exp2.py --gpu 0 --target 508.namd_r # single benchmark
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

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")
from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM,
    CONCORDE_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE,
    CONCORDE_LR, CONCORDE_WEIGHT_DECAY, CONCORDE_BATCH_SIZE, CONCORDE_EPOCHS,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    SEED,
)
from model import (
    GEMModel, ConcordeModel,
    compute_cpi_mean, loss_task, loss_const, loss_mono,
)
from dataset import (
    load_all_benchmarks, FeatureNormalizer, GEMDataset,
)

EXP2_OUTPUT_DIR = "/scratch/gem5-TAO/micro_exp/exp2/results"
OOD_TRAIN_RATIO = 1 / 16   # target benchmark: 1/16 train, 15/16 test


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


# ============================================================
# Data Split
# ============================================================

def create_ood_split(all_data, target_name, ood_train_ratio=OOD_TRAIN_RATIO,
                     batch_size=BATCH_SIZE, seed=SEED):
    """
    OOD split:
      - All other benchmarks: 100% in train
      - Target benchmark: ood_train_ratio (1/16) in train, rest in test
    """
    rng = np.random.RandomState(seed)
    train_gem, train_con, train_mscs, train_cpi, train_wmin = [], [], [], [], []
    test_gem, test_con, test_mscs, test_cpi, test_wmin = [], [], [], [], []

    for name, data in sorted(all_data.items()):
        if name == target_name:
            # Split target: 1/16 train, 15/16 test
            n = data["n_samples"]
            idx = rng.permutation(n)
            split = max(1, int(n * ood_train_ratio))
            tr_idx, te_idx = idx[:split], idx[split:]

            train_gem.append(data["gem_features"][tr_idx])
            train_con.append(data["concord_features"][tr_idx])
            train_mscs.append(data["mscs_target"][tr_idx])
            train_cpi.append(data["cpi_golden"][tr_idx])
            train_wmin.append(data["w_min"][tr_idx])

            test_gem.append(data["gem_features"][te_idx])
            test_con.append(data["concord_features"][te_idx])
            test_mscs.append(data["mscs_target"][te_idx])
            test_cpi.append(data["cpi_golden"][te_idx])
            test_wmin.append(data["w_min"][te_idx])
        else:
            # Other benchmarks: all in train
            train_gem.append(data["gem_features"])
            train_con.append(data["concord_features"])
            train_mscs.append(data["mscs_target"])
            train_cpi.append(data["cpi_golden"])
            train_wmin.append(data["w_min"])

    # Concatenate
    train_gem  = np.concatenate(train_gem)
    train_con  = np.concatenate(train_con)
    train_mscs = np.concatenate(train_mscs)
    train_cpi  = np.concatenate(train_cpi)
    train_wmin = np.concatenate(train_wmin)

    test_gem  = np.concatenate(test_gem)
    test_con  = np.concatenate(test_con)
    test_mscs = np.concatenate(test_mscs)
    test_cpi  = np.concatenate(test_cpi)
    test_wmin = np.concatenate(test_wmin)

    # Normalize (fit on train)
    gem_norm = FeatureNormalizer()
    train_gem = gem_norm.fit_transform(train_gem)
    test_gem  = gem_norm.transform(test_gem)

    con_norm = FeatureNormalizer()
    train_con = con_norm.fit_transform(train_con)
    test_con  = con_norm.transform(test_con)

    train_ds = GEMDataset(train_gem, train_con, train_mscs, train_cpi, train_wmin)
    test_ds  = GEMDataset(test_gem, test_con, test_mscs, test_cpi, test_wmin)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    n_target_train = max(1, int(all_data[target_name]["n_samples"] * ood_train_ratio))
    n_target_test  = all_data[target_name]["n_samples"] - n_target_train

    return train_loader, test_loader, n_target_train, n_target_test


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_gem(model, loader, device, use_gem_features=True):
    model.eval()
    all_pred, all_gold = [], []
    for batch in loader:
        feat  = batch["gem_features" if use_gem_features else "concord_features"].to(device)
        w_min = batch["w_min"].to(device)
        mscs_pred = model(feat)
        cpi_pred = compute_cpi_mean(mscs_pred, w_min)
        all_pred.append(cpi_pred.cpu())
        all_gold.append(batch["cpi_golden"])
    return compute_mape(torch.cat(all_pred), torch.cat(all_gold))


@torch.no_grad()
def evaluate_concorde(model, loader, device):
    model.eval()
    all_pred, all_gold = [], []
    for batch in loader:
        feat = batch["concord_features"].to(device)
        cpi_pred = model(feat)
        all_pred.append(cpi_pred.cpu())
        all_gold.append(batch["cpi_golden"])
    return compute_mape(torch.cat(all_pred), torch.cat(all_gold))


# ============================================================
# Training
# ============================================================

def train_gem_quiet(model, train_loader, test_loader, device,
                    use_gem_features=True, num_epochs=NUM_EPOCHS, label="GEM"):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    feat_key = "gem_features" if use_gem_features else "concord_features"
    best_mape, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            feat     = batch[feat_key].to(device)
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

        mape = evaluate_gem(model, test_loader, device, use_gem_features)
        if mape < best_mape:
            best_mape = mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
        if epoch % 50 == 0:
            print(f"        [{label}] epoch {epoch:3d} | MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    return model, best_mape


def train_concorde_quiet(model, train_loader, test_loader, device,
                         num_epochs=CONCORDE_EPOCHS):
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
            feat = batch["concord_features"].to(device)
            cpi_gold = batch["cpi_golden"].to(device)
            cpi_pred = model(feat)
            loss = relative_loss(cpi_pred, cpi_gold)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        mape = evaluate_concorde(model, test_loader, device)
        if mape < best_mape:
            best_mape = mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
        if epoch % 50 == 0:
            print(f"        [Concorde] epoch {epoch:3d} | MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    return model, best_mape


# ============================================================
# Run One Target
# ============================================================

def run_one_target(all_data, target_name, device):
    """Run all 3 variants for one target benchmark."""
    n_total = all_data[target_name]["n_samples"]
    train_loader, test_loader, n_train, n_test = create_ood_split(all_data, target_name)

    print(f"\n  Target: {target_name}")
    print(f"    Target samples: {n_train} train / {n_test} test (ratio {OOD_TRAIN_RATIO:.4f})")
    print(f"    Total train: {len(train_loader.dataset)}")

    results = {"target": target_name, "n_total": n_total,
               "n_target_train": n_train, "n_target_test": n_test}

    # --- GEM+D3 ---
    print(f"    Training GEM+D3...")
    t0 = time.time()
    m = GEMModel(GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT).to(device)
    _, mape = train_gem_quiet(m, train_loader, test_loader, device, True, label="GEM+D3")
    results["gem_full"] = mape
    print(f"    GEM+D3: {mape:.2f}% ({time.time()-t0:.0f}s)")

    # --- GEM-noD3 ---
    print(f"    Training GEM-noD3...")
    t0 = time.time()
    m = GEMModel(CONCORDE_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT).to(device)
    _, mape = train_gem_quiet(m, train_loader, test_loader, device, False, label="GEM-noD3")
    results["gem_nod3"] = mape
    print(f"    GEM-noD3: {mape:.2f}% ({time.time()-t0:.0f}s)")

    # --- Concorde ---
    print(f"    Training Concorde...")
    t0 = time.time()
    con_batch = min(CONCORDE_BATCH_SIZE, len(train_loader.dataset))
    con_loader = DataLoader(train_loader.dataset, batch_size=con_batch,
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    m = ConcordeModel(CONCORDE_INPUT_DIM, CONCORDE_HIDDEN_DIMS, 0.0).to(device)
    _, mape = train_concorde_quiet(m, con_loader, test_loader, device)
    results["concorde"] = mape
    print(f"    Concorde: {mape:.2f}% ({time.time()-t0:.0f}s)")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Exp2: OOD (1:15 split)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target", type=str, default=None,
                        help="Single target benchmark (for parallel runs)")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(EXP2_OUTPUT_DIR, exist_ok=True)

    print("\nLoading benchmarks...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    benchmark_names = sorted(all_data.keys())

    targets = [args.target] if args.target else benchmark_names
    if args.target:
        assert args.target in benchmark_names, f"Unknown: {args.target}"

    print(f"\n{'='*70}")
    print(f"  Exp2: OOD Generalization (target gets {OOD_TRAIN_RATIO:.2%} train data)")
    print(f"  Targets: {len(targets)} benchmarks")
    print(f"{'='*70}")

    all_results = []
    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}]", "=" * 50)
        result = run_one_target(all_data, target, device)
        all_results.append(result)

        # Save per-benchmark (for parallel runs)
        per_path = os.path.join(EXP2_OUTPUT_DIR, f"exp2_ood_{target}.json")
        with open(per_path, "w") as f:
            json.dump(result, f, indent=2)

    # Summary (if running all)
    if not args.target:
        print(f"\n{'='*75}")
        print(f"{'Benchmark':<20} {'Samples':>8} {'GEM+D3':>10} {'GEM-noD3':>10} {'Concorde':>10} {'Best':>10}")
        print(f"{'-'*75}")
        for r in all_results:
            g, n, c = r["gem_full"], r["gem_nod3"], r["concorde"]
            best = min([(g,"GEM+D3"), (n,"GEM-noD3"), (c,"Concorde")])[1]
            print(f"{r['target']:<20} {r['n_target_test']:>8} {g:>9.2f}% {n:>9.2f}% {c:>9.2f}%  {best:>10}")

        avg_g = np.mean([r["gem_full"] for r in all_results])
        avg_n = np.mean([r["gem_nod3"] for r in all_results])
        avg_c = np.mean([r["concorde"] for r in all_results])
        print(f"{'-'*75}")
        print(f"{'Average':<20} {'':>8} {avg_g:>9.2f}% {avg_n:>9.2f}% {avg_c:>9.2f}%")

        gf_wins = sum(1 for r in all_results
                      if r["gem_full"] <= r["gem_nod3"] and r["gem_full"] <= r["concorde"])
        print(f"GEM+D3 wins: {gf_wins}/{len(all_results)}")
        print(f"{'='*75}")

        summary = {
            "per_benchmark": all_results,
            "average": {"gem_full": avg_g, "gem_nod3": avg_n, "concorde": avg_c},
            "ood_train_ratio": OOD_TRAIN_RATIO,
        }
        with open(os.path.join(EXP2_OUTPUT_DIR, "exp2_results.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()