"""
Zero-retrain diagnostic: verify that 3.1% MAPE comes from Base dims.

Loads trained model, runs forward on test set, computes:
1. CPI via compute_cpi_mean (normal eval path)
2. CPI directly from Base dims: 1/(Base * W_min)
3. Compare both with golden CPI

If (1) ≈ (2) ≈ 3.1%: error is in Base prediction, no hidden path.
If (2) << 3.1%: something else is contaminating the eval.

Usage:
    python diagnose_base_cpi.py --ckpt <path_to_checkpoint.pt>
    
    Or without checkpoint (will retrain briefly just to get a model):
    python diagnose_base_cpi.py --use-result <path_to_result.json>
"""

import sys, os, argparse, json
import numpy as np
import torch

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")

from config import (
    GEM_INPUT_DIM, DROPOUT, BENCHMARK_NAMES,
    SEED, TRAIN_RATIO, BASE_INDICES,
)
from model import GEMModel, compute_cpi_mean, compute_cpi_from_mscs
from dataset import load_all_benchmarks, FeatureNormalizer

DATASET_PATHS = {
    "18k": {
        "gem":     "/scratch2/wl730/micro_traindata_packed/gem",
        "concord": "/scratch2/wl730/micro_traindata_packed/concord",
    },
    "40k": {
        "gem":     "/scratch2/wl730/micro_traindata_packed_all/gem",
        "concord": "/scratch2/wl730/micro_traindata_packed_all/concord",
    },
}

def mape(pred, gold, eps=1e-8):
    return (torch.abs(pred - gold) / (torch.abs(gold) + eps) * 100).mean().item()

def per_bm_mape(pred, gold, bm_idx, eps=1e-8):
    results = {}
    for name, (s, e) in bm_idx.items():
        if s < e:
            results[name] = mape(pred[s:e], gold[s:e])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--dataset", type=str, default="18k", choices=["18k", "40k"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load data
    print("Loading data...")
    all_data = load_all_benchmarks(
        DATASET_PATHS[args.dataset]["gem"],
        DATASET_PATHS[args.dataset]["concord"],
        BENCHMARK_NAMES,
    )

    # Recreate test split (same as training)
    rng = np.random.RandomState(SEED)
    test_gem, test_mscs, test_cpi, test_wmin = [], [], [], []
    bm_idx = {}
    offset = 0
    all_train_gem = []

    # First pass: collect train data for normalizer
    rng1 = np.random.RandomState(SEED)
    for name, data in sorted(all_data.items()):
        n = data["n_samples"]
        indices = rng1.permutation(n)
        split = int(n * TRAIN_RATIO)
        all_train_gem.append(data["gem_features"][indices[:split]])
    
    gem_norm = FeatureNormalizer()
    gem_norm.fit(np.concatenate(all_train_gem))

    # Second pass: get test data
    rng2 = np.random.RandomState(SEED)
    for name, data in sorted(all_data.items()):
        n = data["n_samples"]
        indices = rng2.permutation(n)
        split = int(n * TRAIN_RATIO)
        te_idx = indices[split:]

        test_gem.append(data["gem_features"][te_idx])
        test_mscs.append(data["mscs_target"][te_idx])
        test_cpi.append(data["cpi_golden"][te_idx])
        test_wmin.append(data["w_min"][te_idx])

        n_test = len(te_idx)
        bm_idx[name] = (offset, offset + n_test)
        offset += n_test

    test_gem  = gem_norm.transform(np.concatenate(test_gem))
    test_mscs = np.concatenate(test_mscs)
    test_cpi  = np.concatenate(test_cpi)
    test_wmin = np.concatenate(test_wmin)

    test_gem_t  = torch.from_numpy(test_gem).to(device)
    test_mscs_t = torch.from_numpy(test_mscs).to(device)
    test_cpi_t  = torch.from_numpy(test_cpi).to(device)
    test_wmin_t = torch.from_numpy(test_wmin).to(device)

    print(f"Test samples: {len(test_cpi)}")

    # Load model
    model = GEMModel(GEM_INPUT_DIM, [512, 256], DROPOUT).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # Forward pass
    with torch.no_grad():
        mscs_pred = model(test_gem_t)

    # ============================================================
    # Path 1: Normal eval (compute_cpi_mean)
    # ============================================================
    with torch.no_grad():
        cpi_path1 = compute_cpi_mean(mscs_pred, test_wmin_t)
    mape_path1 = mape(cpi_path1, test_cpi_t)

    # ============================================================
    # Path 2: Direct from Base dims only: 1/(Base * W_min)
    # ============================================================
    with torch.no_grad():
        base_pred = mscs_pred[:, BASE_INDICES]                        # (N, 4)
        cpi_per_stage = 1.0 / (base_pred * test_wmin_t.unsqueeze(1) + 1e-8)  # (N, 4)
        cpi_path2 = cpi_per_stage.mean(dim=1)                        # (N,)
    mape_path2 = mape(cpi_path2, test_cpi_t)

    # ============================================================
    # Path 3: Use GOLDEN MSCS Base dims → CPI (upper bound)
    # ============================================================
    with torch.no_grad():
        golden_base = test_mscs_t[:, BASE_INDICES]
        cpi_per_stage_gold = 1.0 / (golden_base * test_wmin_t.unsqueeze(1) + 1e-8)
        cpi_path3 = cpi_per_stage_gold.mean(dim=1)
    mape_path3 = mape(cpi_path3, test_cpi_t)

    # ============================================================
    # Path 4: Per-stage CPI MAPE (which stage is worst?)
    # ============================================================
    stage_names = ["Decode", "Dispatch", "Issue", "Commit"]
    stage_mapes = []
    with torch.no_grad():
        for i, si in enumerate(BASE_INDICES):
            cpi_stage = 1.0 / (mscs_pred[:, si] * test_wmin_t + 1e-8)
            stage_mapes.append(mape(cpi_stage, test_cpi_t))

    # ============================================================
    # Per-dim MSE breakdown
    # ============================================================
    with torch.no_grad():
        dim_mse = ((mscs_pred - test_mscs_t) ** 2).mean(dim=0)
        base_mse = dim_mse[BASE_INDICES].mean().item()
        other_indices = [i for i in range(16) if i not in BASE_INDICES]
        other_mse = dim_mse[other_indices].mean().item()

    # ============================================================
    # Report
    # ============================================================
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: Base → CPI Error Chain")
    print(f"{'='*60}")
    
    print(f"\n--- CPI MAPE by computation path ---")
    print(f"  Path 1 (compute_cpi_mean, normal eval):  {mape_path1:.4f}%")
    print(f"  Path 2 (direct from Base dims):           {mape_path2:.4f}%")
    print(f"  Path 3 (golden Base dims → CPI):          {mape_path3:.4f}%")
    
    print(f"\n  Path1 ≈ Path2?  Δ = {abs(mape_path1 - mape_path2):.4f}%")
    if abs(mape_path1 - mape_path2) < 0.01:
        print(f"  → YES: 3.1% error comes purely from Base prediction.")
    else:
        print(f"  → NO: Something else is contaminating the eval path!")
        print(f"       This is a BUG. The difference ({abs(mape_path1 - mape_path2):.2f}%) is fixable.")
    
    print(f"\n  Golden Base MAPE = {mape_path3:.4f}%")
    if mape_path3 > 0.1:
        print(f"  → WARNING: Even golden MSCS Base gives {mape_path3:.2f}% MAPE.")
        print(f"       This means CPI ≠ 1/(Base·W_min) exactly in your data,")
        print(f"       or W_min computation has noise.")
    else:
        print(f"  → Good: golden Base → CPI is near-perfect ({mape_path3:.4f}%).")

    print(f"\n--- Per-stage CPI MAPE ---")
    for i, name in enumerate(stage_names):
        print(f"  {name:<10} CPI MAPE: {stage_mapes[i]:.2f}%")
    print(f"  Mean (=Path2):      {mape_path2:.2f}%")
    worst = stage_names[np.argmax(stage_mapes)]
    best  = stage_names[np.argmin(stage_mapes)]
    print(f"  Worst: {worst} ({max(stage_mapes):.2f}%), Best: {best} ({min(stage_mapes):.2f}%)")

    print(f"\n--- Per-dim MSE ---")
    print(f"  Base dims (4):   {base_mse:.6f}")
    print(f"  Other dims (12): {other_mse:.6f}")
    print(f"  Ratio Other/Base: {other_mse/base_mse:.0f}x")

    print(f"\n--- Per-benchmark (Path 1 vs Path 2) ---")
    pbm1 = per_bm_mape(cpi_path1, test_cpi_t, bm_idx)
    pbm2 = per_bm_mape(cpi_path2, test_cpi_t, bm_idx)
    print(f"  {'Benchmark':<20} {'Path1':>8} {'Path2':>8} {'Δ':>8}")
    for name in sorted(pbm1.keys()):
        d = pbm1[name] - pbm2[name]
        print(f"  {name:<20} {pbm1[name]:>7.2f}% {pbm2[name]:>7.2f}% {d:>+7.2f}%")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
