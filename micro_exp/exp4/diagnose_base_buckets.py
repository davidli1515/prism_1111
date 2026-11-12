"""
Zero-retrain diagnostic: check if small-Base samples have larger residuals.

Hypothesis: MSE treats all Base errors equally, but CPI = 1/(Base*W_min)
amplifies small-Base errors. If small-Base bucket has significantly larger
CPI residuals, the 1/x nonlinearity is the bottleneck.

Usage:
    python diagnose_base_buckets.py
    
    (Uses results from most recent extra run - reads test predictions inline)
"""

import sys, os, json
import numpy as np
import torch

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")

from config import (
    GEM_INPUT_DIM, DROPOUT, BENCHMARK_NAMES,
    SEED, TRAIN_RATIO, BASE_INDICES,
)
from model import GEMModel, compute_cpi_mean
from dataset import load_all_benchmarks, FeatureNormalizer

import hashlib

DATASET_PATHS_18K = {
    "gem":     "/data/micro_traindata_packed/gem",
    "concord": "/data/micro_traindata_packed/concord",
}
DATASET_PATHS_40K = {
    "gem":     "/data/micro_traindata_packed_all/gem",
    "concord": "/data/micro_traindata_packed_all/concord",
}


def find_extra_indices(data_18k_bench, data_40k_bench):
    gem_18k = data_18k_bench["gem_features"]
    gem_40k = data_40k_bench["gem_features"]
    set_18k = set()
    for i in range(len(gem_18k)):
        set_18k.add(gem_18k[i].tobytes())
    extra_idx = []
    for i in range(len(gem_40k)):
        if gem_40k[i].tobytes() not in set_18k:
            extra_idx.append(i)
    return np.array(extra_idx, dtype=np.int64)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading 18k + 40k data...")
    data_18k = load_all_benchmarks(DATASET_PATHS_18K["gem"], DATASET_PATHS_18K["concord"], BENCHMARK_NAMES)
    data_40k = load_all_benchmarks(DATASET_PATHS_40K["gem"], DATASET_PATHS_40K["concord"], BENCHMARK_NAMES)

    # Recreate the EXACT same data pipeline as train_exp4_extra_unlabeled.py
    rng_split = np.random.RandomState(SEED)

    train_gem, test_gem = [], []
    test_mscs, test_cpi, test_wmin = [], [], []
    test_benchmarks = []  # track which benchmark each test sample belongs to

    for name in sorted(data_18k.keys()):
        d18 = data_18k[name]
        d40 = data_40k[name]

        n = d18["n_samples"]
        indices = rng_split.permutation(n)
        split = int(n * TRAIN_RATIO)
        tr_idx = indices[:split]
        te_idx = indices[split:]

        # Collect ALL train data for normalizer (18k train + 40k extra)
        train_gem.append(d18["gem_features"][tr_idx])
        extra_idx = find_extra_indices(d18, d40)
        if len(extra_idx) > 0:
            train_gem.append(d40["gem_features"][extra_idx])

        # Test data
        test_gem.append(d18["gem_features"][te_idx])
        test_mscs.append(d18["mscs_target"][te_idx])
        test_cpi.append(d18["cpi_golden"][te_idx])
        test_wmin.append(d18["w_min"][te_idx])
        test_benchmarks.extend([name] * len(te_idx))

    # Normalize using the SAME normalizer as training
    all_train_gem = np.concatenate(train_gem)
    gem_norm = FeatureNormalizer()
    gem_norm.fit(all_train_gem)

    test_gem  = gem_norm.transform(np.concatenate(test_gem))
    test_mscs = np.concatenate(test_mscs)
    test_cpi  = np.concatenate(test_cpi)
    test_wmin = np.concatenate(test_wmin)

    print(f"Test samples: {len(test_cpi)}")

    # Try to find checkpoint
    ckpt_candidates = [
        "results_relobralo/exp4_18k_extra_unlabeled/checkpoints/extra_60pct.pt",
        "results_run2/exp4_18k_extra_unlabeled/checkpoints/extra_60pct.pt",
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if os.path.exists(c):
            ckpt_path = c
            break

    if ckpt_path is None:
        print("ERROR: No checkpoint found. Tried:")
        for c in ckpt_candidates:
            print(f"  {c}")
        print("\nRun with checkpoint path as argument or check paths.")
        return

    # Load model
    model = GEMModel(GEM_INPUT_DIM, [512, 256], DROPOUT).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded: {ckpt_path}")

    # Forward pass
    test_gem_t  = torch.from_numpy(test_gem).to(device)
    test_mscs_t = torch.from_numpy(test_mscs).to(device)
    test_cpi_t  = torch.from_numpy(test_cpi)
    test_wmin_t = torch.from_numpy(test_wmin).to(device)

    with torch.no_grad():
        mscs_pred = model(test_gem_t).cpu()
        cpi_pred = compute_cpi_mean(mscs_pred, torch.from_numpy(test_wmin)).numpy()

    mscs_pred = mscs_pred.numpy()

    # ============================================================
    # Bucket analysis: group by golden Base magnitude
    # ============================================================
    golden_base = test_mscs[:, BASE_INDICES]  # (N, 4)
    pred_base   = mscs_pred[:, BASE_INDICES]  # (N, 4)

    # Per-sample: mean Base across 4 stages
    mean_golden_base = golden_base.mean(axis=1)  # (N,)

    # Base absolute error and relative error
    base_abs_err = np.abs(pred_base - golden_base).mean(axis=1)  # (N,)
    base_rel_err = (np.abs(pred_base - golden_base) / (np.abs(golden_base) + 1e-8)).mean(axis=1)

    # CPI relative error
    cpi_rel_err = np.abs(cpi_pred - test_cpi) / (np.abs(test_cpi) + 1e-8) * 100  # (N,) in %

    # Bucket by Base magnitude (quartiles)
    percentiles = [0, 25, 50, 75, 100]
    bucket_edges = np.percentile(mean_golden_base, percentiles)

    print(f"\n{'='*70}")
    print(f"BUCKET ANALYSIS: Residuals by Base Magnitude")
    print(f"{'='*70}")
    print(f"{'Bucket':<20} {'Base Range':<18} {'N':>6} {'Base AbsErr':>12} {'Base RelErr':>12} {'CPI MAPE%':>10}")
    print(f"{'-'*78}")

    for i in range(len(percentiles) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        if i == len(percentiles) - 2:
            mask = (mean_golden_base >= lo) & (mean_golden_base <= hi)
        else:
            mask = (mean_golden_base >= lo) & (mean_golden_base < hi)

        n = mask.sum()
        if n == 0:
            continue
        label = f"Q{i+1} ({percentiles[i]}-{percentiles[i+1]}%)"
        b_abs = base_abs_err[mask].mean()
        b_rel = base_rel_err[mask].mean()
        c_mape = cpi_rel_err[mask].mean()

        print(f"  {label:<18} [{lo:.3f}, {hi:.3f}]  {n:>5}  {b_abs:>11.6f}  {b_rel:>11.4f}  {c_mape:>9.2f}%")

    print(f"\n  {'Overall':<18} {'':18} {len(test_cpi):>5}  "
          f"{base_abs_err.mean():>11.6f}  {base_rel_err.mean():>11.4f}  "
          f"{cpi_rel_err.mean():>9.2f}%")

    # ============================================================
    # Per-benchmark breakdown with Base magnitude
    # ============================================================
    print(f"\n{'='*70}")
    print(f"PER-BENCHMARK: Base Magnitude vs CPI Error")
    print(f"{'='*70}")
    print(f"{'Benchmark':<20} {'Mean Base':>10} {'Base RelErr':>12} {'CPI MAPE%':>10}")
    print(f"{'-'*54}")

    bm_stats = []
    benchmarks = sorted(set(test_benchmarks))
    for bm in benchmarks:
        mask = np.array([b == bm for b in test_benchmarks])
        mb = mean_golden_base[mask].mean()
        br = base_rel_err[mask].mean()
        cm = cpi_rel_err[mask].mean()
        bm_stats.append((bm, mb, br, cm))

    # Sort by Mean Base (ascending = small Base first)
    bm_stats.sort(key=lambda x: x[1])
    for bm, mb, br, cm in bm_stats:
        print(f"  {bm:<20} {mb:>9.4f}  {br:>11.4f}  {cm:>9.2f}%")

    # Correlation
    bases = np.array([x[1] for x in bm_stats])
    mapes = np.array([x[3] for x in bm_stats])
    corr = np.corrcoef(bases, mapes)[0, 1]
    print(f"\n  Pearson correlation (Base magnitude vs CPI MAPE): {corr:.3f}")
    if corr < -0.3:
        print(f"  → NEGATIVE correlation: small Base → high MAPE. 1/x amplification confirmed.")
        print(f"  → Fix: change L_task Base dims to relative/CPI-space loss.")
    elif corr > 0.3:
        print(f"  → POSITIVE correlation: large Base → high MAPE. Unexpected.")
    else:
        print(f"  → WEAK correlation: 1/x amplification is NOT the dominant factor.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
