"""
Base Prediction Bias Diagnostic

Checks if trained model's Base predictions have systematic bias
that can be corrected with a simple post-hoc calibration.

Four checks:
1. Per-dim residual mean vs std (bias/std ratio)
2. pred ≈ a·true + b regression (systematic scaling/shift?)
3. 4-dim residual correlation matrix (same-direction bias?)
4. Per-benchmark mean SIGNED relative residual (same sign → calibratable)

Usage:
    cd /path/to/gem5/micro_exp/exp4
    python diagnose_base_bias.py --ckpt <path_to_checkpoint.pt>
"""

import sys, os, argparse
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


def find_extra_indices(d18, d40):
    set_18k = set()
    for i in range(len(d18["gem_features"])):
        set_18k.add(d18["gem_features"][i].tobytes())
    return np.array([i for i in range(len(d40["gem_features"]))
                     if d40["gem_features"][i].tobytes() not in set_18k], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load data — replicate exact normalizer from extra_unlabeled training
    print("Loading data...")
    data_18k = load_all_benchmarks(DATASET_PATHS_18K["gem"], DATASET_PATHS_18K["concord"], BENCHMARK_NAMES)
    data_40k = load_all_benchmarks(DATASET_PATHS_40K["gem"], DATASET_PATHS_40K["concord"], BENCHMARK_NAMES)

    rng = np.random.RandomState(SEED)
    train_gem_all = []
    test_gem, test_mscs, test_cpi, test_wmin = [], [], [], []
    test_bench = []

    for name in sorted(data_18k.keys()):
        d18 = data_18k[name]
        d40 = data_40k[name]
        n = d18["n_samples"]
        indices = rng.permutation(n)
        split = int(n * TRAIN_RATIO)
        tr_idx, te_idx = indices[:split], indices[split:]

        # ALL train data for normalizer (18k train + 40k extra)
        train_gem_all.append(d18["gem_features"][tr_idx])
        extra_idx = find_extra_indices(d18, d40)
        if len(extra_idx) > 0:
            train_gem_all.append(d40["gem_features"][extra_idx])

        # Test
        test_gem.append(d18["gem_features"][te_idx])
        test_mscs.append(d18["mscs_target"][te_idx])
        test_cpi.append(d18["cpi_golden"][te_idx])
        test_wmin.append(d18["w_min"][te_idx])
        test_bench.extend([name] * len(te_idx))

    gem_norm = FeatureNormalizer()
    gem_norm.fit(np.concatenate(train_gem_all))

    test_gem  = gem_norm.transform(np.concatenate(test_gem))
    test_mscs = np.concatenate(test_mscs)
    test_cpi  = np.concatenate(test_cpi)
    test_wmin = np.concatenate(test_wmin)
    test_bench = np.array(test_bench)

    print(f"Test: {len(test_cpi)} samples, {len(set(test_bench))} benchmarks")

    # Load model
    model = GEMModel(GEM_INPUT_DIM, [512, 256], DROPOUT).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"Loaded: {args.ckpt}")

    # Forward
    with torch.no_grad():
        mscs_pred = model(torch.from_numpy(test_gem).to(device)).cpu().numpy()

    base_pred = mscs_pred[:, BASE_INDICES]   # (N, 4)
    base_true = test_mscs[:, BASE_INDICES]   # (N, 4)
    residual  = base_pred - base_true        # (N, 4) signed

    # CPI from predicted Base (same as eval)
    cpi_pred = (1.0 / (base_pred * test_wmin[:, None] + 1e-8)).mean(axis=1)
    cpi_err_signed = (cpi_pred - test_cpi) / (np.abs(test_cpi) + 1e-8)  # signed relative
    cpi_err_abs    = np.abs(cpi_err_signed) * 100  # MAPE %

    stage_names = ["Decode", "Dispatch", "Issue", "Commit"]

    # ============================================================
    # Check 1: Per-dim residual mean vs std
    # ============================================================
    print(f"\n{'='*70}")
    print(f"CHECK 1: Per-dim Residual Mean vs Std (bias/std ratio)")
    print(f"{'='*70}")
    print(f"  {'Stage':<10} {'Mean(bias)':>12} {'Std':>12} {'|bias/std|':>12} {'Verdict':>10}")
    print(f"  {'-'*58}")
    for i, name in enumerate(stage_names):
        m = residual[:, i].mean()
        s = residual[:, i].std()
        ratio = abs(m) / (s + 1e-12)
        verdict = "BIASED" if ratio > 0.1 else "clean"
        print(f"  {name:<10} {m:>12.6f} {s:>12.6f} {ratio:>12.4f} {verdict:>10}")

    overall_bias = residual.mean()
    print(f"\n  Overall mean bias: {overall_bias:.6f}")
    if abs(overall_bias) > 0.001:
        print(f"  → Systematic shift detected. Global scale/shift calibration may help.")
    else:
        print(f"  → No significant overall bias.")

    # ============================================================
    # Check 2: pred ≈ a·true + b regression
    # ============================================================
    print(f"\n{'='*70}")
    print(f"CHECK 2: Linear Regression pred = a·true + b")
    print(f"{'='*70}")
    for i, name in enumerate(stage_names):
        p, t = base_pred[:, i], base_true[:, i]
        # Simple OLS
        A = np.vstack([t, np.ones(len(t))]).T
        result = np.linalg.lstsq(A, p, rcond=None)
        a, b = result[0]
        print(f"  {name:<10}: a={a:.6f}  b={b:+.6f}  (ideal: a=1, b=0)")

    # All 4 dims pooled
    p_all, t_all = base_pred.flatten(), base_true.flatten()
    A = np.vstack([t_all, np.ones(len(t_all))]).T
    result = np.linalg.lstsq(A, p_all, rcond=None)
    a_global, b_global = result[0]
    print(f"\n  Global:    a={a_global:.6f}  b={b_global:+.6f}")
    if abs(a_global - 1.0) > 0.01 or abs(b_global) > 0.005:
        print(f"  → Systematic scaling/shift detected!")
        print(f"    Calibration: Base_calibrated = (Base_pred - {b_global:.6f}) / {a_global:.6f}")
    else:
        print(f"  → Near-perfect (a≈1, b≈0). No calibration needed.")

    # ============================================================
    # Check 3: 4-dim residual correlation matrix
    # ============================================================
    print(f"\n{'='*70}")
    print(f"CHECK 3: Residual Correlation Matrix (4 stages)")
    print(f"{'='*70}")
    corr = np.corrcoef(residual.T)  # (4, 4)
    print(f"  {'':>10}", end="")
    for name in stage_names:
        print(f"  {name:>10}", end="")
    print()
    for i, name in enumerate(stage_names):
        print(f"  {name:>10}", end="")
        for j in range(4):
            print(f"  {corr[i,j]:>10.4f}", end="")
        print()

    avg_corr = (corr.sum() - 4) / 12  # off-diagonal average
    print(f"\n  Average off-diagonal correlation: {avg_corr:.4f}")
    if avg_corr > 0.8:
        print(f"  → HIGHLY correlated residuals. A single global factor can capture most bias.")
    elif avg_corr > 0.5:
        print(f"  → Moderately correlated. Global calibration will help partially.")
    else:
        print(f"  → Weakly correlated. Per-stage calibration needed (if at all).")

    # ============================================================
    # Check 4: Per-benchmark SIGNED relative residual (THE KEY CHECK)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"CHECK 4: Per-benchmark Signed CPI Relative Error")
    print(f"{'='*70}")
    print(f"  {'Benchmark':<20} {'Signed Err%':>12} {'|MAPE|%':>10} {'Direction':>10}")
    print(f"  {'-'*54}")

    bm_signed = {}
    bm_mape = {}
    benchmarks = sorted(set(test_bench))
    n_pos, n_neg = 0, 0

    for bm in benchmarks:
        mask = test_bench == bm
        signed = cpi_err_signed[mask].mean() * 100
        unsigned = cpi_err_abs[mask].mean()
        direction = "OVER" if signed > 0 else "UNDER"
        bm_signed[bm] = signed
        bm_mape[bm] = unsigned
        if signed > 0:
            n_pos += 1
        else:
            n_neg += 1
        print(f"  {bm:<20} {signed:>+11.2f}% {unsigned:>9.2f}% {direction:>10}")

    print(f"\n  Over-predicting CPI: {n_pos}/{len(benchmarks)} benchmarks")
    print(f"  Under-predicting CPI: {n_neg}/{len(benchmarks)} benchmarks")

    if n_pos >= len(benchmarks) * 0.8 or n_neg >= len(benchmarks) * 0.8:
        dominant = "over" if n_pos > n_neg else "under"
        print(f"\n  → STRONG same-sign bias ({dominant}-predicting).")
        print(f"    Global calibration WILL help. Proceed to calibration.")
    elif n_pos >= len(benchmarks) * 0.6 or n_neg >= len(benchmarks) * 0.6:
        print(f"\n  → Moderate same-sign tendency.")
        print(f"    Global calibration may help slightly.")
    else:
        print(f"\n  → Mixed signs ({n_pos} over / {n_neg} under).")
        print(f"    Global calibration will cancel out. NOT recommended.")

    # ============================================================
    # Quick simulation: what if we apply global scale?
    # ============================================================
    print(f"\n{'='*70}")
    print(f"SIMULATION: Global Scale Calibration (on test, for reference only)")
    print(f"{'='*70}")

    # Find optimal scale: minimize mean(|scale*cpi_pred - cpi_true| / cpi_true)
    # Grid search
    best_scale, best_mape = 1.0, cpi_err_abs.mean()
    for s in np.linspace(0.90, 1.10, 201):
        scaled_cpi = cpi_pred * s
        m = (np.abs(scaled_cpi - test_cpi) / (np.abs(test_cpi) + 1e-8) * 100).mean()
        if m < best_mape:
            best_scale, best_mape = s, m

    print(f"  Before calibration: CPI MAPE = {cpi_err_abs.mean():.4f}%")
    print(f"  Optimal scale: {best_scale:.4f}")
    print(f"  After calibration:  CPI MAPE = {best_mape:.4f}%")
    print(f"  Improvement: {cpi_err_abs.mean() - best_mape:.4f}%")

    if best_mape < cpi_err_abs.mean() - 0.05:
        print(f"\n  → Calibration gives >{cpi_err_abs.mean() - best_mape:.2f}% improvement. Worth doing properly with val/test split.")
    else:
        print(f"\n  → Calibration improvement < 0.05%. Not worth the complexity.")

    # Per-benchmark after calibration
    print(f"\n  Per-benchmark after optimal scale ({best_scale:.4f}):")
    scaled_cpi = cpi_pred * best_scale
    print(f"  {'Benchmark':<20} {'Before':>8} {'After':>8} {'Δ':>8}")
    for bm in benchmarks:
        mask = test_bench == bm
        before = cpi_err_abs[mask].mean()
        after = (np.abs(scaled_cpi[mask] - test_cpi[mask]) / (np.abs(test_cpi[mask]) + 1e-8) * 100).mean()
        delta = after - before
        print(f"  {bm:<20} {before:>7.2f}% {after:>7.2f}% {delta:>+7.2f}%")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
