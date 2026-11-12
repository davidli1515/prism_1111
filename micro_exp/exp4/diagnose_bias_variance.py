"""
Bias²/Variance Decomposition across multiple seeds.

Given K models trained with different seeds:
  mean_pred = (1/K) Σ pred_k
  bias²     = (mean_pred - true)²           → systematic error all models share
  variance  = (1/K) Σ (pred_k - mean_pred)² → random variation across seeds

MSE = bias² + variance (exact decomposition)

If bias² >> variance → ensemble useless, residual head promising
If variance >> bias² → ensemble effective, residual head unnecessary

Usage:
    python diagnose_bias_variance.py --ckpts ckpt1.pt ckpt2.pt ckpt3.pt ...
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
    parser.add_argument("--ckpts", type=str, nargs="+", required=True,
                        help="Paths to model checkpoints (different seeds)")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    K = len(args.ckpts)
    print(f"Bias²/Variance decomposition with {K} models")

    # Load data (same normalizer as extra training)
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

        train_gem_all.append(d18["gem_features"][tr_idx])
        extra_idx = find_extra_indices(d18, d40)
        if len(extra_idx) > 0:
            train_gem_all.append(d40["gem_features"][extra_idx])

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

    test_gem_t  = torch.from_numpy(test_gem).to(device)
    test_wmin_t = torch.from_numpy(test_wmin)

    N = len(test_cpi)
    print(f"Test: {N} samples, {len(set(test_bench))} benchmarks")

    # Forward pass for each model
    all_cpi_preds = []
    all_base_preds = []

    for i, ckpt in enumerate(args.ckpts):
        model = GEMModel(GEM_INPUT_DIM, [512, 256], DROPOUT).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        with torch.no_grad():
            mscs_pred = model(test_gem_t).cpu()
            cpi_pred = compute_cpi_mean(mscs_pred, test_wmin_t).numpy()
            base_pred = mscs_pred[:, BASE_INDICES].numpy()

        mape = (np.abs(cpi_pred - test_cpi) / (np.abs(test_cpi) + 1e-8) * 100).mean()
        all_cpi_preds.append(cpi_pred)
        all_base_preds.append(base_pred)
        print(f"  Model {i+1} ({os.path.basename(ckpt)}): MAPE = {mape:.2f}%")

    all_cpi_preds  = np.stack(all_cpi_preds)   # (K, N)
    all_base_preds = np.stack(all_base_preds)   # (K, N, 4)

    # ============================================================
    # CPI-level bias²/variance decomposition
    # ============================================================
    mean_cpi   = all_cpi_preds.mean(axis=0)              # (N,)
    bias2_cpi  = (mean_cpi - test_cpi) ** 2               # (N,)
    var_cpi    = ((all_cpi_preds - mean_cpi[None, :]) ** 2).mean(axis=0)  # (N,)
    mse_cpi    = bias2_cpi + var_cpi

    # Also compute relative versions
    rel_bias2  = (bias2_cpi / (test_cpi ** 2 + 1e-16))
    rel_var    = (var_cpi / (test_cpi ** 2 + 1e-16))

    print(f"\n{'='*70}")
    print(f"CPI-LEVEL BIAS²/VARIANCE DECOMPOSITION ({K} models)")
    print(f"{'='*70}")
    print(f"  MSE       = {mse_cpi.mean():.8f}")
    print(f"  Bias²     = {bias2_cpi.mean():.8f}  ({bias2_cpi.mean()/mse_cpi.mean()*100:.1f}% of MSE)")
    print(f"  Variance  = {var_cpi.mean():.8f}  ({var_cpi.mean()/mse_cpi.mean()*100:.1f}% of MSE)")
    print(f"")
    print(f"  Relative (÷ CPI²):")
    print(f"  Rel Bias² = {rel_bias2.mean():.8f}  → √ = {np.sqrt(rel_bias2.mean())*100:.2f}% (≈ irreducible MAPE)")
    print(f"  Rel Var   = {rel_var.mean():.8f}  → √ = {np.sqrt(rel_var.mean())*100:.2f}% (reducible by ensemble)")

    # ============================================================
    # Ensemble result
    # ============================================================
    ensemble_mape = (np.abs(mean_cpi - test_cpi) / (np.abs(test_cpi) + 1e-8) * 100).mean()
    individual_mapes = [(np.abs(all_cpi_preds[k] - test_cpi) / (np.abs(test_cpi) + 1e-8) * 100).mean()
                        for k in range(K)]

    print(f"\n{'='*70}")
    print(f"ENSEMBLE RESULT")
    print(f"{'='*70}")
    print(f"  Individual MAPEs: {', '.join(f'{m:.2f}%' for m in individual_mapes)}")
    print(f"  Mean individual:  {np.mean(individual_mapes):.2f}%")
    print(f"  Ensemble ({K} models): {ensemble_mape:.2f}%")
    print(f"  Improvement:      {np.mean(individual_mapes) - ensemble_mape:.2f}%")

    # ============================================================
    # Per-benchmark decomposition
    # ============================================================
    print(f"\n{'='*70}")
    print(f"PER-BENCHMARK DECOMPOSITION")
    print(f"{'='*70}")
    print(f"  {'Benchmark':<20} {'Bias²%':>8} {'Var%':>8} {'B²/(B²+V)':>10} {'Ensemble':>10} {'Best Single':>12}")
    print(f"  {'-'*70}")

    benchmarks = sorted(set(test_bench))
    for bm in benchmarks:
        mask = test_bench == bm
        b2 = bias2_cpi[mask].mean()
        v  = var_cpi[mask].mean()
        total = b2 + v
        ratio = b2 / (total + 1e-16)

        # Ensemble MAPE for this benchmark
        ens_mape_bm = (np.abs(mean_cpi[mask] - test_cpi[mask]) / (np.abs(test_cpi[mask]) + 1e-8) * 100).mean()
        # Best single model for this benchmark
        best_bm = min(
            (np.abs(all_cpi_preds[k][mask] - test_cpi[mask]) / (np.abs(test_cpi[mask]) + 1e-8) * 100).mean()
            for k in range(K)
        )

        print(f"  {bm:<20} {np.sqrt(b2/test_cpi[mask].mean()**2)*100:>7.2f}% "
              f"{np.sqrt(v/test_cpi[mask].mean()**2)*100:>7.2f}% "
              f"{ratio:>9.1%} "
              f"{ens_mape_bm:>9.2f}% "
              f"{best_bm:>11.2f}%")

    # ============================================================
    # Base-level decomposition
    # ============================================================
    mean_base = all_base_preds.mean(axis=0)  # (N, 4)
    base_true = test_mscs[:, BASE_INDICES]
    bias2_base = ((mean_base - base_true) ** 2).mean()
    var_base   = ((all_base_preds - mean_base[None, :, :]) ** 2).mean()

    print(f"\n{'='*70}")
    print(f"BASE-LEVEL DECOMPOSITION")
    print(f"{'='*70}")
    print(f"  Base MSE      = {(bias2_base + var_base):.8f}")
    print(f"  Base Bias²    = {bias2_base:.8f}  ({bias2_base/(bias2_base+var_base)*100:.1f}%)")
    print(f"  Base Variance = {var_base:.8f}  ({var_base/(bias2_base+var_base)*100:.1f}%)")

    # ============================================================
    # Verdict
    # ============================================================
    bias_frac = bias2_cpi.mean() / (mse_cpi.mean() + 1e-16)
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    if bias_frac > 0.8:
        print(f"  Bias² dominates ({bias_frac:.0%} of MSE).")
        print(f"  → Ensemble will give minimal benefit (~{np.mean(individual_mapes)-ensemble_mape:.2f}% actual).")
        print(f"  → Residual correction head is the right tool.")
        print(f"  → Irreducible MAPE floor ≈ {np.sqrt(rel_bias2.mean())*100:.2f}%.")
    elif bias_frac > 0.5:
        print(f"  Mixed ({bias_frac:.0%} bias, {1-bias_frac:.0%} variance).")
        print(f"  → Both ensemble and residual head can help.")
    else:
        print(f"  Variance dominates ({1-bias_frac:.0%} of MSE).")
        print(f"  → Ensemble is the right tool.")
        print(f"  → Expected ensemble gain: ~{np.mean(individual_mapes)-ensemble_mape:.2f}%.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
