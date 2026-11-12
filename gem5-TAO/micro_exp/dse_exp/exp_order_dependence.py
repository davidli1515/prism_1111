"""
Order-Dependence Experiment: MSCS Jacobian explains WHY sequential ablation
gives different results depending on parameter ordering.

Concorde (Fig 15) shows: Cache→LQ vs LQ→Cache gives different attributions.
Shapley resolves this at O(2^n) cost but only gives scalar importance.
MSCS Jacobian shows the root cause: both parameters compete on the same
bottleneck type at the same pipeline stage.

Experiment:
  Step 1: ARM N1 baseline → Jacobian → cache & LQ overlap on BP
  Step 2: Increase cache → Jacobian → LQ's BP effect drops (cache already fixed it)
  Step 3: Increase LQ    → Jacobian → cache's BP effect drops (LQ already fixed it)

Usage:
    python exp_order_dependence.py --benchmark 605.mcf_s
"""

import argparse
import os
import sys
import numpy as np
import torch

np.seterr(divide='ignore', invalid='ignore')

EXP1_DIR = "/scratch/gem5-TAO/micro_exp/exp1"
DSE_DIR  = "/scratch/gem5-TAO/micro_exp/dse_exp"
sys.path.insert(0, EXP1_DIR)
sys.path.insert(0, DSE_DIR)

from model import GEMModel
from dataset import FeatureNormalizer
from dse_infer import DseEngine
from config_dse import (
    GRADIENT_PARAMS, PARAM_TO_GD_INDEX, ALL_PARAMS,
    compute_w_min, get_param_values,
    GEM_DATA_ARCH_START, GEM_DATA_ARCH_END,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GEM_INPUT_DIM   = 4681
GEM_HIDDEN_DIMS = [512, 256]
GEM_WEIGHT_PATH = "/scratch/gem5-TAO/micro_exp/exp1/results/exp1_v2/exp1_512_256/gem_full.pt"
NORMALIZER_CACHE = os.path.join(DSE_DIR, "normalizer_cache")
OUTPUT_DIR = os.path.join(DSE_DIR, "results")

PRECOMPUTE_DIRS = {
    "605.mcf_s":       os.path.join(DSE_DIR, "precompute/605.mcf_s"),
    "623.xalancbmk_s": os.path.join(DSE_DIR, "precompute/623.xalancbmk_s"),
}

STAGE_NAMES = ['Decode', 'Dispatch', 'Issue', 'Commit']
FP_INDICES = [1, 5, 9, 13]
BP_INDICES = [2, 6, 10, 14]
SP_INDICES = [3, 7, 11, 15]

ARM_N1_CONFIG = {
    "fetchWidth": 4, "decodeWidth": 4, "dispatchWidth": 4,
    "commitWidth": 8, "wbWidth": 8,
    "numROBEntries": 128, "numIQEntries": 32,
    "LQEntries": 12, "SQEntries": 18, "fetchBufferSize": 1,
    "IntALU": 3, "IntMultDiv": 1, "FP_ALU": 2, "FP_MultDiv": 1, "RdWrPort": 2,
    "l1_size": 64, "l2_size": 1024,
    "l1d_assoc": 4, "l1i_assoc": 4, "l2_assoc": 8, "bp_type": 0,
}


def predict_mscs(model, normalizer, dse_engine, workload, config):
    all_feats = dse_engine.get_features(config, workload=workload)
    batch = np.stack([f[:GEM_INPUT_DIM] for f in all_feats])
    x = torch.from_numpy(normalizer.transform(batch)).float()
    with torch.no_grad():
        return model(x).mean(dim=0).cpu().numpy()


def compute_jacobian_row(model, normalizer, dse_engine, workload, config, param_name):
    """Compute ∂MSCS/∂param for a single parameter, averaged over all slices."""
    all_feats = dse_engine.get_features(config, workload=workload)
    a_s, a_e = GEM_DATA_ARCH_START, GEM_DATA_ARCH_END
    norm_std = normalizer.std[a_s:a_e]
    local_idx = PARAM_TO_GD_INDEX[param_name] - GEM_DATA_ARCH_START

    avg_grad = np.zeros(16)
    n = len(all_feats)

    for si in range(n):
        feat = all_feats[si][:GEM_INPUT_DIM].copy()
        normalized = normalizer.transform(feat.reshape(1, -1)).flatten()
        x = torch.from_numpy(normalized).float()

        for oi in range(16):
            x_arch = x[a_s:a_e].detach().requires_grad_(True)
            x_full = torch.cat([x[:a_s].detach(), x_arch, x[a_e:].detach()])
            out = model(x_full.unsqueeze(0))
            out[0, oi].backward()
            avg_grad[oi] += x_arch.grad.detach().cpu().numpy()[local_idx]

    avg_grad /= n
    avg_grad /= (norm_std[local_idx] + 1e-8)
    return avg_grad


def print_bp_profile(label, jac_row):
    """Print BP gradient across 4 stages."""
    bp = jac_row[BP_INDICES]
    print(f"    {label:>12s}: ", end="")
    for s, name in enumerate(STAGE_NAMES):
        print(f" {name[:3]}={bp[s]:+.5f}", end="")
    r = abs(bp.sum()) / (np.abs(bp).sum() + 1e-10)
    print(f"  | sum={bp.sum():+.5f} | r={r:.2f}")


def plot_comparison(results, benchmark, output_path):
    """
    Plot: BP gradient profile across stages for cache and LQ
    under three conditions (baseline, cache increased, LQ increased).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    conditions = [
        ("ARM N1 Baseline\n(cache=64KB, LQ=12)", "baseline"),
        ("After cache↑\n(cache=256KB, LQ=12)", "cache_up"),
        ("After LQ↑\n(cache=64KB, LQ=48)", "lq_up"),
    ]

    x = np.arange(4)
    w = 0.35

    for ax_i, (title, key) in enumerate(conditions):
        cache_bp = results[key]["l1_size"][BP_INDICES]
        lq_bp = results[key]["LQEntries"][BP_INDICES]

        ax = axes[ax_i]
        bars1 = ax.bar(x - w/2, -cache_bp, w, label='cache (L1 size)',
                        color='#00bcd4', alpha=0.85, edgecolor='white')
        bars2 = ax.bar(x + w/2, -lq_bp, w, label='LQ entries',
                        color='#ff9800', alpha=0.85, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(STAGE_NAMES, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.5)

        if ax_i == 0:
            ax.set_ylabel('BP Pressure Reduction\n(-∂BP/∂θ)', fontsize=10)
            ax.legend(fontsize=9, loc='upper left')

        # Annotate cancellation ratio
        for name, bp, offset in [("cache", cache_bp, -w/2), ("LQ", lq_bp, w/2)]:
            r = abs(bp.sum()) / (np.abs(bp).sum() + 1e-10)
            ax.text(3 + offset, max(-bp) * 0.5, f'r={r:.2f}',
                    ha='center', fontsize=7, color='gray')

    fig.suptitle(f'Order-Dependence Explained by MSCS Jacobian — {benchmark}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def plot_summary_bar(results, benchmark, output_path):
    """
    Summary: total BP reduction of cache and LQ under 3 conditions.
    Shows how adjusting one param reduces the other's marginal effect.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = ["baseline", "cache_up", "lq_up"]
    labels = ["ARM N1\nBaseline", "After\ncache↑", "After\nLQ↑"]

    cache_totals = []
    lq_totals = []
    for key in conditions:
        cache_bp = results[key]["l1_size"][BP_INDICES]
        lq_bp = results[key]["LQEntries"][BP_INDICES]
        cache_totals.append(np.maximum(0, -cache_bp).sum())
        lq_totals.append(np.maximum(0, -lq_bp).sum())

    x = np.arange(3)
    w = 0.35

    ax.bar(x - w/2, cache_totals, w, label='cache (L1 size)',
           color='#00bcd4', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, lq_totals, w, label='LQ entries',
           color='#ff9800', alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Total BP Reduction\nΣ max(0, -∂BP_s/∂θ)', fontsize=11)
    ax.set_title(f'Marginal Effect Shift — {benchmark}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate percentage drops
    for i in [1, 2]:
        if cache_totals[0] > 0:
            pct = (cache_totals[i] / cache_totals[0] - 1) * 100
            ax.text(i - w/2, cache_totals[i] + 0.001, f'{pct:+.0f}%',
                    ha='center', fontsize=8, color='#00838f', fontweight='bold')
        if lq_totals[0] > 0:
            pct = (lq_totals[i] / lq_totals[0] - 1) * 100
            ax.text(i + w/2, lq_totals[i] + 0.001, f'{pct:+.0f}%',
                    ha='center', fontsize=8, color='#e65100', fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="605.mcf_s",
                        choices=list(PRECOMPUTE_DIRS.keys()))
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    benchmark = args.benchmark
    bname = {"605.mcf_s": "mcf", "623.xalancbmk_s": "xalan"}[benchmark]
    output_dir = os.path.join(args.output_dir, benchmark)
    os.makedirs(output_dir, exist_ok=True)

    # Load
    model = GEMModel(input_dim=GEM_INPUT_DIM, hidden_dims=GEM_HIDDEN_DIMS, dropout=0.1)
    model.load_state_dict(torch.load(GEM_WEIGHT_PATH, map_location="cpu"))
    model.eval()

    gem_norm = FeatureNormalizer()
    gem_norm.mean = np.load(os.path.join(NORMALIZER_CACHE, "gem_mean.npy"))
    gem_norm.std  = np.load(os.path.join(NORMALIZER_CACHE, "gem_std.npy"))

    dse_engine = DseEngine(list(PRECOMPUTE_DIRS.values()))

    params_of_interest = ["l1_size", "LQEntries"]

    # ============================================================
    # Condition 1: ARM N1 baseline (cache=64KB, LQ=12)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"Order-Dependence Experiment — {benchmark}")
    print(f"{'='*70}")

    config_base = ARM_N1_CONFIG.copy()
    mscs_base = predict_mscs(model, gem_norm, dse_engine, benchmark, config_base)
    w_min = compute_w_min(config_base)
    cpi_base = 1.0 / (mscs_base[12] * w_min + 1e-8)

    print(f"\n[1/3] ARM N1 Baseline: cache={config_base['l1_size']}KB, "
          f"LQ={config_base['LQEntries']}, CPI={cpi_base:.4f}")

    results = {"baseline": {}, "cache_up": {}, "lq_up": {}}

    for pname in params_of_interest:
        jac = compute_jacobian_row(model, gem_norm, dse_engine, benchmark,
                                    config_base, pname)
        results["baseline"][pname] = jac
        print_bp_profile(pname, jac)

    # Overlap analysis
    cache_bp = results["baseline"]["l1_size"][BP_INDICES]
    lq_bp = results["baseline"]["LQEntries"][BP_INDICES]
    # Cosine similarity of BP profiles
    dot = np.dot(cache_bp, lq_bp)
    norm_c = np.linalg.norm(cache_bp) + 1e-10
    norm_l = np.linalg.norm(lq_bp) + 1e-10
    cos_sim = dot / (norm_c * norm_l)
    print(f"  BP profile cosine similarity: {cos_sim:.3f}")
    print(f"  → {'HIGH overlap: competing on same BP stages' if cos_sim > 0.7 else 'Low overlap'}")

    # ============================================================
    # Condition 2: Increase cache (64KB → 256KB), keep LQ=12
    # ============================================================
    config_cache_up = ARM_N1_CONFIG.copy()
    config_cache_up["l1_size"] = 256

    mscs_cu = predict_mscs(model, gem_norm, dse_engine, benchmark, config_cache_up)
    cpi_cu = 1.0 / (mscs_cu[12] * compute_w_min(config_cache_up) + 1e-8)

    print(f"\n[2/3] After cache↑: cache={config_cache_up['l1_size']}KB, "
          f"LQ={config_cache_up['LQEntries']}, CPI={cpi_cu:.4f} "
          f"(ΔCPI={cpi_cu-cpi_base:+.4f})")

    for pname in params_of_interest:
        jac = compute_jacobian_row(model, gem_norm, dse_engine, benchmark,
                                    config_cache_up, pname)
        results["cache_up"][pname] = jac
        print_bp_profile(pname, jac)

    # Show marginal effect shift
    lq_before = np.maximum(0, -results["baseline"]["LQEntries"][BP_INDICES]).sum()
    lq_after = np.maximum(0, -results["cache_up"]["LQEntries"][BP_INDICES]).sum()
    print(f"  LQ total BP reduction: {lq_before:.5f} → {lq_after:.5f} "
          f"({(lq_after/lq_before-1)*100:+.0f}%)")

    # ============================================================
    # Condition 3: Increase LQ (12 → 48), keep cache=64KB
    # ============================================================
    config_lq_up = ARM_N1_CONFIG.copy()
    config_lq_up["LQEntries"] = 48

    mscs_lu = predict_mscs(model, gem_norm, dse_engine, benchmark, config_lq_up)
    cpi_lu = 1.0 / (mscs_lu[12] * compute_w_min(config_lq_up) + 1e-8)

    print(f"\n[3/3] After LQ↑: cache={config_lq_up['l1_size']}KB, "
          f"LQ={config_lq_up['LQEntries']}, CPI={cpi_lu:.4f} "
          f"(ΔCPI={cpi_lu-cpi_base:+.4f})")

    for pname in params_of_interest:
        jac = compute_jacobian_row(model, gem_norm, dse_engine, benchmark,
                                    config_lq_up, pname)
        results["lq_up"][pname] = jac
        print_bp_profile(pname, jac)

    cache_before = np.maximum(0, -results["baseline"]["l1_size"][BP_INDICES]).sum()
    cache_after = np.maximum(0, -results["lq_up"]["l1_size"][BP_INDICES]).sum()
    print(f"  Cache total BP reduction: {cache_before:.5f} → {cache_after:.5f} "
          f"({(cache_after/cache_before-1)*100:+.0f}%)")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Baseline:   cache BP_red={cache_before:.5f}, LQ BP_red={lq_before:.5f}")
    print(f"  After cache↑: LQ BP_red drops to {lq_after:.5f} "
          f"({(lq_after/lq_before-1)*100:+.0f}%) — cache already solved the BP")
    print(f"  After LQ↑:    cache BP_red drops to {cache_after:.5f} "
          f"({(cache_after/cache_before-1)*100:+.0f}%) — LQ already solved the BP")
    print(f"\n  This explains Concorde's order-dependence:")
    print(f"  Cache→LQ: cache gets full credit, LQ's marginal effect is small")
    print(f"  LQ→Cache: LQ gets full credit, cache's marginal effect is small")
    print(f"  MSCS Jacobian sees this IN ONE PASS — no need for O(2^n) Shapley")
    print(f"{'='*70}")

    # Plots
    plot_comparison(results, bname, os.path.join(output_dir, f"{bname}_order_dependence.png"))
    plot_summary_bar(results, bname, os.path.join(output_dir, f"{bname}_marginal_shift.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
