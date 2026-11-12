"""
MSCS-Attributed DSE Decision Visualization

Shows reduction × r for each parameter:
  reduction = Σ_stage max(0, -∂T/∂θ) × param_range  (how much it reduces pressure)
  r = |Σ_stage g| / Σ_stage |g|                      (stage consistency)
  reduction × r = reliable pressure reduction         (what DSE should act on)

Only shows FP and BP (SP omitted — negligible for these workloads).
Green border = DSE would INCREASE this param (high reduction×r for dominant type)
Red border   = DSE would RECLAIM this param (total effect ≈ 0)

Usage:
    python visualize_cancellation.py --benchmark 605.mcf_s
    python visualize_cancellation.py --benchmark 623.xalancbmk_s
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
from matplotlib.patches import FancyBboxPatch

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

RECLAIM_THRESHOLD = 0.05  # total effect below this → reclaim


def compute_full_jacobian(model, normalizer, dse_engine, workload, config):
    """Backprop Jacobian ∂MSCS/∂(arch_params), all slices."""
    all_feats = dse_engine.get_features(config, workload=workload)
    a_s, a_e = GEM_DATA_ARCH_START, GEM_DATA_ARCH_END
    norm_std = normalizer.std[a_s:a_e]
    n = len(all_feats)

    avg_jac = np.zeros((a_e - a_s, 16))
    for si in range(n):
        feat = all_feats[si][:GEM_INPUT_DIM].copy()
        normalized = normalizer.transform(feat.reshape(1, -1)).flatten()
        x = torch.from_numpy(normalized).float()
        for oi in range(16):
            x_arch = x[a_s:a_e].detach().requires_grad_(True)
            x_full = torch.cat([x[:a_s].detach(), x_arch, x[a_e:].detach()])
            out = model(x_full.unsqueeze(0))
            out[0, oi].backward()
            avg_jac[:, oi] += x_arch.grad.detach().cpu().numpy()
    avg_jac /= n
    for j in range(16):
        avg_jac[:, j] /= (norm_std + 1e-8)

    param_names = []
    jacobian = []
    param_ranges = []
    for name in GRADIENT_PARAMS:
        if name not in PARAM_TO_GD_INDEX:
            continue
        local_idx = PARAM_TO_GD_INDEX[name] - GEM_DATA_ARCH_START
        vals = get_param_values(name)
        param_names.append(name)
        jacobian.append(avg_jac[local_idx, :])
        param_ranges.append(max(vals) - min(vals))
    return param_names, np.array(jacobian), np.array(param_ranges)


def compute_metrics(param_names, jacobian, param_ranges):
    """
    For each param, compute:
      reduction_fp/bp = Σ max(0, -∂T/∂θ) × range
      r_fp/bp = |Σg| / Σ|g|
      effective_fp/bp = reduction × r
      total = reduction_fp + reduction_bp + reduction_sp (for reclaim check)
    """
    n = len(param_names)
    metrics = []
    for i in range(n):
        r = param_ranges[i]
        result = {"name": param_names[i]}

        for tname, indices in [("FP", FP_INDICES), ("BP", BP_INDICES), ("SP", SP_INDICES)]:
            g = jacobian[i, indices]
            reduction = np.maximum(0, -g).sum() * r
            abs_sum = np.abs(g).sum()
            ratio = abs(g.sum()) / (abs_sum + 1e-10)
            effective = reduction * ratio
            result[f"red_{tname}"] = reduction
            result[f"r_{tname}"] = ratio
            result[f"eff_{tname}"] = effective

        result["total"] = result["red_FP"] + result["red_BP"] + result["red_SP"]
        metrics.append(result)
    return metrics


def plot_cancellation_heatmap(metrics, benchmark, output_dir):
    """
    Heatmap: params × (FP, BP), color = cancellation ratio r.
    Green border on INCREASE candidate, red text for RECLAIM params.
    Only shows FP and BP. Filters weak-signal params.
    """
    max_total = max(m["total"] for m in metrics)
    strong = [m for m in metrics if m["total"] >= max_total * 0.05]
    weak = [m for m in metrics if m["total"] < max_total * 0.05]

    # Sort by eff_FP + eff_BP descending
    strong.sort(key=lambda m: m["eff_FP"] + m["eff_BP"], reverse=True)

    # Find dominant and increase candidate
    fp_sum = sum(m["eff_FP"] for m in strong)
    bp_sum = sum(m["eff_BP"] for m in strong)
    dominant = "BP" if bp_sum > fp_sum else "FP"
    dom_key = f"eff_{dominant}"
    increase_name = max(strong, key=lambda m: m[dom_key])["name"]

    names = [m["name"] for m in strong]
    n = len(names)
    data = np.array([[m["r_FP"], m["r_BP"]] for m in strong])

    fig, ax = plt.subplots(figsize=(6, max(5, n * 0.45)))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=1, interpolation='nearest')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['FP', 'BP'], fontsize=13, fontweight='bold')
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=10)

    # Annotate r values + eff values
    for i in range(n):
        for j, tname in enumerate(["FP", "BP"]):
            r = data[i, j]
            eff = strong[i][f"eff_{tname}"]
            color = 'white' if r < 0.3 or r > 0.8 else 'black'
            ax.text(j, i, f'r={r:.2f}\n({eff:.3f})',
                    ha='center', va='center', fontsize=8, color=color,
                    fontweight='bold')

    # Green border on increase candidate
    for i, m in enumerate(strong):
        if m["name"] == increase_name:
            j = 1 if dominant == "BP" else 0
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  linewidth=3, edgecolor='#27ae60',
                                  facecolor='none', zorder=10)
            ax.add_patch(rect)
            ax.text(j, i + 0.42, '↑', ha='center', fontsize=10,
                    color='#27ae60', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Cancellation Ratio r\n(1=consistent, 0=stage transfer)', fontsize=9)

    title = f'Stage Consistency — {benchmark} (ARM N1)'
    if weak:
        weak_names = ", ".join(m["name"] for m in weak)
        title += f'\nRECLAIM (total≈0): {weak_names}'
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{benchmark}_cancellation_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_effective_reduction(metrics, benchmark, output_dir):
    """
    Grouped bar chart: FP and BP effective reduction (reduction × r) per parameter.
    Green border = increase candidate, Red border = reclaim candidate.
    """
    # Filter: only params with total > 5% of max
    max_total = max(m["total"] for m in metrics)
    strong = [m for m in metrics if m["total"] >= max_total * 0.05]
    weak = [m for m in metrics if m["total"] < max_total * 0.05]

    # Sort by total effective (FP+BP)
    strong.sort(key=lambda m: m["eff_FP"] + m["eff_BP"], reverse=True)

    # Find dominant type
    fp_avg = np.mean([m["eff_FP"] for m in strong])
    bp_avg = np.mean([m["eff_BP"] for m in strong])
    dominant = "BP" if bp_avg > fp_avg else "FP"

    # Top increase candidate: highest eff for dominant type
    dom_key = f"eff_{dominant}"
    increase_param = max(strong, key=lambda m: m[dom_key])["name"]

    names = [m["name"] for m in strong]
    fp_eff = [m["eff_FP"] for m in strong]
    bp_eff = [m["eff_BP"] for m in strong]
    n = len(names)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n)
    w = 0.35

    bars_fp = ax.bar(x - w/2, fp_eff, w, label='FP reduction × r',
                      color='#e74c3c', alpha=0.85, edgecolor='white')
    bars_bp = ax.bar(x + w/2, bp_eff, w, label='BP reduction × r',
                      color='#3498db', alpha=0.85, edgecolor='white')

    # Annotate r values on top of bars
    for i, m in enumerate(strong):
        if m["eff_FP"] > 0.001:
            ax.text(x[i] - w/2, fp_eff[i] + 0.002, f'r={m["r_FP"]:.2f}',
                    ha='center', va='bottom', fontsize=7, color='#c0392b')
        if m["eff_BP"] > 0.001:
            ax.text(x[i] + w/2, bp_eff[i] + 0.002, f'r={m["r_BP"]:.2f}',
                    ha='center', va='bottom', fontsize=7, color='#2471a3')

    # Green border on increase candidate
    for i, m in enumerate(strong):
        if m["name"] == increase_param:
            offset = w/2 if dominant == "BP" else -w/2
            bar_height = m[dom_key]
            rect = plt.Rectangle((x[i] + offset - w/2 - 0.03, -0.003),
                                  w + 0.06, bar_height + 0.01,
                                  linewidth=2.5, edgecolor='#27ae60',
                                  facecolor='none', zorder=10)
            ax.add_patch(rect)
            ax.text(x[i] + offset, bar_height + 0.015, '↑ INCREASE',
                    ha='center', fontsize=8, color='#27ae60', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Effective Pressure Reduction\n(reduction × stage consistency r)', fontsize=10)
    ax.set_title(f'MSCS-Attributed DSE Decision — {benchmark} (ARM N1)\n'
                 f'Dominant bottleneck: {dominant}', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add reclaim annotation at bottom
    if weak:
        weak_names = [m["name"] for m in weak]
        ax.text(0.02, 0.95,
                f'RECLAIM (total≈0): {", ".join(weak_names)}',
                transform=ax.transAxes, fontsize=9, va='top',
                color='#c0392b', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, f"{benchmark}_effective_reduction.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    return increase_param, weak


def plot_stage_breakdown(param_names, jacobian, param_ranges, metrics,
                          benchmark, output_dir, top_k=8):
    """
    For top-K params by effective reduction: show 4-stage signed gradients
    for FP and BP side by side.
    """
    # Sort by total effective
    sorted_metrics = sorted(metrics, key=lambda m: m["eff_FP"] + m["eff_BP"], reverse=True)
    name_to_idx = {n: i for i, n in enumerate(param_names)}

    top = sorted_metrics[:top_k]
    ncols = (top_k + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(3.5 * ncols, 7), sharey=True)
    axes = axes.flatten()

    for ax_i, m in enumerate(top):
        name = m["name"]
        pi = name_to_idx[name]
        ax = axes[ax_i]

        fp_g = jacobian[pi, FP_INDICES]
        bp_g = jacobian[pi, BP_INDICES]

        xx = np.arange(4)
        ww = 0.35
        ax.bar(xx - ww/2, fp_g * param_ranges[pi], ww, label='FP',
               color='#e74c3c', alpha=0.8)
        ax.bar(xx + ww/2, bp_g * param_ranges[pi], ww, label='BP',
               color='#3498db', alpha=0.8)

        ax.set_xticks(xx)
        ax.set_xticklabels(['Dec', 'Dis', 'Iss', 'Com'], fontsize=8)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(f'{name}\nFP r={m["r_FP"]:.2f} | BP r={m["r_BP"]:.2f}',
                     fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        if ax_i == 0:
            ax.legend(fontsize=7, loc='best')
        if ax_i % ncols == 0:
            ax.set_ylabel('∂pressure/∂θ × range', fontsize=9)

    for ax_i in range(len(top), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle(f'Stage-wise Gradient Breakdown — {benchmark} (ARM N1)\n'
                 f'Negative = reduces pressure | Positive = increases pressure',
                 fontsize=11)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{benchmark}_stage_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


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

    print(f"\n{'='*60}")
    print(f"Effective Reduction Analysis — {benchmark}")
    print(f"{'='*60}")

    # Compute Jacobian
    print(f"\nComputing Jacobian (all slices)...")
    param_names, jacobian, param_ranges = compute_full_jacobian(
        model, gem_norm, dse_engine, benchmark, ARM_N1_CONFIG)
    print(f"  {len(param_names)} params × 16 MSCS components")

    # Compute metrics
    metrics = compute_metrics(param_names, jacobian, param_ranges)

    # Print table
    print(f"\n  {'Param':<18s} {'red_FP':>8s} {'r_FP':>6s} {'eff_FP':>8s} "
          f"{'red_BP':>8s} {'r_BP':>6s} {'eff_BP':>8s} {'total':>8s} {'action'}")
    print(f"  {'-'*90}")

    max_total = max(m["total"] for m in metrics)
    for m in sorted(metrics, key=lambda x: x["eff_FP"] + x["eff_BP"], reverse=True):
        if m["total"] < max_total * 0.05:
            action = "RECLAIM"
        elif m["eff_FP"] + m["eff_BP"] > 0:
            action = "candidate"
        else:
            action = "-"
        print(f"  {m['name']:<18s} {m['red_FP']:>8.4f} {m['r_FP']:>6.2f} {m['eff_FP']:>8.4f} "
              f"{m['red_BP']:>8.4f} {m['r_BP']:>6.2f} {m['eff_BP']:>8.4f} "
              f"{m['total']:>8.4f} {action}")

    # Plots
    print(f"\nGenerating plots...")
    plot_cancellation_heatmap(metrics, bname, output_dir)
    increase_param, weak = plot_effective_reduction(metrics, bname, output_dir)
    plot_stage_breakdown(param_names, jacobian, param_ranges, metrics, bname, output_dir)

    # Summary
    print(f"\n  DSE Decision:")
    print(f"    INCREASE: {increase_param} (highest effective reduction for dominant type)")
    print(f"    RECLAIM:  {[m['name'] for m in weak]}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()