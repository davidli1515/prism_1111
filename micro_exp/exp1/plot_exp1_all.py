#!/usr/bin/env python3
"""
Plot all Exp1 results across configurations.
Publication-quality style (Nature/Science aesthetics).

Usage:
    python plot_exp1_all.py --base /path/to/gem5/micro_exp/exp1/results/exp1_v2
"""

import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Publication-quality aesthetics
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
})

# ============================================================
# Configure paths here — edit these to point to your results
# Each config: (display_label, {variant: path_to_json})
# Set path to None to skip a variant
# ============================================================
BASE = "/path/to/gem5/micro_exp/exp1/results"

CONFIGS = [
    ("[256,128] 408k", {
        "concorde":    f"{BASE}/exp1_v2_run2/exp1_256_128/concorde.json",
        "gem_full":    f"{BASE}/exp1_v2_run2/exp1_256_128/gem_full.json",
        "gem_const":   f"{BASE}/exp1_v3_74/exp1_256_128/gem_const.json",
        "gem_nod3":    f"{BASE}/exp1_v2_run2/exp1_256_128/gem_nod3.json",
        "gem_vanilla": f"{BASE}/exp1_v2_run2/exp1_256_128/gem_vanilla.json",
    }),
    ("[512,256] 408k", {
        "concorde":    f"{BASE}/exp1_v2_run2/exp1_512_256/concorde.json",
        "gem_full":    f"{BASE}/exp1_v2_run2/exp1_512_256/gem_full.json",
        "gem_const":   f"{BASE}/exp1_v2_run2/exp1_512_256/gem_const.json",
        "gem_nod3":    f"{BASE}/exp1_v2_run2/exp1_512_256/gem_nod3.json",
        "gem_vanilla": f"{BASE}/exp1_v2_run2/exp1_512_256/gem_vanilla.json",
    }),
    ("[256,128] 183k", {
        "concorde":    f"{BASE}/exp1_v2_run2/exp1_256_128_183k/concorde.json",
        "gem_full":    f"{BASE}/exp1_v3_run2/exp1_256_128_183k/gem_full.json",
        "gem_const":   f"{BASE}/exp1_v2_76/exp1_256_128_183k/gem_const.json",
        "gem_nod3":    f"{BASE}/exp1_v2_76/exp1_256_128_183k/gem_nod3.json",
        "gem_vanilla": f"{BASE}/exp1_v2_run2/exp1_256_128_183k/gem_vanilla.json",
    }),
    ("[512,256] 183k", {
        "concorde":    f"{BASE}/exp1_v2_run2/exp1_512_256_183k/concorde.json",
        "gem_full":    f"{BASE}/exp1_v3_run2/exp1_512_256_183k/gem_full.json",
        "gem_const":   f"{BASE}/exp1_v2_76/exp1_512_256_183k/gem_const.json",
        "gem_nod3":    f"{BASE}/exp1_v2_76/exp1_512_256_183k/gem_nod3.json",
        "gem_vanilla": f"{BASE}/exp1_v2_run2/exp1_512_256_183k/gem_vanilla.json",
    }),
]

OUTPUT = f"{BASE}/exp1_v2_run2/exp1_all.png"

VARIANTS = ["concorde", "gem_full", "gem_const", "gem_nod3", "gem_vanilla"]
VARIANT_LABELS = {
    "concorde":    "Concorde",
    "gem_full":    "PRISM Full",
    "gem_const":   "PRISM (L_const)",
    "gem_nod3":    "PRISM-withD3",
    "gem_vanilla": "PRISM (vanilla)",
}

VARIANT_COLORS = {
    "concorde":    "#2E86C1",
    "gem_full":    "#E67E22",
    "gem_const":   "#F39C12",
    "gem_nod3":    "#D35400",
    "gem_vanilla": "#95A5A6",
}


def _style_ax(ax):
    """Apply publication styling to an axes."""
    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)


def _save(fig, path):
    """Save as PNG and PDF."""
    fig.savefig(path, dpi=600, bbox_inches='tight')
    pdf_path = path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=600)
    print(f"  Saved: {path}")
    print(f"  Saved: {pdf_path}")


def load_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
        if "overall_mape" in data:
            return data["overall_mape"], data.get("per_benchmark", {})
        elif "overall" in data:
            for k in data["overall"]:
                return data["overall"][k], data["per_benchmark"][k]
        return None, None
    except:
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=OUTPUT)
    args = parser.parse_args()

    # Load all results
    results = {}
    per_bm = {}
    for label, paths in CONFIGS:
        for v in VARIANTS:
            path = paths.get(v)
            if path is None:
                continue
            overall, pbm = load_json(path)
            if overall is not None:
                results[(label, v)] = overall
                per_bm[(label, v)] = pbm
                print(f"  {label:>20} {VARIANT_LABELS[v]:>15}: {overall:.2f}%")
            else:
                print(f"  {label:>20} {VARIANT_LABELS[v]:>15}: (missing: {path})")

    if not results:
        print("ERROR: No results found."); sys.exit(1)

    # ============================================================
    # Table
    # ============================================================
    active_keys = [k for k in VARIANTS if any((l, k) in results for l, _ in CONFIGS)]
    active_configs = [l for l, _ in CONFIGS if any((l, v) in results for v in VARIANTS)]

    print(f"\n{'='*90}")
    print(f"{'Config':<22}", end="")
    for v in active_keys:
        print(f" {VARIANT_LABELS[v]:>14}", end="")
    print()
    print("-" * 90)
    for cfg in active_configs:
        print(f"{cfg:<22}", end="")
        for v in active_keys:
            val = results.get((cfg, v))
            if val is not None:
                print(f" {val:>13.2f}%", end="")
            else:
                print(f" {'—':>14}", end="")
        print()
    print(f"{'='*90}\n")

    # ============================================================
    # Plot 1: Grouped bar chart — Overall MAPE by config × variant
    # ============================================================
    n_configs = len(active_configs)
    n_variants = len(active_keys)

    fig, ax = plt.subplots(figsize=(max(10, n_configs * 2.5), 5.5))
    bar_width = 0.75 / n_variants
    x = np.arange(n_configs)

    for i, v in enumerate(active_keys):
        vals = [results.get((cfg, v), 0) for cfg in active_configs]
        mask = [results.get((cfg, v)) is not None for cfg in active_configs]
        offset = (i - n_variants / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width * 0.85,
                      label=VARIANT_LABELS[v], color=VARIANT_COLORS.get(v, "#333"),
                      edgecolor='black', linewidth=0.4, zorder=3)
        for j, (bar, val, m) in enumerate(zip(bars, vals, mask)):
            if m and val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("MAPE (%)", fontweight='bold')
    ax.set_title("Exp1: In-Distribution Overall MAPE", fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(active_configs, fontsize=10)
    ax.set_ylim(0, max(v for v in results.values()) * 1.3)
    _style_ax(ax)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels,
                       loc='upper center', bbox_to_anchor=(0.5, -0.1),
                       ncol=min(n_variants, 5), frameon=True, edgecolor='black',
                       facecolor='white', fancybox=False, shadow=False,
                       framealpha=1.0, columnspacing=1.2, handletextpad=0.5,
                       borderpad=0.8)
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    _save(fig, args.output)

    # ============================================================
    # Plot 2: Data scaling — 183k vs 408k for each variant
    # ============================================================
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, hidden_label in enumerate(["[256,128]", "[512,256]"]):
        ax2 = axes[ax_idx]
        cfg_183k = f"{hidden_label} 183k"
        cfg_408k = f"{hidden_label} 408k"

        present_variants = [v for v in VARIANTS
                            if (cfg_183k, v) in results or (cfg_408k, v) in results]
        x_pos = np.arange(len(present_variants))
        w = 0.35

        vals_183k = [results.get((cfg_183k, v), 0) for v in present_variants]
        vals_408k = [results.get((cfg_408k, v), 0) for v in present_variants]
        mask_183k = [(cfg_183k, v) in results for v in present_variants]
        mask_408k = [(cfg_408k, v) in results for v in present_variants]

        bars1 = ax2.bar(x_pos - w/2, vals_183k, w, label="183k",
                        color="#3498DB", alpha=0.85, edgecolor='black',
                        linewidth=0.4, zorder=3)
        bars2 = ax2.bar(x_pos + w/2, vals_408k, w, label="408k",
                        color="#E74C3C", alpha=0.85, edgecolor='black',
                        linewidth=0.4, zorder=3)

        for bars, vals, mask in [(bars1, vals_183k, mask_183k),
                                  (bars2, vals_408k, mask_408k)]:
            for bar, val, m in zip(bars, vals, mask):
                if m and val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.05,
                             f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        ax2.set_ylabel("MAPE (%)", fontweight='bold')
        ax2.set_title(f"Data Scaling: {hidden_label}", fontweight='bold', pad=15)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([VARIANT_LABELS[v] for v in present_variants],
                            rotation=20, ha="right", fontsize=9)
        _style_ax(ax2)

        legend = ax2.legend(frameon=True, edgecolor='black', facecolor='white',
                            fancybox=False, framealpha=1.0, fontsize=9)
        legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    scaling_path = args.output.replace(".png", "_scaling.png")
    _save(fig2, scaling_path)

    # ============================================================
    # Plot 3: Model capacity — [256,128] vs [512,256] for each variant
    # ============================================================
    fig3, axes = plt.subplots(1, 2, figsize=(14, 4.0))

    for ax_idx, data_label in enumerate(["183k", "408k"]):
        ax3 = axes[ax_idx]
        cfg_256 = f"[256,128] {data_label}"
        cfg_512 = f"[512,256] {data_label}"

        present_variants = [v for v in VARIANTS
                            if (cfg_256, v) in results or (cfg_512, v) in results]
        x_pos = np.arange(len(present_variants))
        w = 0.35

        vals_256 = [results.get((cfg_256, v), 0) for v in present_variants]
        vals_512 = [results.get((cfg_512, v), 0) for v in present_variants]
        mask_256 = [(cfg_256, v) in results for v in present_variants]
        mask_512 = [(cfg_512, v) in results for v in present_variants]

        bars1 = ax3.bar(x_pos - w/2, vals_256, w, label="[256,128]",
                        color="#F39C12", alpha=0.85, edgecolor='black',
                        linewidth=0.4, zorder=3)
        bars2 = ax3.bar(x_pos + w/2, vals_512, w, label="[512,256]",
                        color="#2ECC71", alpha=0.85, edgecolor='black',
                        linewidth=0.4, zorder=3)

        for bars, vals, mask in [(bars1, vals_256, mask_256),
                                  (bars2, vals_512, mask_512)]:
            for bar, val, m in zip(bars, vals, mask):
                if m and val > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.05,
                             f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        ax3.set_ylabel("MAPE (%)", fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([VARIANT_LABELS[v] for v in present_variants],
                            rotation=20, ha="right", fontsize=11, fontweight='bold')
        _style_ax(ax3)

        legend = ax3.legend(frameon=True, edgecolor='black', facecolor='white',
                            fancybox=False, framealpha=1.0, fontsize=11,
                            borderpad=0.8, handletextpad=0.5)
        legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    capacity_path = args.output.replace(".png", "_capacity.png")
    _save(fig3, capacity_path)

    plt.close("all")
    print("\nDone!")


if __name__ == "__main__":
    main()