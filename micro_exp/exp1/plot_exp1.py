#!/usr/bin/env python3
"""
Plot Exp1 results: per-benchmark and overall MAPE comparison.
Publication-quality style (Nature/Science aesthetics).

Usage:
    python plot_exp1.py \
        --data 408k \
        --concorde results/exp1_v2/exp1_512_256/concorde.json \
        --gem_full results/exp1_v2/exp1_512_256/gem_full.json \
        --output results/exp1_v2/exp1_plot.png
"""

import argparse, json, sys
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

VARIANT_LABELS = {
    "concorde":      "Concorde",
    "gem_full":      "PRISM Full",
    "gem_const":     "PRISM (L_const only)",
    "gem_vanilla":   "PRISM (L_task only)",
    "gem_nod3":      "PRISM-noD3 Full",
    "gem_mono":      "PRISM (L_mono only)",
}

VARIANT_COLORS = {
    "concorde":      "#E74C3C",
    "gem_full":      "#2ECC71",
    "gem_const":     "#3498DB",
    "gem_vanilla":   "#95A5A6",
    "gem_nod3":      "#F39C12",
    "gem_mono":      "#9B59B6",
}

VARIANT_HATCHES = {
    "concorde":      "",
    "gem_full":      "",
    "gem_const":     "//",
    "gem_vanilla":   "xx",
    "gem_nod3":      "..",
    "gem_mono":      "\\\\",
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


def load_variant(path):
    if path is None:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if "overall_mape" in data:
            return data["overall_mape"], data["per_benchmark"]
        elif "overall" in data:
            for key in data["overall"]:
                return data["overall"][key], data["per_benchmark"][key]
        return None
    except FileNotFoundError:
        print(f"  WARNING: File not found: {path}")
        return None
    except Exception as e:
        print(f"  WARNING: Error loading {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Plot Exp1 Results")
    parser.add_argument("--data", type=str, required=True,
                        help="Data size label: '183k' or '408k'")
    parser.add_argument("--concorde", type=str, default=None)
    parser.add_argument("--gem_full", type=str, default=None)
    parser.add_argument("--gem_const", type=str, default=None)
    parser.add_argument("--gem_vanilla", type=str, default=None)
    parser.add_argument("--gem_nod3", type=str, default=None)
    parser.add_argument("--gem_mono", type=str, default=None)
    parser.add_argument("--output", type=str, default="exp1_plot.png")
    args = parser.parse_args()

    # Load all provided variants
    variant_keys = ["concorde", "gem_full", "gem_const", "gem_vanilla", "gem_nod3", "gem_mono"]
    variants = {}
    for key in variant_keys:
        path = getattr(args, key)
        result = load_variant(path)
        if result is not None:
            variants[key] = result
            print(f"  Loaded {VARIANT_LABELS[key]}: {result[0]:.2f}%")

    if not variants:
        print("ERROR: No valid variant files provided."); sys.exit(1)

    print(f"\nExp1 Results ({args.data} data)")
    print("=" * 60)

    all_benchmarks = set()
    for key, (overall, per_bm) in variants.items():
        all_benchmarks.update(per_bm.keys())
    benchmarks = sorted(all_benchmarks)

    active_keys = [k for k in variant_keys if k in variants]

    # ============================================================
    # Table
    # ============================================================
    header = f"{'Benchmark':<22}"
    for key in active_keys:
        header += f" {VARIANT_LABELS[key]:>18}"
    header += f" {'Best':>12}"
    print(header)
    print("-" * len(header))

    for bm in benchmarks:
        row = f"{bm:<22}"
        scores = {}
        for key in active_keys:
            _, per_bm = variants[key]
            val = per_bm.get(bm, None)
            if val is not None:
                row += f" {val:>17.2f}%"
                scores[key] = val
            else:
                row += f" {'N/A':>18}"
        if scores:
            best_key = min(scores, key=scores.get)
            row += f" {VARIANT_LABELS[best_key]:>12}"
        print(row)

    print("-" * len(header))
    row = f"{'Overall':<22}"
    overall_scores = {}
    for key in active_keys:
        overall, _ = variants[key]
        row += f" {overall:>17.2f}%"
        overall_scores[key] = overall
    if overall_scores:
        best_key = min(overall_scores, key=overall_scores.get)
        row += f" {VARIANT_LABELS[best_key]:>12}"
    print(row)
    print()

    # ============================================================
    # Plot 1: Per-benchmark grouped bar chart
    # ============================================================
    n_variants = len(active_keys)
    n_benchmarks = len(benchmarks)

    fig, ax = plt.subplots(figsize=(max(14, n_benchmarks * 0.9), 5.5))
    bar_width = 0.8 / n_variants
    x = np.arange(n_benchmarks)

    for i, key in enumerate(active_keys):
        _, per_bm = variants[key]
        vals = [per_bm.get(bm, 0) for bm in benchmarks]
        offset = (i - n_variants / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width * 0.9,
               label=VARIANT_LABELS[key],
               color=VARIANT_COLORS[key],
               hatch=VARIANT_HATCHES[key],
               edgecolor='black', linewidth=0.4, zorder=3)

    ax.set_xlabel("Benchmark", fontweight='bold')
    ax.set_ylabel("MAPE (%)", fontweight='bold')
    ax.set_title(f"Exp1: In-Distribution CPI Prediction ({args.data} data)",
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    short_names = [b.replace("_r", "").replace("_s", "").split(".")[-1] for b in benchmarks]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    _style_ax(ax)

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels,
                       loc='upper center', bbox_to_anchor=(0.5, -0.2),
                       ncol=min(n_variants, 5), frameon=True, edgecolor='black',
                       facecolor='white', fancybox=False, shadow=False,
                       framealpha=1.0, columnspacing=1.2, handletextpad=0.5,
                       borderpad=0.8)
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    _save(fig, args.output)

    # ============================================================
    # Plot 2: Overall comparison bar chart
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(max(6, n_variants * 1.5), 5.5))
    labels_list = [VARIANT_LABELS[k] for k in active_keys]
    values = [variants[k][0] for k in active_keys]
    colors = [VARIANT_COLORS[k] for k in active_keys]

    bars = ax2.bar(range(len(labels_list)), values, width=0.6,
                   color=colors, edgecolor='black', linewidth=0.4, zorder=3)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}%", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

    ax2.set_ylabel("MAPE (%)", fontweight='bold')
    ax2.set_title(f"Exp1: Overall MAPE ({args.data} data)",
                  fontweight='bold', pad=15)
    ax2.set_xticks(range(len(labels_list)))
    ax2.set_xticklabels(labels_list, rotation=20, ha="right", fontsize=10)
    ax2.set_ylim(0, max(values) * 1.3)
    _style_ax(ax2)

    plt.tight_layout()
    overall_path = args.output.replace(".png", "_overall.png")
    _save(fig2, overall_path)

    plt.close("all")
    print("Done!")


if __name__ == "__main__":
    main()