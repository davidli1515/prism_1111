"""
Plot Exp2 Sweep: per-benchmark MAPE vs target train ratio

Usage:
    python plot_exp2_sweep.py --dir /path/to/gem5/micro_exp/exp2/results_sweep
"""

import os
import json
import glob
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all_sweeps(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "sweep_*.json")))
    sweeps = {}
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            sweeps[data["target"]] = data
    return sweeps


def plot_per_benchmark(sweeps, save_dir):
    """One subplot per benchmark: MAPE vs train ratio for 3 variants."""
    bms = sorted(sweeps.keys())
    n = len(bms)
    cols = 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)

    for i, bm in enumerate(bms):
        ax = axes[i // cols][i % cols]
        data = sweeps[bm]["ratios"]

        ratios = sorted(data.keys(), key=float)
        r_vals = [float(r) * 100 for r in ratios]  # percent

        gf = [data[r]["gem_full"] for r in ratios]
        gn = [data[r]["gem_nod3"] for r in ratios]
        cn = [data[r]["concorde"] for r in ratios]

        ax.plot(r_vals, gf, "o-", color="#2196F3", label="GEM+D3", markersize=4, linewidth=1.5)
        ax.plot(r_vals, gn, "s-", color="#FF9800", label="GEM-noD3", markersize=4, linewidth=1.5)
        ax.plot(r_vals, cn, "^-", color="#4CAF50", label="Concorde", markersize=4, linewidth=1.5)

        ax.set_title(bm.replace("_", " "), fontsize=9)
        ax.set_xlabel("Target train %", fontsize=8)
        ax.set_ylabel("MAPE %", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide empty subplots
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    plt.suptitle("OOD Generalization: MAPE vs. Target Train Data Ratio", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "exp2_sweep_per_benchmark.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Per-benchmark plot saved to {path}")


def plot_average(sweeps, save_dir):
    """Single plot: average MAPE across all benchmarks vs train ratio."""
    bms = sorted(sweeps.keys())
    # Get common ratios
    sample_ratios = sorted(sweeps[bms[0]]["ratios"].keys(), key=float)

    fig, ax = plt.subplots(figsize=(7, 5))

    for key, label, color, marker in [
        ("gem_full",  "GEM (full)",  "#2196F3", "o"),
        ("gem_nod3",  "GEM (no D3)", "#FF9800", "s"),
        ("concorde",  "Concorde",    "#4CAF50", "^"),
    ]:
        r_vals = []
        m_vals = []
        for r in sample_ratios:
            r_pct = float(r) * 100
            mapes = [sweeps[bm]["ratios"][r][key] for bm in bms if r in sweeps[bm]["ratios"]]
            if mapes:
                r_vals.append(r_pct)
                m_vals.append(np.mean(mapes))

        ax.plot(r_vals, m_vals, f"{marker}-", color=color, label=label,
                markersize=6, linewidth=2)

    ax.set_xlabel("Target Benchmark Train Data (%)", fontsize=12)
    ax.set_ylabel("Average MAPE (%)", fontsize=12)
    ax.set_title("OOD Generalization vs. Available Target Data", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(save_dir, "exp2_sweep_average.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Average plot saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Exp2 Sweep")
    parser.add_argument("--dir", type=str,
                        default="/path/to/gem5/micro_exp/exp2/results_sweep")
    args = parser.parse_args()

    sweeps = load_all_sweeps(args.dir)
    print(f"Loaded {len(sweeps)} benchmarks")

    if not sweeps:
        print("No results found!")
        return

    plot_per_benchmark(sweeps, args.dir)
    plot_average(sweeps, args.dir)
    print("Done!")


if __name__ == "__main__":
    main()