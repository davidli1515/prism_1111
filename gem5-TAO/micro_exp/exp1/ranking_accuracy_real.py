#!/usr/bin/env python3
"""
Realistic ranking accuracy simulation.
1. Load actual CPI data from test set
2. Sample config pairs within same workload (same simpoint/slice)
3. Add prediction noise at different MAPE levels
4. Check if ranking is preserved

Usage:
    python ranking_accuracy_real.py
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")
from config import *
from dataset import load_all_benchmarks, create_exp1_dataloaders

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 600,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

def main():
    # Load real CPI data
    print("Loading data...")
    np.random.seed(SEED)
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, None)

    # Collect all CPI values
    all_cpi = []
    for name, data in all_data.items():
        all_cpi.append(data["cpi_golden"])
    all_cpi = np.concatenate(all_cpi)
    print(f"Total samples: {len(all_cpi)}")
    print(f"CPI range: [{all_cpi.min():.3f}, {all_cpi.max():.3f}], mean={all_cpi.mean():.3f}, std={all_cpi.std():.3f}")

    # Sample random config pairs and compute true CPI difference
    rng = np.random.RandomState(42)
    n_pairs = 500000
    idx_a = rng.randint(0, len(all_cpi), n_pairs)
    idx_b = rng.randint(0, len(all_cpi), n_pairs)

    cpi_a = all_cpi[idx_a]
    cpi_b = all_cpi[idx_b]

    # True relative difference (%)
    true_diff_pct = np.abs(cpi_a - cpi_b) / np.minimum(cpi_a, cpi_b) * 100
    true_rank = cpi_a < cpi_b  # True: A is better (lower CPI)

    print(f"\nTrue CPI difference distribution:")
    for pct in [1, 2, 3, 5, 10]:
        frac = (true_diff_pct < pct).mean() * 100
        print(f"  < {pct}%: {frac:.1f}% of pairs")

    # For each MAPE level, simulate prediction and check ranking
    mape_levels = [1.0, 1.5, 2.0, 2.5, 2.7, 2.9, 3.5, 4.2, 5.0, 7.0]
    diff_bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10)]

    print(f"\nRanking Accuracy by True CPI Difference and Predictor MAPE:")
    print(f"{'Diff range':>12}", end="")
    for m in mape_levels:
        print(f" {m:>5.1f}%", end="")
    print(f"  {'#pairs':>8}")
    print("-" * (12 + len(mape_levels) * 7 + 10))

    results = {}  # (diff_bin, mape) -> accuracy

    for lo, hi in diff_bins:
        mask = (true_diff_pct >= lo) & (true_diff_pct < hi)
        n_in_bin = mask.sum()
        print(f"  {lo}-{hi}%    ", end="")

        for mape in mape_levels:
            # Add relative noise: pred = true * (1 + noise)
            # noise ~ N(0, sigma), sigma chosen so E[|noise|] = mape/100
            sigma = (mape / 100) * np.sqrt(np.pi / 2)  # E[|N(0,s)|] = s*sqrt(2/pi)
            noise_a = rng.normal(0, sigma, n_pairs)
            noise_b = rng.normal(0, sigma, n_pairs)
            pred_a = cpi_a * (1 + noise_a)
            pred_b = cpi_b * (1 + noise_b)
            pred_rank = pred_a < pred_b

            # Ranking accuracy in this bin
            correct = (pred_rank[mask] == true_rank[mask]).mean() * 100
            results[(f"{lo}-{hi}%", mape)] = correct
            print(f" {correct:>5.1f}%", end="")

        print(f"  {n_in_bin:>8}")

    # ============================================================
    # Plot: Line plot for key models
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    highlight = {
        "PRISM (2.9%)":   2.9,
        "Concorde (2.7%)": 2.7,
        "Vanilla (4.2%)": 4.2,
    }
    colors = {
        "PRISM (2.9%)":   "#E67E22",
        "Concorde (2.7%)": "#2E86C1",
        "Vanilla (4.2%)": "#95A5A6",
    }

    # Finer bins for smooth lines
    fine_bins = [(i * 0.5, (i + 1) * 0.5) for i in range(0, 20)]  # 0-10% in 0.5% steps

    for label, mape in highlight.items():
        sigma = (mape / 100) * np.sqrt(np.pi / 2)
        noise_a = rng.normal(0, sigma, n_pairs)
        noise_b = rng.normal(0, sigma, n_pairs)
        pred_a = cpi_a * (1 + noise_a)
        pred_b = cpi_b * (1 + noise_b)
        pred_rank = pred_a < pred_b

        x_vals, y_vals = [], []
        for lo, hi in fine_bins:
            mask = (true_diff_pct >= lo) & (true_diff_pct < hi)
            if mask.sum() < 100:
                continue
            acc = (pred_rank[mask] == true_rank[mask]).mean() * 100
            x_vals.append((lo + hi) / 2)
            y_vals.append(acc)

        ax.plot(x_vals, y_vals, 'o-', label=label, color=colors[label],
                linewidth=2, markersize=4)

    # Reference lines
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    ax.text(9.5, 90.5, "90%", ha='right', fontsize=9, color='gray')
    ax.axhline(y=50, color='red', linestyle=':', alpha=0.3)
    ax.text(9.5, 51, "coin flip", ha='right', fontsize=9, color='red', alpha=0.5)

    # ArchExplorer reference
    ax.axvline(x=1.88, color='green', linestyle='--', alpha=0.4)
    ax.text(2.0, 52, "ArchExplorer\n(1.88%)", fontsize=8, color='green', alpha=0.7)

    ax.set_xlabel("True CPI Difference Between Configs (%)", fontweight='bold')
    ax.set_ylabel("Pairwise Ranking Accuracy (%)", fontweight='bold')
    ax.set_title("Config Ranking Reliability (Real CPI Distribution)", fontweight='bold', pad=10)
    ax.set_ylim(45, 102)
    ax.set_xlim(0, 10)

    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    legend = ax.legend(frameon=True, edgecolor='black', facecolor='white',
                       fancybox=False, framealpha=1.0, fontsize=11,
                       loc='lower right')
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    fig.savefig("ranking_accuracy_real.png", dpi=600, bbox_inches='tight')
    fig.savefig("ranking_accuracy_real.pdf", format='pdf', bbox_inches='tight')
    print("\nSaved: ranking_accuracy_real.png/pdf")

    # Key insight
    print("\n" + "=" * 60)
    print("At ArchExplorer level (1-2% true CPI difference):")
    mask_arch = (true_diff_pct >= 1.0) & (true_diff_pct < 2.0)
    for label, mape in sorted(highlight.items(), key=lambda x: x[1]):
        sigma = (mape / 100) * np.sqrt(np.pi / 2)
        noise_a = rng.normal(0, sigma, n_pairs)
        noise_b = rng.normal(0, sigma, n_pairs)
        pred_a = cpi_a * (1 + noise_a)
        pred_b = cpi_b * (1 + noise_b)
        pred_rank = pred_a < pred_b
        acc = (pred_rank[mask_arch] == true_rank[mask_arch]).mean() * 100
        wrong = 100 - acc
        print(f"  {label:>20}: {acc:.1f}% correct, {wrong:.1f}% wrong")
    print("=" * 60)

    plt.close("all")


if __name__ == "__main__":
    main()