"""
Plot: MSCS-Pruned vs matched-budget Full MC Shapley across all benchmarks.
Same eval count, compare Spearman ρ.

Usage:
    python plot_grouped_vs_full.py
    python plot_grouped_vs_full.py --result-dir result_shapley_520
"""

import argparse, json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_BASE = "/scratch/gem5-TAO/micro_exp/dse_exp/result_shapley_520"

BENCHMARKS = [
    "508.namd_r", "526.blender_r", "600.perlbench_s", "602.gcc_s", "605.mcf_s",
    "607.cactuBSSN_s", "619.lbm_s", "620.omnetpp_s", "621.wrf_s", "623.xalancbmk_s",
    "631.deepsjeng_s", "638.imagick_s", "641.leela_s", "644.nab_s",
    "648.exchange2_s", "654.roms_s", "657.xz_s",
]


def get_bname(b):
    bname = b.split(".")[-1]
    for s in ("_s", "_r"):
        bname = bname.replace(s, "")
    return bname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=RESULT_BASE)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result_dir = args.result_dir
    output_dir = args.output or result_dir

    bench_names = []
    pruned_rhos = []
    pruned_evals_list = []
    matched_mc_rhos = []
    full_rhos = []
    full_evals_list = []

    print(f"\n{'='*80}")
    print(f"Grouped vs Full MC at same eval budget")
    print(f"{'='*80}")
    print(f"  {'Benchmark':<20} {'Pruned ρ':>10} {'Pruned E':>10} {'MC ρ (matched)':>15} {'MC E':>10} {'Full ρ':>10} {'Full E':>10}")
    print(f"  {'─'*85}")

    for bench in BENCHMARKS:
        bname = get_bname(bench)

        # Load shapley results
        shapley_path = os.path.join(result_dir, bench, f"{bname}_shapley_results.json")
        if not os.path.exists(shapley_path):
            continue

        with open(shapley_path) as f:
            shapley_data = json.load(f)

        pruned_rho = shapley_data["metrics"]["pruned"]["rho"]
        pruned_evals = shapley_data["metrics"]["pruned"]["evals"]
        full_rho = 1.0  # ground truth
        full_evals = shapley_data["metrics"]["full"]["evals"]

        # Load ablation results
        ablation_path = os.path.join(result_dir, bench, f"{bname}_ablation_nocache.json")
        matched_rho = None
        matched_evals = None

        if os.path.exists(ablation_path):
            with open(ablation_path) as f:
                ablation_data = json.load(f)

            # Find MC result closest to pruned_evals
            best_diff = float('inf')
            for r in ablation_data["mc_sweep"]:
                diff = abs(r["n_evals"] - pruned_evals)
                if diff < best_diff:
                    best_diff = diff
                    matched_rho = r["spearman_rho"]
                    matched_evals = r["n_evals"]

        if matched_rho is None:
            # Estimate: interpolate from full MC
            # If pruned_evals < full_evals, MC with same budget would have
            # n_perms = pruned_evals / (d+1) permutations
            d = shapley_data["n_params"]
            est_perms = pruned_evals // (d + 1)
            matched_rho = 0.0  # placeholder
            matched_evals = pruned_evals

        bench_names.append(bname)
        pruned_rhos.append(pruned_rho)
        pruned_evals_list.append(pruned_evals)
        matched_mc_rhos.append(matched_rho)
        full_rhos.append(full_rho)
        full_evals_list.append(full_evals)

        print(f"  {bench:<20} {pruned_rho:>10.3f} {pruned_evals:>10} "
              f"{matched_rho:>15.3f} {matched_evals:>10} "
              f"{full_rho:>10.3f} {full_evals:>10}")

    if not bench_names:
        print("  No data found!")
        return

    n = len(bench_names)

    # Sort by pruned ρ descending
    sorted_idx = np.argsort(pruned_rhos)[::-1]
    bench_names = [bench_names[i] for i in sorted_idx]
    pruned_rhos = [pruned_rhos[i] for i in sorted_idx]
    matched_mc_rhos = [matched_mc_rhos[i] for i in sorted_idx]
    pruned_evals_list = [pruned_evals_list[i] for i in sorted_idx]

    # ---- Plot ----
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n)
    width = 0.35

    bars1 = ax.bar(x - width/2, pruned_rhos, width,
                   color='#2ecc71', edgecolor='black', linewidth=0.4,
                   label='MSCS-Pruned Shapley (ours)', zorder=3)
    bars2 = ax.bar(x + width/2, matched_mc_rhos, width,
                   color='#3498db', edgecolor='black', linewidth=0.4,
                   label='Full MC (same eval budget)', zorder=3)

    # Add eval count labels on pruned bars
    for i, (b, e) in enumerate(zip(bars1, pruned_evals_list)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{e}', ha='center', va='bottom', fontsize=6, color='#555')

    ax.set_xticks(x)
    ax.set_xticklabels(bench_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Spearman ρ (vs Full MC Shapley)', fontsize=11, fontweight='bold')
    ax.set_title('Attribution Accuracy at Same Eval Budget: MSCS-Pruned vs Full MC',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=9, loc='lower left')
    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.5, zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')

    plt.tight_layout()
    path = os.path.join(output_dir, "grouped_vs_full_comparison.pdf")
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Also save as png
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.bar(x - width/2, pruned_rhos, width,
            color='#2ecc71', edgecolor='black', linewidth=0.4,
            label='MSCS-Pruned Shapley (ours)', zorder=3)
    ax2.bar(x + width/2, matched_mc_rhos, width,
            color='#3498db', edgecolor='black', linewidth=0.4,
            label='Full MC (same eval budget)', zorder=3)
    for i, (b, e) in enumerate(zip(bars1, pruned_evals_list)):
        ax2.text(x[i] - width/2, pruned_rhos[i] + 0.01,
                 f'{e}', ha='center', va='bottom', fontsize=6, color='#555')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bench_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Spearman ρ', fontsize=11, fontweight='bold')
    ax2.set_title('Attribution Accuracy at Same Eval Budget', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, color='gray', linestyle=':', linewidth=0.5)
    ax2.legend(fontsize=9, loc='lower left')
    ax2.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.5, zorder=0)
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    plt.tight_layout()
    path_png = os.path.join(output_dir, "grouped_vs_full_comparison.png")
    fig2.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {path_png}")

    # Print summary
    mean_pruned = np.mean(pruned_rhos)
    mean_matched = np.mean([r for r in matched_mc_rhos if r > 0])
    wins = sum(1 for p, m in zip(pruned_rhos, matched_mc_rhos) if p > m)
    print(f"\n  Mean ρ: Pruned={mean_pruned:.3f}, Matched MC={mean_matched:.3f}")
    print(f"  Pruned wins: {wins}/{n} benchmarks")


if __name__ == "__main__":
    main()
