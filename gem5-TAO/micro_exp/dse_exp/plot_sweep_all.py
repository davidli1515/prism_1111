"""
Plot aggregated threshold sweep results across all benchmarks.

Usage:
    python plot_sweep_all.py
    python plot_sweep_all.py --result-dir result_shapley
"""

import argparse, json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gmean


RESULT_BASE = "/scratch/gem5-TAO/micro_exp/dse_exp/result_shapley"

BENCHMARKS = [
    "508.namd_r", "526.blender_r", "600.perlbench_s", "602.gcc_s", "605.mcf_s",
    "607.cactuBSSN_s", "619.lbm_s", "620.omnetpp_s", "621.wrf_s", "623.xalancbmk_s",
    "631.deepsjeng_s", "638.imagick_s", "641.leela_s", "644.nab_s",
    "648.exchange2_s", "654.roms_s", "657.xz_s",
]


def get_bname(benchmark):
    bname = benchmark.split(".")[-1]
    for suffix in ("_s", "_r"):
        bname = bname.replace(suffix, "")
    return bname


def load_sweep(benchmark, result_dir):
    bname = get_bname(benchmark)
    path = os.path.join(result_dir, benchmark, f"{bname}_threshold_sweep.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=RESULT_BASE)
    parser.add_argument("--output", default=None, help="Output directory (default: result_dir)")
    args = parser.parse_args()

    output_dir = args.output or args.result_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load all sweep results
    all_data = {}
    for bench in BENCHMARKS:
        data = load_sweep(bench, args.result_dir)
        if data is not None:
            all_data[bench] = data
            print(f"  Loaded {bench}")
        else:
            print(f"  SKIP {bench} (no sweep JSON)")

    if not all_data:
        print("No data found!")
        return

    # Extract thresholds (assume all have same thresholds)
    first = next(iter(all_data.values()))
    thresholds = first["thresholds"]

    # Collect per-benchmark, per-threshold metrics
    rho_matrix = []       # shape: [n_bench, n_thresholds]
    evals_matrix = []
    rho_oaat_list = []
    full_evals_list = []
    bench_names = []

    for bench, data in all_data.items():
        bname = get_bname(bench)
        bench_names.append(bname)

        rhos = [r["spearman_rho"] for r in data["results"]]
        evals = [r["n_evals"] for r in data["results"]]

        rho_matrix.append(rhos)
        evals_matrix.append(evals)
        rho_oaat_list.append(data["reference"]["oaat_rho"])
        full_evals_list.append(data["reference"]["full_evals"])

    rho_matrix = np.array(rho_matrix)       # [n_bench, n_thresh]
    evals_matrix = np.array(evals_matrix)
    n_bench = len(bench_names)

    # Compute aggregates
    rho_mean = np.mean(rho_matrix, axis=0)          # arithmetic mean of ρ
    rho_median = np.median(rho_matrix, axis=0)
    rho_min = np.min(rho_matrix, axis=0)
    rho_max = np.max(rho_matrix, axis=0)
    evals_gmean = gmean(evals_matrix, axis=0)       # geometric mean of evals
    rho_oaat_mean = np.mean(rho_oaat_list)
    full_evals_mean = np.mean(full_evals_list)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"Aggregated Threshold Sweep — {n_bench} benchmarks")
    print(f"{'='*80}")
    print(f"  {'Thresh':>7} {'Mean ρ':>8} {'Med ρ':>8} {'Min ρ':>8} {'Max ρ':>8} {'GMean Evals':>12} {'Speedup':>8}")
    print(f"  {'─'*65}")
    for i, t in enumerate(thresholds):
        speedup = full_evals_mean / evals_gmean[i]
        print(f"  {t:>7.1f} {rho_mean[i]:>8.3f} {rho_median[i]:>8.3f} {rho_min[i]:>8.3f} "
              f"{rho_max[i]:>8.3f} {evals_gmean[i]:>12.0f} {speedup:>7.1f}x")
    print(f"  {'─'*65}")
    print(f"  {'OaaT':>7} {rho_oaat_mean:>8.3f} {'':>8} {'':>8} {'':>8} {n_bench:>12} {'':>8}")
    print(f"  {'Full MC':>7} {'1.000':>8} {'':>8} {'':>8} {'':>8} {full_evals_mean:>12.0f} {'1.0x':>8}")

    # ================================================================
    # Plot 1: Aggregate ρ vs threshold (with band)
    # ================================================================
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = '#2980b9'
    ax1.set_xlabel('Cosine Similarity Threshold (τ)', fontsize=12)
    ax1.set_ylabel('Spearman ρ (vs Full Shapley)', fontsize=12, color=color1)

    # Shaded band: min to max
    ax1.fill_between(thresholds, rho_min, rho_max, color=color1, alpha=0.12, label='Min–Max range')
    # Mean line
    ax1.plot(thresholds, rho_mean, 'o-', color=color1, linewidth=2.5, markersize=8,
             label=f'Mean ρ (n={n_bench})', zorder=5)

    # OaaT reference
    ax1.axhline(rho_oaat_mean, color='#e74c3c', linestyle='--', linewidth=1.5,
                label=f'One-at-a-time mean ρ={rho_oaat_mean:.3f}')
    ax1.axhline(1.0, color='#27ae60', linestyle=':', linewidth=1, alpha=0.4)

    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(min(rho_min.min(), rho_oaat_mean) - 0.05, 1.02)

    # Secondary axis: evaluation count
    color2 = '#e67e22'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Evaluations (geometric mean)', fontsize=12, color=color2)
    ax2.plot(thresholds, evals_gmean, 's--', color=color2, linewidth=1.5,
             markersize=6, alpha=0.7, label='Pruned evals (gmean)')
    ax2.axhline(full_evals_mean, color='#95a5a6', linestyle=':', linewidth=1, alpha=0.5,
                label=f'Full MC evals ({full_evals_mean:.0f})')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yscale('log')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)

    ax1.set_title(f'Shapley Pruning: Accuracy vs Cost ({n_bench} benchmarks)', fontsize=13, pad=10)
    ax1.set_xticks(thresholds)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "shapley_sweep_aggregate.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # ================================================================
    # Plot 2: Per-benchmark ρ at each threshold (heatmap)
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(rho_matrix, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'{t:.1f}' for t in thresholds], fontsize=9)
    ax.set_yticks(range(n_bench))
    ax.set_yticklabels(bench_names, fontsize=8)
    ax.set_xlabel('Cosine Similarity Threshold (τ)', fontsize=11)
    ax.set_ylabel('Benchmark', fontsize=11)
    ax.set_title(f'Spearman ρ: Pruned vs Full Shapley ({n_bench} benchmarks)', fontsize=13)

    # Annotate cells
    for i in range(n_bench):
        for j in range(len(thresholds)):
            val = rho_matrix[i, j]
            color = 'white' if val < 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='Spearman ρ', shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, "shapley_sweep_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # ================================================================
    # Plot 3: Per-benchmark ρ at default threshold (bar chart)
    # ================================================================
    default_idx = thresholds.index(0.3) if 0.3 in thresholds else 2
    rho_at_default = rho_matrix[:, default_idx]
    rho_oaat_arr = np.array(rho_oaat_list)

    sorted_idx = np.argsort(rho_at_default)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_bench)
    width = 0.35

    ax.bar(x - width/2, rho_at_default[sorted_idx], width, color='#2ecc71',
           label=f'MSCS-Pruned (τ={thresholds[default_idx]})', edgecolor='white')
    ax.bar(x + width/2, rho_oaat_arr[sorted_idx], width, color='#e74c3c',
           label='One-at-a-time (ΔCPI)', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([bench_names[i] for i in sorted_idx], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Spearman ρ (vs Full Shapley)', fontsize=11)
    ax.set_title(f'Attribution Accuracy per Benchmark (τ={thresholds[default_idx]})', fontsize=13)
    ax.legend(fontsize=9)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "shapley_per_benchmark.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Save aggregate JSON
    agg_path = os.path.join(output_dir, "shapley_sweep_aggregate.json")
    agg = {
        "n_benchmarks": n_bench,
        "benchmarks": bench_names,
        "thresholds": thresholds,
        "rho_mean": rho_mean.tolist(),
        "rho_median": rho_median.tolist(),
        "rho_min": rho_min.tolist(),
        "rho_max": rho_max.tolist(),
        "evals_gmean": evals_gmean.tolist(),
        "rho_oaat_mean": float(rho_oaat_mean),
        "full_evals_mean": float(full_evals_mean),
        "per_benchmark": {
            bench_names[i]: {
                "rho_by_threshold": rho_matrix[i].tolist(),
                "evals_by_threshold": evals_matrix[i].tolist(),
                "rho_oaat": rho_oaat_list[i],
            }
            for i in range(n_bench)
        }
    }
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"  Saved: {agg_path}")


if __name__ == "__main__":
    main()
