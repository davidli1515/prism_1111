"""
Sweep cosine similarity thresholds for Shapley pruning.
Reuses cached Full MC Shapley from existing JSON results.
Only re-runs MSCS-pruned Shapley at each threshold.

Usage:
    python sweep_threshold.py --benchmark 605.mcf_s
    python sweep_threshold.py --benchmark 605.mcf_s --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    
    # All 4 benchmarks in parallel:
    python sweep_threshold.py --benchmark 605.mcf_s &
    python sweep_threshold.py --benchmark 623.xalancbmk_s &
    python sweep_threshold.py --benchmark 641.leela_s &
    python sweep_threshold.py --benchmark 600.perlbench_s &
    wait
"""

import argparse, json, os, sys, time, math, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ============================================================
# Import from shapley_pruning.py
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapley_pruning import (
    load_model, predict_mscs, predict_cpi,
    compute_mscs_vectors, compute_cosine_matrix,
    group_params, shapley_pruned,
    make_min_config, get_param_min,
    DEFAULT_CONFIG, MSCS_12_INDICES,
)

# Try importing SHAPLEY_PARAMS, fallback to GRADIENT_PARAMS
try:
    from shapley_pruning import SHAPLEY_PARAMS
except ImportError:
    from config_dse import GRADIENT_PARAMS as SHAPLEY_PARAMS

RESULT_BASE = "/scratch/gem5-TAO/micro_exp/dse_exp/result_shapley_520"


def get_bname(benchmark):
    """Extract short name: '605.mcf_s' -> 'mcf', '508.namd_r' -> 'namd'"""
    bname = benchmark.split(".")[-1]
    for suffix in ("_s", "_r"):
        bname = bname.replace(suffix, "")
    return bname


def get_json_path(benchmark):
    bname = get_bname(benchmark)
    return os.path.join(RESULT_BASE, benchmark, f"{bname}_shapley_results.json")


def run_pruned_shapley(model, normalizer, dse_engine, benchmark,
                       param_list, vectors, target_vals,
                       threshold, max_group_size=8):
    """Run pruned Shapley at a given cosine threshold."""
    # Step 2: cosine matrix
    sim_matrix = compute_cosine_matrix(vectors, param_list)
    
    # Step 3: grouping
    groups = group_params(sim_matrix, param_list, threshold, max_group_size)
    
    # Step 4: pruned Shapley (uses DEFAULT context for outside-group params)
    sv_pruned, total_evals = shapley_pruned(
        model, normalizer, dse_engine, benchmark, DEFAULT_CONFIG,
        groups, target_vals, param_list, n_permutations=200)
    
    return sv_pruned, total_evals, groups


def main():
    parser = argparse.ArgumentParser(description="Sweep cosine thresholds for Shapley pruning")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--max-group-size", type=int, default=8)
    args = parser.parse_args()
    
    benchmark = args.benchmark
    bname = get_bname(benchmark)
    
    print(f"\n{'='*70}")
    print(f"Threshold Sweep — {benchmark}")
    print(f"{'='*70}")
    
    # Load cached Full MC Shapley
    json_path = get_json_path(benchmark)
    if not os.path.exists(json_path):
        print(f"ERROR: No cached results at {json_path}")
        print(f"Run shapley_pruning.py --benchmark {benchmark} first.")
        sys.exit(1)
    
    with open(json_path) as f:
        cached = json.load(f)
    
    sv_full = cached["shapley_full"]
    sv_oaat = cached["shapley_oaat"]
    param_list = cached["params"]
    
    print(f"  Loaded Full MC Shapley: {len(sv_full)} params")
    print(f"  Thresholds to sweep: {args.thresholds}")
    
    # Load model
    print(f"\n  Loading model...")
    model, gem_norm, dse_engine = load_model(benchmark)
    
    # Compute MSCS vectors (same for all thresholds)
    print(f"  Computing MSCS vectors...")
    vectors, target_vals, vector_info = compute_mscs_vectors(
        model, gem_norm, dse_engine, benchmark, DEFAULT_CONFIG, param_list)
    param_list = [p for p in param_list if p in vectors]
    
    # Sweep thresholds
    results = []
    
    for threshold in args.thresholds:
        print(f"\n  --- Threshold = {threshold} ---")
        t0 = time.time()
        sv_pruned, n_evals, groups = run_pruned_shapley(
            model, gem_norm, dse_engine, benchmark,
            param_list, vectors, target_vals,
            threshold, args.max_group_size)
        t_elapsed = time.time() - t0
        
        # Compute metrics
        full_vals = [sv_full.get(p, 0) for p in param_list]
        pruned_vals = [sv_pruned.get(p, 0) for p in param_list]
        
        rho, _ = spearmanr(full_vals, pruned_vals)
        mse = np.mean([(f - p)**2 for f, p in zip(full_vals, pruned_vals)])
        
        group_sizes = [len(g) for g in groups]
        max_gs = max(group_sizes)
        n_groups = len(groups)
        
        # Pruned cost (factorial sum)
        pruned_cost = sum(math.factorial(len(g)) for g in groups)
        
        print(f"    Groups: {n_groups} (sizes: {group_sizes})")
        print(f"    Evals: {n_evals}, Time: {t_elapsed:.1f}s")
        print(f"    Spearman ρ: {rho:.3f}, MSE: {mse:.6f}")
        
        results.append({
            "threshold": threshold,
            "n_groups": n_groups,
            "group_sizes": group_sizes,
            "max_group_size": max_gs,
            "pruned_cost": pruned_cost,
            "n_evals": n_evals,
            "time_s": round(t_elapsed, 1),
            "spearman_rho": round(rho, 4),
            "mse": round(mse, 6),
            "shapley_pruned": {p: round(sv_pruned.get(p, 0), 6) for p in param_list},
            "groups": [[str(p) for p in g] for g in groups],
        })
    
    # Also compute one-at-a-time metrics for reference
    oaat_vals = [sv_oaat.get(p, 0) for p in param_list]
    full_vals = [sv_full.get(p, 0) for p in param_list]
    rho_oaat, _ = spearmanr(full_vals, oaat_vals)
    mse_oaat = np.mean([(f - o)**2 for f, o in zip(full_vals, oaat_vals)])
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"Summary — {bname}")
    print(f"{'='*70}")
    print(f"  {'Threshold':>10} {'Groups':>7} {'MaxGrp':>7} {'Evals':>7} {'Time':>7} {'ρ':>8} {'MSE':>12}")
    print(f"  {'─'*62}")
    for r in results:
        print(f"  {r['threshold']:>10.1f} {r['n_groups']:>7} {r['max_group_size']:>7} "
              f"{r['n_evals']:>7} {r['time_s']:>6.1f}s {r['spearman_rho']:>8.3f} {r['mse']:>12.6f}")
    print(f"  {'─'*62}")
    print(f"  {'1-at-a-time':>10} {'─':>7} {'─':>7} {len(param_list):>7} {'─':>7} {rho_oaat:>8.3f} {mse_oaat:>12.6f}")
    print(f"  {'Full MC':>10} {'─':>7} {'─':>7} {cached['metrics']['full']['evals']:>7} {'─':>7} {'1.000':>8} {'0.000000':>12}")
    
    # Save results
    output_dir = os.path.join(RESULT_BASE, benchmark)
    sweep_path = os.path.join(output_dir, f"{bname}_threshold_sweep.json")
    sweep_data = {
        "benchmark": benchmark,
        "params": param_list,
        "thresholds": args.thresholds,
        "results": results,
        "reference": {
            "oaat_rho": round(rho_oaat, 4),
            "oaat_mse": round(mse_oaat, 6),
            "full_evals": cached["metrics"]["full"]["evals"],
        }
    }
    with open(sweep_path, "w") as f:
        json.dump(sweep_data, f, indent=2)
    print(f"\n  Saved: {sweep_path}")
    
    # Plot
    plot_sweep(results, rho_oaat, mse_oaat, cached["metrics"]["full"]["evals"],
               bname, output_dir)


def plot_sweep(results, rho_oaat, mse_oaat, full_evals, bname, output_dir):
    """Plot threshold vs Spearman ρ and evaluation count."""
    thresholds = [r["threshold"] for r in results]
    rhos = [r["spearman_rho"] for r in results]
    evals = [r["n_evals"] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Spearman ρ
    color1 = '#2980b9'
    ax1.set_xlabel('Cosine Similarity Threshold', fontsize=12)
    ax1.set_ylabel('Spearman ρ (vs Full Shapley)', fontsize=12, color=color1)
    line1, = ax1.plot(thresholds, rhos, 'o-', color=color1, linewidth=2, markersize=8, label='Pruned Shapley ρ')
    ax1.axhline(rho_oaat, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'One-at-a-time ρ={rho_oaat:.3f}')
    ax1.axhline(1.0, color='#27ae60', linestyle=':', linewidth=1, alpha=0.5, label='Full Shapley ρ=1.0')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(min(min(rhos), rho_oaat) - 0.05, 1.02)
    
    # Evaluation count on secondary axis
    color2 = '#e67e22'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of Evaluations', fontsize=12, color=color2)
    line2, = ax2.plot(thresholds, evals, 's--', color=color2, linewidth=1.5, markersize=6, alpha=0.7, label='Pruned evals')
    ax2.axhline(full_evals, color='#95a5a6', linestyle=':', linewidth=1, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yscale('log')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)
    
    ax1.set_title(f'Shapley Pruning: Accuracy vs Cost — {bname}', fontsize=13, pad=10)
    ax1.set_xticks(thresholds)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"{bname}_threshold_sweep.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()