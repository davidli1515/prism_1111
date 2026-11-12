"""
Ablation: Matched-budget MC Shapley vs MSCS-Pruned Shapley.
MC permutation count = pruned_evals // (d+1), so eval budgets match exactly.

Usage:
    python ablation_mc_vs_pruned.py --benchmark 605.mcf_s
"""

import argparse, json, os, sys, time
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapley_pruning import (
    load_model, predict_cpi, make_min_config, get_param_min,
    DEFAULT_CONFIG, SHAPLEY_PARAMS,
)

RESULT_BASE = "/scratch/gem5-TAO/micro_exp/dse_exp/result_shapley_520"


def get_bname(benchmark):
    bname = benchmark.split(".")[-1]
    for suffix in ("_s", "_r"):
        bname = bname.replace(suffix, "")
    return bname


def shapley_mc_nocache(model, normalizer, dse_engine, workload, base_config,
                        param_list, target_vals, n_permutations, seed=42):
    """MC Shapley WITHOUT caching. Every eval is a real inference call."""
    min_config = make_min_config(base_config, param_list)
    rng = np.random.RandomState(seed)
    d = len(param_list)
    shapley = {name: 0.0 for name in param_list}
    total_evals = 0

    t0 = time.time()
    for perm_idx in range(n_permutations):
        perm = rng.permutation(d)
        config = min_config.copy()
        prev_cpi = predict_cpi(model, normalizer, dse_engine, workload, config)
        total_evals += 1
        for pos in range(d):
            name = param_list[perm[pos]]
            config[name] = target_vals[name]
            curr_cpi = predict_cpi(model, normalizer, dse_engine, workload, config)
            total_evals += 1
            marginal = prev_cpi - curr_cpi
            shapley[name] += marginal / n_permutations
            prev_cpi = curr_cpi

    wall_time = time.time() - t0
    return shapley, total_evals, wall_time


def main():
    parser = argparse.ArgumentParser(description="Ablation: Matched-budget MC vs Pruned")
    parser.add_argument("--benchmark", required=True)
    args = parser.parse_args()

    benchmark = args.benchmark
    bname = get_bname(benchmark)
    output_dir = os.path.join(RESULT_BASE, benchmark)

    # Load cached results
    json_path = os.path.join(output_dir, f"{bname}_shapley_results.json")
    if not os.path.exists(json_path):
        print(f"ERROR: No cached results at {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        cached = json.load(f)

    sv_full = cached["shapley_full"]
    sv_pruned = cached["shapley_pruned"]
    sv_oaat = cached["shapley_oaat"]
    param_list = cached["params"]
    target_vals = cached["target_values"]
    d = len(param_list)

    full_vals = [sv_full[p] for p in param_list]
    pruned_vals = [sv_pruned[p] for p in param_list]
    oaat_vals = [sv_oaat[p] for p in param_list]

    rho_pruned = float(spearmanr(full_vals, pruned_vals)[0])
    rho_oaat = float(spearmanr(full_vals, oaat_vals)[0])
    pruned_evals = cached["metrics"]["pruned"]["evals"]
    pruned_time = cached["metrics"]["pruned"]["time_s"]
    full_evals = cached["metrics"]["full"]["evals"]

    # Compute matched MC perm count from pruned eval budget
    n_perms_matched = max(1, pruned_evals // (d + 1))
    mc_evals_expected = n_perms_matched * (d + 1)

    print(f"\n{'='*70}")
    print(f"Ablation: Matched-budget MC vs Pruned — {benchmark}")
    print(f"{'='*70}")
    print(f"  Params: {d}")
    print(f"  Pruned: ρ={rho_pruned:.3f}, {pruned_evals} evals")
    print(f"  Matched MC budget: {n_perms_matched} perms × {d+1} = {mc_evals_expected} evals")

    # Load model
    print(f"\n  Loading model...")
    model, gem_norm, dse_engine = load_model(benchmark)

    # Run matched-budget MC
    print(f"\n  Running MC {n_perms_matched} perms (no cache)...")
    sv_mc, ne_mc, t_mc = shapley_mc_nocache(
        model, gem_norm, dse_engine, benchmark, DEFAULT_CONFIG,
        param_list, target_vals, n_permutations=n_perms_matched, seed=42)

    mc_vals = [sv_mc[p] for p in param_list]
    rho_mc = float(spearmanr(full_vals, mc_vals)[0])
    mse_mc = float(np.mean([(f - m)**2 for f, m in zip(full_vals, mc_vals)]))
    mse_pruned = float(np.mean([(f - p)**2 for f, p in zip(full_vals, pruned_vals)]))

    pruned_wins = bool(rho_pruned > rho_mc)

    # Summary
    print(f"\n  {'Method':<25} {'ρ':>8} {'Evals':>8} {'Time':>8}")
    print(f"  {'─'*50}")
    print(f"  {'Full MC (ground truth)':<25} {'1.000':>8} {full_evals:>8}")
    print(f"  {'Pruned (ours)':<25} {rho_pruned:>8.3f} {pruned_evals:>8} {pruned_time:>7.1f}s")
    print(f"  {'Matched MC':<25} {rho_mc:>8.3f} {ne_mc:>8} {t_mc:>7.1f}s")
    print(f"  {'One-at-a-time':<25} {rho_oaat:>8.3f} {d:>8}")
    print(f"\n  >>> Pruned {'WINS' if pruned_wins else 'LOSES'} (ρ {rho_pruned:.3f} vs {rho_mc:.3f})")

    # Save
    ablation_data = {
        "benchmark": benchmark,
        "n_params": d,
        "pruned": {"evals": pruned_evals, "rho": round(rho_pruned, 4), "mse": round(mse_pruned, 6)},
        "matched_mc": {"perms": n_perms_matched, "evals": ne_mc, "rho": round(rho_mc, 4),
                       "mse": round(mse_mc, 6), "time_s": round(t_mc, 1)},
        "oaat": {"evals": d, "rho": round(rho_oaat, 4)},
        "pruned_wins": pruned_wins,
    }
    ablation_path = os.path.join(output_dir, f"{bname}_ablation_matched.json")
    with open(ablation_path, "w") as f:
        json.dump(ablation_data, f, indent=2, default=lambda o: 
                  bool(o) if isinstance(o, (np.bool_,)) else
                  int(o) if isinstance(o, (np.integer,)) else
                  float(o) if isinstance(o, (np.floating,)) else o)
    print(f"  Saved: {ablation_path}")


if __name__ == "__main__":
    main()