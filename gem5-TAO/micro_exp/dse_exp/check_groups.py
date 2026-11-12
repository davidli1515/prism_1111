"""
Quick diagnostic: show MSCS vectors and cosine grouping for all benchmarks.
Uses max→min perturbation for MSCS vectors.

Usage:
    python check_groups.py --threshold 0.3
"""

import argparse, os, sys, io, contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapley_pruning import (
    load_model, predict_mscs, compute_cosine_matrix,
    get_param_values, get_param_min,
    DEFAULT_CONFIG, SHAPLEY_PARAMS, MSCS_12_INDICES,
)


def group_params(sim_matrix, param_list, threshold=0.5, max_group_size=999,
                 norms=None, norm_gate=0.05):
    """Clique-based grouping. Only strong-norm members serve as gatekeepers."""
    n = len(param_list)
    groups = []
    assigned = [False] * n

    strong = set(range(n))
    if norms is not None:
        strong = {i for i in range(n) if norms[i] >= norm_gate}

    avg_sim = [np.mean([sim_matrix[i][j] for j in range(n) if j != i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: avg_sim[i], reverse=True)

    for i in order:
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in order:
            if assigned[j] or j == i:
                continue
            if len(group) >= max_group_size:
                break
            strong_members = [k for k in group if k in strong]
            if len(strong_members) == 0 or all(sim_matrix[j][k] >= threshold for k in strong_members):
                group.append(j)
                assigned[j] = True
        groups.append(group)

    named_groups = [[param_list[i] for i in g] for g in groups]
    named_groups.sort(key=len, reverse=True)
    return named_groups

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


def compute_mscs_vectors_maxmin(model, normalizer, dse_engine, workload, config, params):
    """MSCS vectors using max→min perturbation (full range)."""
    # Build max config
    max_config = config.copy()
    for name in params:
        max_config[name] = max(get_param_values(name))
    
    mscs_max = predict_mscs(model, normalizer, dse_engine, workload, max_config)
    
    vectors = {}
    for name in params:
        param_min = min(get_param_values(name))
        param_max = max(get_param_values(name))
        if param_max <= param_min:
            continue
        
        config_removed = max_config.copy()
        config_removed[name] = param_min
        mscs_removed = predict_mscs(model, normalizer, dse_engine, workload, config_removed)
        
        vectors[name] = mscs_removed[MSCS_12_INDICES] - mscs_max[MSCS_12_INDICES]
    
    return vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-group-size", type=int, default=999)
    args = parser.parse_args()

    param_list = [p for p in SHAPLEY_PARAMS if p in DEFAULT_CONFIG]

    print(f"threshold={args.threshold}, {len(param_list)} params, max→min perturbation")

    for bench in BENCHMARKS:
        bname = get_bname(bench)
        try:
            # Suppress DseEngine loading output
            with contextlib.redirect_stdout(io.StringIO()):
                model, norm, dse = load_model(bench)

            vectors = compute_mscs_vectors_maxmin(
                model, norm, dse, bench, DEFAULT_CONFIG, param_list)
            valid_params = [p for p in param_list if p in vectors]

            # Print vectors
            print(f"\n  {bname}:")
            for p in valid_params:
                v = vectors[p]
                n = np.linalg.norm(v)
                pmin = min(get_param_values(p))
                pmax = max(get_param_values(p))
                print(f"    {p:<16} norm={n:.4f} [{pmax}→{pmin}]  "
                      f"FP=[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f},{v[3]:+.3f}] "
                      f"BP=[{v[4]:+.3f},{v[5]:+.3f},{v[6]:+.3f},{v[7]:+.3f}] "
                      f"SP=[{v[8]:+.3f},{v[9]:+.3f},{v[10]:+.3f},{v[11]:+.3f}]")

            # Grouping
            sim = compute_cosine_matrix(vectors, valid_params)
            param_norms = [np.linalg.norm(vectors[p]) for p in valid_params]
            groups = group_params(sim, valid_params, args.threshold, args.max_group_size,
                                  norms=param_norms)

            sizes = [len(g) for g in groups]
            print(f"    → {len(groups)} groups: {sizes}")
            for gi, g in enumerate(groups):
                print(f"      G{gi}: {g}")

        except Exception as e:
            print(f"  {bname:<15} ERROR: {e}")


if __name__ == "__main__":
    main()