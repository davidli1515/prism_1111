"""
Diagnostic: show pairwise cosine between width params.
Reveals why they get split into different groups.

Usage:
    python debug_width_cosine.py
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapley_pruning import (
    load_model, compute_mscs_vectors, compute_cosine_matrix,
    DEFAULT_CONFIG, SHAPLEY_PARAMS,
)

WIDTH_PARAMS = ["fetchWidth", "decodeWidth", "dispatchWidth", "commitWidth", "wbWidth"]

BENCHMARKS = [
    "508.namd_r",
    "641.leela_s",
    "648.exchange2_s",
    "657.xz_s",
]


def get_bname(b):
    bname = b.split(".")[-1]
    for s in ("_s", "_r"):
        bname = bname.replace(s, "")
    return bname


def main():
    param_list = [p for p in SHAPLEY_PARAMS if p in DEFAULT_CONFIG]

    for bench in BENCHMARKS:
        bname = get_bname(bench)
        print(f"\n{'='*70}")
        print(f"{bench} ({bname})")
        print(f"{'='*70}")

        model, norm, dse = load_model(bench)
        vectors, target_vals, info = compute_mscs_vectors(
            model, norm, dse, bench, DEFAULT_CONFIG, param_list)

        valid = [p for p in param_list if p in vectors]

        # Print width param vectors
        print(f"\n  Width param MSCS vectors (12-dim):")
        for p in WIDTH_PARAMS:
            if p in vectors:
                v = vectors[p]
                n = np.linalg.norm(v)
                print(f"    {p:<16} norm={n:.4f}  "
                      f"FP=[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f},{v[3]:+.3f}] "
                      f"BP=[{v[4]:+.3f},{v[5]:+.3f},{v[6]:+.3f},{v[7]:+.3f}] "
                      f"SP=[{v[8]:+.3f},{v[9]:+.3f},{v[10]:+.3f},{v[11]:+.3f}]")

        # Pairwise cosine between width params
        print(f"\n  Pairwise |cosine| between width params:")
        print(f"    {'':>16}", end="")
        for p in WIDTH_PARAMS:
            print(f"  {p[:8]:>8}", end="")
        print()

        for p1 in WIDTH_PARAMS:
            if p1 not in vectors:
                continue
            print(f"    {p1:<16}", end="")
            for p2 in WIDTH_PARAMS:
                if p2 not in vectors:
                    print(f"  {'---':>8}", end="")
                    continue
                v1, v2 = vectors[p1], vectors[p2]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-10 or n2 < 1e-10:
                    cos = 0
                else:
                    cos = abs(np.dot(v1, v2) / (n1 * n2))
                marker = " ✓" if cos > 0.3 else " ✗"
                print(f"  {cos:>6.3f}{marker}", end="")
            print()

        # Also show cosine of width params vs top non-width params
        non_width = [p for p in valid if p not in WIDTH_PARAMS]
        print(f"\n  Width vs non-width cosine (top interactions):")
        for wp in WIDTH_PARAMS:
            if wp not in vectors:
                continue
            pairs = []
            for nwp in non_width:
                v1, v2 = vectors[wp], vectors[nwp]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-10 or n2 < 1e-10:
                    continue
                cos = abs(np.dot(v1, v2) / (n1 * n2))
                pairs.append((nwp, cos))
            pairs.sort(key=lambda x: x[1], reverse=True)
            top3 = pairs[:3]
            top_str = ", ".join(f"{p}={c:.3f}" for p, c in top3)
            print(f"    {wp:<16} → {top_str}")


if __name__ == "__main__":
    main()
