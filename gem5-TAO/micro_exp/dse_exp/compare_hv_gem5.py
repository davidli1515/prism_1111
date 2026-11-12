#!/usr/bin/env python3
"""
Compare BO vs BO+MSCS Pareto hypervolume using gem5 ground-truth CPI.

The surrogate claim (BO+MSCS ~5% higher HV than BO) is re-evaluated on
gem5-measured geomean CPI. Area is analytic (identical for surrogate and
gem5), so we take it from the joint results JSON. Both fronts are scored
against a COMMON reference point for a fair HV comparison.

Usage:
  python compare_hv_gem5.py \
      --bo-json     results_17bm_54/joint_17_gem_bo_results.json \
      --mscs-json   results_17bm_54/joint_17_gem_bo_mscs_results.json \
      --bo-val      validate_joint_bo \
      --mscs-val    validate_joint_mscs
"""
import argparse, json, os
import numpy as np
from collections import defaultdict


def compute_pareto(points):
    """points: list of (cpi, area). Return indices of non-dominated points."""
    idx = []
    for i, (ci, ai) in enumerate(points):
        dom = False
        for j, (cj, aj) in enumerate(points):
            if j != i and cj <= ci and aj <= ai and (cj < ci or aj < ai):
                dom = True
                break
        if not dom:
            idx.append(i)
    return idx


def hv_2d(pareto_points, ref):
    valid = [(c, a) for c, a in pareto_points if c < ref[0] and a < ref[1]]
    if not valid:
        return 0.0
    valid.sort(key=lambda x: x[0])
    hv, prev_area = 0.0, ref[1]
    for cpi, area in valid:
        hv += (ref[0] - cpi) * (prev_area - area)
        prev_area = area
    return hv


def load_gem5_gmean(val_dir, good_benches):
    """Recompute gem5 gmean per config from raw results (good benches only)."""
    raw = json.load(open(os.path.join(val_dir, "joint_validation_raw.json")))
    cfg_cpis = defaultdict(dict)
    for r in raw:
        if r["bench"] in good_benches and r.get("cpi"):
            cfg_cpis[r["cfg_id"]][r["bench"]] = r["cpi"]
    gmean = {}
    for ci, d in cfg_cpis.items():
        if len(d) == len(good_benches):
            vals = np.array(list(d.values()))
            gmean[ci] = float(np.exp(np.mean(np.log(vals))))
    return gmean


def bench_success(val_dir):
    raw = json.load(open(os.path.join(val_dir, "joint_validation_raw.json")))
    ok = defaultdict(int)
    tot = defaultdict(int)
    for r in raw:
        tot[r["bench"]] += 1
        if r.get("cpi"):
            ok[r["bench"]] += 1
    return {b for b in tot if ok[b] == tot[b]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bo-json", required=True)
    ap.add_argument("--mscs-json", required=True)
    ap.add_argument("--bo-val", required=True)
    ap.add_argument("--mscs-val", required=True)
    args = ap.parse_args()

    # Use benchmarks that succeeded in BOTH validations for a consistent gmean
    good = bench_success(args.bo_val) & bench_success(args.mscs_val)
    print(f"Common successful benchmarks: {len(good)}")
    for b in sorted(good):
        print(f"  {b}")
    print()

    bo_data = json.load(open(args.bo_json))
    mscs_data = json.load(open(args.mscs_json))

    bo_gem5 = load_gem5_gmean(args.bo_val, good)
    mscs_gem5 = load_gem5_gmean(args.mscs_val, good)

    # Build (cpi, area) lists from validated configs
    def front(data, gem5_gmean):
        pts, surr_pts = [], []
        for ci, cpi in gem5_gmean.items():
            area = data["areas"][ci]
            pts.append((cpi, area))
            surr_pts.append((data["cpis"][ci], area))
        return pts, surr_pts

    bo_pts, bo_surr = front(bo_data, bo_gem5)
    mscs_pts, mscs_surr = front(mscs_data, mscs_gem5)

    print(f"Validated configs: BO={len(bo_pts)}, BO+MSCS={len(mscs_pts)}\n")

    # Common reference point: worst CPI and worst area across BOTH fronts, + margin
    all_cpi = [c for c, a in bo_pts + mscs_pts]
    all_area = [a for c, a in bo_pts + mscs_pts]
    ref = (max(all_cpi) * 1.05, max(all_area) * 1.05)
    print(f"Common reference point: CPI={ref[0]:.4f}, area={ref[1]:.1f}\n")

    def score(pts):
        pidx = compute_pareto(pts)
        pf = [pts[i] for i in pidx]
        return hv_2d(pf, ref), len(pidx)

    # gem5-based HV
    bo_hv, bo_np = score(bo_pts)
    mscs_hv, mscs_np = score(mscs_pts)
    # surrogate-based HV (for reference, same configs, same ref)
    bo_hv_s, _ = score(bo_surr)
    mscs_hv_s, _ = score(mscs_surr)

    print("=" * 56)
    print(f"{'':<14}{'gem5 HV':>12}{'surrogate HV':>15}{'#pareto':>10}")
    print("-" * 56)
    print(f"{'BO':<14}{bo_hv:>12.2f}{bo_hv_s:>15.2f}{bo_np:>10}")
    print(f"{'BO+MSCS':<14}{mscs_hv:>12.2f}{mscs_hv_s:>15.2f}{mscs_np:>10}")
    print("-" * 56)
    gem5_gain = (mscs_hv - bo_hv) / bo_hv * 100
    surr_gain = (mscs_hv_s - bo_hv_s) / bo_hv_s * 100
    print(f"{'MSCS gain':<14}{gem5_gain:>11.2f}%{surr_gain:>14.2f}%")
    print("=" * 56)
    print(f"\nSurrogate claimed BO+MSCS advantage: {surr_gain:.1f}%")
    print(f"gem5-verified BO+MSCS advantage:     {gem5_gain:.1f}%")


if __name__ == "__main__":
    main()
