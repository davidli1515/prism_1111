#!/usr/bin/env python3
"""
Summarize dse17/results/*_s<seed>.json: HV gain over the BO base at matched cost (common reference),
min CPI per area budget and per-budget win share, area drift of the walks, decision statistics.
Usage: python summarize17.py [--seed 42]
"""
import os, sys, json, glob, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import hypervolume


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    d0 = os.path.join(HERE, "results")
    R = {"bo_base": json.load(open(os.path.join(d0, f"bo_base_s{args.seed}.json")))}
    for f in sorted(glob.glob(os.path.join(d0, f"*_s{args.seed}.json"))):
        n = os.path.basename(f)[:-len(f"_s{args.seed}.json")]
        if n != "bo_base":
            R[n] = json.load(open(f))
    arms = [a for a in R if a != "bo_base"]
    allP = np.concatenate([np.c_[d["cpis"], d["areas"]] for d in R.values()])
    ref = (float(allP[:, 0].max()), float(allP[:, 1].max()))
    P0 = np.c_[R["bo_base"]["cpis"], R["bo_base"]["areas"]]; hv0 = hypervolume(P0, ref)
    print(f"ref={ref}  BO base {len(P0)} pts cost {R['bo_base']['total_cost']:.0f} HV {hv0:.3f}")
    if not arms:
        return
    lo = R["bo_base"]["total_cost"]; hi = min(R[a]["total_cost"] for a in arms)
    grid = sorted({int(x) for x in np.linspace(lo, hi, 6)})
    print("\nHV gain over BO base (%) at matched cost")
    print(f"  {'arm':30s} {'cost':>7s} {'final':>7s}  " + "  ".join(f"@{g:>6d}" for g in grid))
    for a in arms:
        d = R[a]; P = np.c_[d["cpis"], d["areas"]]; c = np.array(d["cost"])
        hv = lambda m: 100 * (hypervolume(P[m], ref) / hv0 - 1)
        print(f"  {a:30s} {d['total_cost']:7.0f} {hv(np.ones(len(c), bool)):+7.2f}  " + "  ".join(f"{hv(c <= g):+7.2f}" for g in grid))

    budgets = np.arange(15, 131, 1.0)
    best = {}
    for a in ["bo_base"] + arms:
        P = np.c_[R[a]["cpis"], R[a]["areas"]]
        best[a] = np.array([P[P[:, 1] <= b, 0].min() if (P[:, 1] <= b).any() else np.inf for b in budgets])
    print("\nmin CPI at area budget (mm^2), relative to BO base")
    for b in [18, 20, 25, 30, 40, 50, 60, 80, 100, 130]:
        i = int(np.where(budgets == b)[0][0])
        print(f"  {b:4d}  bo={best['bo_base'][i]:.4f}  " + "  ".join(f"{a}={100*(best[a][i]/best['bo_base'][i]-1):+.2f}%" for a in arms))
    M = np.vstack([best[a] for a in arms]); win = M == M.min(0)
    print("  share of budgets 15..130 with the lowest CPI: " + "  ".join(f"{a}={win[i].mean():.2f}" for i, a in enumerate(arms)))
    print("  share of budgets improved over BO: " + "  ".join(f"{a}={np.mean(best[a] < best['bo_base'] - 1e-12):.2f}" for a in arms))

    print("\nwalks")
    for a in arms:
        d = R[a]; area_of = {tuple(map(tuple, [[k, c[k]] for k in sorted(c)])): ar for c, ar in zip(d["configs"], d["areas"])}
        drift = [area_of[tuple(map(tuple, w["end_key"]))] / w["start_area"] - 1 for w in d["walks"]
                 if tuple(map(tuple, w["end_key"])) in area_of and w["n_steps"] > 0]
        st = [s for w in d["walks"] for s in w["log"] if "changes" in s]
        stops = [s["stop"] for w in d["walks"] for s in w["log"] if "stop" in s]
        print(f"  {a:30s} walks {len(d['walks'])}  steps/walk {len(st)/max(1,len(d['walks'])):.1f}  "
              f"end/start area-1 median {100*np.median(drift) if drift else float('nan'):+.1f}% (p90 {100*np.percentile(drift,90) if drift else float('nan'):+.1f}%)  "
              f"decrease/step {np.mean(['decrease' in s['changes'].values() for s in st]) if st else float('nan'):.2f}  "
              f"stops {dict((k, stops.count(k)) for k in set(stops))}")


if __name__ == "__main__":
    main()
