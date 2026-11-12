#!/usr/bin/env python3
"""
Compare the coordinated-swap arms (results/<arm>_s<seed>.json) and the shared BO base.
  HV (common reference = max CPI / max area over all archives) at the final cost and at
  matched costs; min CPI at area budgets (15..130 mm^2) and the share of budgets each arm wins;
  decision statistics from the refine logs (decreases applied, PROTECT events, dominant type).
Usage: python summarize.py [--seed 42]
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import hypervolume

ARMS = ["mscs_protect", "mscs_noprotect", "scalar_protect", "scalar_noprotect"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="", help='e.g. ".rescored" to read <arm>_s42.rescored.json')
    args = ap.parse_args()
    R = {}
    for a in ["bo_base"] + ARMS:
        p = os.path.join(HERE, "results", f"{a}_s{args.seed}{args.suffix}.json")
        if os.path.exists(p):
            R[a] = json.load(open(p))
    if len(R) < 2:
        sys.exit(f"found {list(R)}")
    allP = np.concatenate([np.c_[d["cpis"], d["areas"]] for d in R.values()])
    ref = (float(allP[:, 0].max()), float(allP[:, 1].max()))
    hv_bo = hypervolume(np.c_[R["bo_base"]["cpis"], R["bo_base"]["areas"]], ref)
    print(f"ref={ref}  BO base: {len(R['bo_base']['cpis'])} pts, cost {R['bo_base']['total_cost']:.0f}, HV {hv_bo:.3f}")

    arms = [a for a in ARMS if a in R]
    grid = sorted({int(x) for x in np.linspace(R["bo_base"]["total_cost"], min(R[a]["total_cost"] for a in arms), 6)})
    print("\nHV gain over BO (%) at matched cost (joint queries)")
    print("  arm                 final_cost  final   " + "  ".join(f"@{g:>6d}" for g in grid))
    for a in arms:
        d = R[a]; P = np.c_[d["cpis"], d["areas"]]; c = np.array(d["cost"])
        hv = lambda m: 100 * (hypervolume(P[m], ref) / hv_bo - 1)
        print(f"  {a:18s} {d['total_cost']:10.0f}  {hv(np.ones(len(c), bool)):+6.2f}   "
              + "  ".join(f"{hv(c <= g):+7.2f}" for g in grid))

    budgets = np.arange(15, 131, 1.0)
    best = {}
    for a in ["bo_base"] + arms:
        P = np.c_[R[a]["cpis"], R[a]["areas"]]
        best[a] = np.array([P[P[:, 1] <= b, 0].min() if (P[:, 1] <= b).any() else np.inf for b in budgets])
    print("\nmin CPI at area budget (mm^2)")
    for b in [20, 25, 30, 40, 50, 60, 80, 100, 130]:
        i = int(np.where(budgets == b)[0][0])
        print(f"  {b:4d}  " + "  ".join(f"{a}={best[a][i]:.4f}" for a in ["bo_base"] + arms))
    M = np.vstack([best[a] for a in arms])
    win = M == M.min(0)
    print("  share of 1-mm^2 budgets 15..130 where the arm has the lowest CPI (ties count for all): "
          + "  ".join(f"{a}={win[i].mean():.2f}" for i, a in enumerate(arms)))

    print("\ndecision statistics (refine logs)")
    for a in arms:
        st = [s for l in R[a]["logs"] for s in l["steps"]]
        if not st:
            continue
        dec = np.mean(["decrease" in s["changes"].values() for s in st])
        prot = [s["n_protected"] for s in st if s.get("n_protected") is not None]
        dom = {k: np.mean([s["dominant"] == k for s in st]) for k in ["FP", "BP", "SP"]}
        print(f"  {a:18s} steps {len(st):5d}  steps/walk {len(st)/max(1,len(R[a]['logs'])):.1f}  "
              f"decrease applied {dec:.2f}  protected/step {np.mean(prot) if prot else float('nan'):.2f}  "
              f"dominant FP/BP/SP {dom['FP']:.2f}/{dom['BP']:.2f}/{dom['SP']:.2f}")


if __name__ == "__main__":
    main()
