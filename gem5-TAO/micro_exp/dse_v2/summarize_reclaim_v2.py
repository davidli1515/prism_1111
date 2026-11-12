#!/usr/bin/env python3
"""Reclaim v1 + v2: HV gain over the BO base vs extra surrogate queries (common reference point)."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import hypervolume, pareto_mask

D = "results_v2_1005"
ref = json.load(open(f"{D}/summary.json"))["ref_point"]
base = json.load(open(f"{D}/bo_s42.json"))
hv_bo = hypervolume(np.c_[base["cpis"], base["areas"]], ref)
print(f"BO base HV = {hv_bo:.2f} | best known from headroom log (2 rounds) = 348.64 "
      f"(+{(348.64-hv_bo)/hv_bo*100:.2f}%)")
Q = [100, 250, 500, 750, 1000]
print(f"{'method':24s} " + " ".join(f"{'+%HV@'+str(q):>10s}" for q in Q) + f" {'used':>5s} {'accepted':>8s}")
out = {}
for f in sorted(glob.glob(f"{D}/reclaim/*_s42.json") + glob.glob(f"{D}/reclaim_v2/*_s42.json")):
    r = json.load(open(f)); m = r["method"]
    C, A, q = np.array(r["cpis"]), np.array(r["areas"]), np.array(r["queries"])
    row = [(hypervolume(np.c_[C[q <= qq], A[q <= qq]], ref) - hv_bo) / hv_bo * 100 for qq in Q]
    out[m] = {"hv_gain_pct": dict(zip(Q, row)), "queries_used": int(q.max()), "accepted": r["accepted"]}
    print(f"{m:24s} " + " ".join(f"{x:10.2f}" for x in row) + f" {int(q.max()):5d} {r['accepted']:8d}")
json.dump({"hv_bo": hv_bo, "methods": out}, open(f"{D}/reclaim_v2/summary_reclaim_v2.json", "w"), indent=2)
