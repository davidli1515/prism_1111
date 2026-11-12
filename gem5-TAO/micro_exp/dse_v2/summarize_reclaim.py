#!/usr/bin/env python3
"""HV gain over the BO base vs extra surrogate queries, common reference point."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import hypervolume, pareto_mask

D = "results_v2_1005"
ref = json.load(open(f"{D}/summary.json"))["ref_point"]
base = json.load(open(f"{D}/bo_s42.json"))
hv_bo = hypervolume(np.c_[base["cpis"], base["areas"]], ref)
hr = f"{D}/headroom.json"
hv_best = json.load(open(hr))["rounds"][-1]["hv"] if os.path.exists(hr) else None
print(f"BO base HV = {hv_bo:.2f}" + (f" | best known (headroom) = {hv_best:.2f} "
      f"(+{(hv_best-hv_bo)/hv_bo*100:.2f}%)" if hv_best else ""))
Q = [100, 250, 500, 750, 1000]
print(f"{'method':12s} " + " ".join(f"{'+%HV@'+str(q):>10s}" for q in Q) + f" {'accepted':>8s} {'|PF|':>5s}")
out = {}
for f in sorted(glob.glob(f"{D}/reclaim/*_s42.json")):
    r = json.load(open(f)); m = r["method"]
    C, A, q = np.array(r["cpis"]), np.array(r["areas"]), np.array(r["queries"])
    row = []
    for qq in Q:
        keep = q <= qq
        row.append((hypervolume(np.c_[C[keep], A[keep]], ref) - hv_bo) / hv_bo * 100)
    pf = int(pareto_mask(np.c_[C, A]).sum())
    out[m] = {"hv_gain_pct": dict(zip(Q, row)), "accepted": r["accepted"], "pareto": pf}
    print(f"{m:12s} " + " ".join(f"{x:10.2f}" for x in row) + f" {r['accepted']:8d} {pf:5d}")
json.dump({"hv_bo": hv_bo, "hv_best_known": hv_best, "methods": out},
          open(f"{D}/reclaim/summary_reclaim.json", "w"), indent=2)
