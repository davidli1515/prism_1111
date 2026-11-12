"""Diagnostic: on the BO-500 Pareto starts (what v3 refine began from), evaluate every
one-stride DECREASE and record geomean dCPI_rel, per-workload ||dMSCS||_1 (max/mean),
averaged-MSCS ||d||_1, and area saved. Shows what fraction of otherwise-acceptable
reclaims the PROTECT rule (max_w ||dMSCS_w||_1 >= tau_m) blocks, and whether blocked
ones are 'hidden value' (CPI cost later) or just noise. Surrogate on CPU."""
import os, sys, json, time, numpy as np
sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/dse_v2")
from common import Area, pareto_mask, ALL_PARAMS
from refine_v3 import SurrogateWL, _shift
from common import SPEC17, DEFAULT_CKPT
S = SurrogateWL(DEFAULT_CKPT, SPEC17, "18k", device="cpu", max_slices=50); A = Area()
r = json.load(open("results_v3_1006/bo_s42.json"))
q = np.array(r["queries"]); keep = q <= 500
P = np.c_[np.array(r["cpis"])[keep], np.array(r["areas"])[keep]]
cfgs = [c for c, k in zip(r["configs"], keep) if k]
starts = [cfgs[i] for i in np.where(pareto_mask(P))[0]]
print(f"{len(starts)} BO-500 Pareto starts", flush=True)
rows = []; t0 = time.time()
for si, c in enumerate(starts):
    cpi0, a0 = S.cpi(c), A(c); m0 = S.mscs_wl(c); mavg0 = S.mscs(c)
    for n in ALL_PARAMS:
        for k in (1, 4):
            cd = _shift(c, n, -k)
            if cd is None: continue
            a = A(cd)
            if a is None: continue
            d = S.mscs_wl(cd) - m0; l1 = np.abs(d).sum(1)
            rows.append(dict(start=si, param=n, k=k, dcpi=(S.cpi(cd)-cpi0)/cpi0,
                             l1_max=float(l1.max()), l1_mean=float(l1.mean()),
                             l1_avg=float(np.abs(S.mscs(cd)-mavg0).sum()),
                             n_wl_over=int((l1 >= 0.02).sum()), area_saved=a0-a, area0=a0))
    print(f"  start {si+1}/{len(starts)} done, q={S.n_queries} [{time.time()-t0:.0f}s]", flush=True)
A.flush()
json.dump(rows, open("diag/protect_diag.json", "w"), indent=0)
R = rows; acc = [x for x in R if 0 <= x["dcpi"] < 0.003]; neg = [x for x in R if x["dcpi"] < 0]
print(f"\nmoves={len(R)}  harmful-to-keep(dCPI<0)={len(neg)}  acceptable(0<=dCPI<0.3%)={len(acc)}  rejected={len(R)-len(acc)-len(neg)}")
for tm in (0.01, 0.02, 0.05, 0.1):
    blk = [x for x in acc if x["l1_max"] >= tm]
    print(f"tau_m={tm}: PROTECT blocks {len(blk)}/{len(acc)} acceptable reclaims; "
          f"area they would save: {sum(x['area_saved'] for x in blk):.1f} of {sum(x['area_saved'] for x in acc):.1f} mm2")
print("\nacceptable reclaims: l1_max percentiles", np.percentile([x["l1_max"] for x in acc],[10,50,90]).round(3),
      "| l1_avg percentiles", np.percentile([x["l1_avg"] for x in acc],[10,50,90]).round(3))
print("rejected (dCPI>=0.3%) l1_max percentiles", np.percentile([x["l1_max"] for x in R if x["dcpi"]>=0.003],[10,50,90]).round(3))
import collections
print("\nblocked@0.02 by param:", collections.Counter(x["param"] for x in acc if x["l1_max"]>=0.02).most_common(8))
print("allowed@0.02 by param:", collections.Counter(x["param"] for x in acc if x["l1_max"]<0.02).most_common(8))
