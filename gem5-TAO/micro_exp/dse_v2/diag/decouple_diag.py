"""Per-benchmark: at the BO-500 Pareto starts, for every one-stride INCREASE compute
gain=(cpi0-cpi)/cpi0, area increase and cancellation score (12 pressure comps).
Reports how often decouple mode (best gain < eps) would trigger, and the gain of the
cancellation pick vs the best-gain pick vs the gain/area pick."""
import os, sys, json, time, numpy as np
sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/dse_v2")
from common import Surrogate, Area, pareto_mask, ALL_PARAMS, DEFAULT_CKPT
from refine_v3 import _shift, _cancel, P3
from mscs_reclaim_v2 import prefetch_areas
out = {}
for b in sys.argv[1:]:
    S = Surrogate(DEFAULT_CKPT, [b], "18k", device="cpu", max_slices=50); A = Area()
    r = json.load(open(f"results_v3_single_1006/{b}/bo_s42.json"))
    q = np.array(r["queries"]); keep = q <= 500
    P = np.c_[np.array(r["cpis"])[keep], np.array(r["areas"])[keep]]
    cfgs = [c for c, k in zip(r["configs"], keep) if k]
    starts = [cfgs[i] for i in np.where(pareto_mask(P))[0]]
    rows = []
    for si, c in enumerate(starts):
        cpi0, a0, m0 = S.cpi(c), A(c), S.mscs(c)
        inc = [(n, cn) for n in ALL_PARAMS for cn in [_shift(c, n, +1)] if cn is not None]
        prefetch_areas(A, [cn for _, cn in inc], 8)
        cand = []
        for n, cn in inc:
            a = A(cn)
            if a is None: continue
            g = (cpi0 - S.cpi(cn)) / cpi0
            cand.append(dict(param=n, gain=g, darea=(a - a0) / a0, cancel=_cancel(m0 - S.mscs(cn))))
        pos = [x for x in cand if x["gain"] > 0]
        if not pos: continue
        bg = max(pos, key=lambda x: x["gain"]); ba = max(pos, key=lambda x: x["gain"] / max(x["darea"], 1e-6))
        bc = max(pos, key=lambda x: x["cancel"])
        rows.append(dict(start=si, cpi0=cpi0, area0=a0, best_gain=bg["gain"], best_gain_param=bg["param"],
                         gain_per_area_param=ba["param"], gain_per_area_gain=ba["gain"],
                         cancel_param=bc["param"], cancel_gain=bc["gain"], cancel_score=bc["cancel"],
                         decouple=bg["gain"] < P3["eps"], n_pos=len(pos)))
    dec = [x for x in rows if x["decouple"]]
    print(f"\n== {b}: {len(rows)} starts, decouple would trigger at {len(dec)} ({len(dec)/len(rows)*100:.0f}%)  eps={P3['eps']}")
    print(f"   best_gain percentiles: {np.percentile([x['best_gain'] for x in rows],[10,50,90]).round(4)}")
    if dec:
        print(f"   in decouple starts: cancel-pick gain median {np.median([x['cancel_gain'] for x in dec]):.4f} "
              f"vs best-gain {np.median([x['best_gain'] for x in dec]):.4f} "
              f"(ratio {np.median([x['cancel_gain']/x['best_gain'] for x in dec]):.2f}); "
              f"cancel picks: {dict(zip(*np.unique([x['cancel_param'] for x in dec],return_counts=True)))}")
    print(f"   best-gain picks: {dict(zip(*np.unique([x['best_gain_param'] for x in rows],return_counts=True)))}")
    print(f"   gain/area picks: {dict(zip(*np.unique([x['gain_per_area_param'] for x in rows],return_counts=True)))}")
    out[b] = rows; A.flush()
json.dump(out, open("diag/decouple_diag.json", "w"), indent=0)
