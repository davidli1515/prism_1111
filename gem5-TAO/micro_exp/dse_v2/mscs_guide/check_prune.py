#!/usr/bin/env python3
"""
Check A: can the current MSCS prune the neighbourhood the refine step has to evaluate?

From the pure-BO front of an existing run, walk --steps with the exact SCALAR rule (all
neighbours evaluated, so the walk itself is unchanged). At every state:
  A1 (increase)  oracle = max gain/area among the +1 moves (the scalar choice).
     Rank the moves WITHOUT evaluating them by:
       mscs_relief      sum_i m_i R[p,i]                     (current MSCS x relief map)
       mscs_relief_area mscs_relief / darea                  (area is free: analytic model)
       mscs_dom         R[p, dominant component]
       mscs_dom_area    mscs_dom / darea
       static, static_area   mean gain of p over random configs (MSCS-free prior)
       prev, prev_area       the gains measured for p at the previous state (scalar memory)
       area_only        cheapest move first
     Reported: rank of the oracle, oracle-in-top-k, and eff ratio@k = best eff among the
     top-k (evaluated) / oracle eff.  Random baseline is analytic (k/n).
  A2 (reclaim)   oracle = scalar reclaim (max area saved among cost < tau).  Evaluating
     candidates in area-saved order, the oracle is reached after its rank in that order;
     mscs order defers candidates whose predicted pressure increase (m1 . R[p] * stride) is
     above the median.  Reported: queries to reach the oracle under both orders.
Also writes the visited states (for check_pairs.py).

Usage: python mscs_guide/check_prune.py --bench 638.imagick_s --map mscs_guide/maps/map_n40_s0.npz
"""
import os, sys, json, time, argparse
from guide_common import (HERE, ROOT, PARAMS, load_map, relief_scores, dom_scores, eval_inc,
                          bo_front_starts, _shift)
import numpy as np
from common import Area, DEFAULT_CKPT
from refine_v3 import SurrogateWL
from mscs_reclaim_v2 import prefetch_areas

KS = (1, 3, 5)


def rank_eval(nb, score, oracle_name):
    """nb: evaluated moves; score: dict name->score (higher = evaluate first)."""
    order = sorted(nb, key=lambda x: -score.get(x["name"], -np.inf))
    names = [x["name"] for x in order]
    best_eff = max(x["eff"] for x in nb if x["gain"] > 0)
    r = {"rank": names.index(oracle_name) + 1}
    for k in KS:
        top = [x for x in order[:k] if x["gain"] > 0]
        r[f"hit@{k}"] = oracle_name in names[:k]
        r[f"eff@{k}"] = max(x["eff"] for x in top) / best_eff if top else 0.0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--map-mode", default="loo", choices=["loo", "own", "all"])
    ap.add_argument("--src", default="results_v4_1007")
    ap.add_argument("--bo-queries", type=int, default=500)
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--tau", type=float, default=0.003)
    ap.add_argument("--mcpat-workers", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"prune_{args.bench}.json")
    spath = os.path.join(args.out, f"states_{args.bench}.json")
    for p in (path, spath):
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")

    R, static = load_map(args.map, args.bench, args.map_mode)
    pidx = {n: i for i, n in enumerate(PARAMS)}
    src = json.load(open(os.path.join(ROOT, args.src, args.bench, "bo_v4_mscs_s42.json")))
    starts, nfront = bo_front_starts(src, args.bo_queries, args.starts)
    S = SurrogateWL(DEFAULT_CKPT, [args.bench], "18k", device="cpu", max_slices=50)
    A = Area()
    pre = lambda cs: prefetch_areas(A, cs, args.mcpat_workers)
    t0 = time.time()
    print(f"[{args.bench}] front {nfront}, {len(starts)} starts x {args.steps} steps, map={args.map_mode}", flush=True)

    inc_rows, dec_rows, states = [], [], []
    for si, c in enumerate(starts):
        prev = None
        for st in range(args.steps):
            nb = eval_inc(c, S, A, pre)
            pos = [x for x in nb if x["gain"] > 0]
            if not pos:
                break
            m = S.mscs(c)
            oracle = max(pos, key=lambda x: x["eff"])
            rel, dom = relief_scores(m, R), dom_scores(m, R)
            dar = {x["name"]: x["darea"] for x in nb}
            sc = {"mscs_relief": {n: rel[pidx[n]] for n in dar},
                  "mscs_relief_area": {n: rel[pidx[n]] / dar[n] for n in dar},
                  "mscs_dom": {n: dom[pidx[n]] for n in dar},
                  "mscs_dom_area": {n: dom[pidx[n]] / dar[n] for n in dar},
                  "static": {n: static[pidx[n]] for n in dar},
                  "static_area": {n: static[pidx[n]] / dar[n] for n in dar},
                  "area_only": {n: -dar[n] for n in dar}}
            if prev is not None:
                sc["prev"] = {x["name"]: x["gain"] for x in prev}
                sc["prev_area"] = {x["name"]: x["eff"] for x in prev}
            inc_rows.append({"start": si, "step": st, "n": len(nb), "oracle": oracle["name"],
                             "best_gain": max(x["gain"] for x in pos),
                             **{k: rank_eval(nb, v, oracle["name"]) for k, v in sc.items()}})
            states.append({"start": si, "step": st, "cfg": c, "best_gain": inc_rows[-1]["best_gain"],
                           "oracle": oracle["name"]})

            # ---- A2: reclaim after the scalar increase ----
            cnew = oracle["cfg"]; cpi1, a1, m1 = S.cpi(cnew), A(cnew), S.mscs(cnew)
            rel1 = relief_scores(m1, R)
            dec = [(n, k, cd) for n in PARAMS if n != oracle["name"] for k in (1, 4)
                   for cd in [_shift(cnew, n, -k)] if cd is not None]
            pre([cd for _, _, cd in dec])
            cands = []
            for n, k, cd in dec:
                a = A(cd)
                if a is None or a >= a1:
                    continue
                cands.append({"param": n, "stride": k, "cfg": cd, "saved": a1 - a,
                              "pred": max(rel1[pidx[n]], 0) * k, "cost": (S.cpi(cd) - cpi1) / cpi1})
            ok = [x for x in cands if x["cost"] < args.tau]
            nxt = cnew
            if ok:
                tk = max(ok, key=lambda x: x["saved"])
                by_area = sorted(cands, key=lambda x: -x["saved"])
                med = np.median([x["pred"] for x in cands])
                by_mscs = ([x for x in by_area if x["pred"] <= med] + [x for x in by_area if x["pred"] > med])
                # first acceptable candidate reached in each order (what a sequential search would take)
                fa = next(x for x in by_area if x["cost"] < args.tau)
                fm = next(x for x in by_mscs if x["cost"] < args.tau)
                dec_rows.append({"start": si, "step": st, "n_cand": len(cands), "n_ok": len(ok),
                                 "oracle": f"{tk['param']}x{tk['stride']}",
                                 "q_area": by_area.index(tk) + 1, "q_mscs": by_mscs.index(tk) + 1,
                                 "first_mscs_saved_ratio": fm["saved"] / tk["saved"],
                                 "first_area_is_oracle": fa is tk})
                nxt = tk["cfg"]
            prev, c = nb, nxt
            print(f"  s{si} t{st}: +{oracle['name']} rank(relief_area)={inc_rows[-1]['mscs_relief_area']['rank']} "
                  f"rank(static_area)={inc_rows[-1]['static_area']['rank']} q={S.n_queries} [{time.time()-t0:.0f}s]",
                  flush=True)
        A.flush()

    json.dump(states, open(spath, "w"), indent=1)
    json.dump({"bench": args.bench, "args": vars(args), "inc": inc_rows, "dec": dec_rows,
               "wall": time.time() - t0, "queries": S.n_queries}, open(path, "w"), indent=1, default=str)
    print(f"[{args.bench}] done -> {path} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
