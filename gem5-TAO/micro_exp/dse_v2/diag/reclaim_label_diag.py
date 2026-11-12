#!/usr/bin/env python3
"""
Decision-level diagnostic for PROTECT: which reclaims are later re-added, and which
signal predicts that?  No DSE is run; nothing existing is modified.

From the pure-BO front of an existing v4 run (same start as scalar/v4), walk a few steps
with the SCALAR rule (increase by gain/area, reclaim = max area saved among cost<tau).
At every step, for each acceptable reclaim candidate cd (after the increase -> cnew):
  label  readd  = the scalar next increase from cd is the reclaimed parameter
  scores cost        scalar CPI cost of the reclaim (baseline predictor, no MSCS)
         l1          ||MSCS(cd)-MSCS(cnew)||_1             (current protect, v3/v4)
         dom_cat_up  increase of the dominant pressure category (FP/BP/SP summed over stages)
         dom_comp_up increase of the dominant single pressure component (of 12)
         max_cat_up  largest increase of any pressure category
         target_up   increase of the component the next increase (from cnew) relieves most
Report: ROC-AUC of each score for readd, on cost>=0 candidates (the ones protect decides)
and on all acceptable candidates; plus readd rate of the reclaim scalar actually takes.

Usage: python diag/reclaim_label_diag.py --bench 638.imagick_s --out diag/reclaim_label
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from common import Area, ALL_PARAMS, DEFAULT_CKPT, pareto_mask
from refine_v3 import SurrogateWL, _shift
from mscs_reclaim_v2 import prefetch_areas

CATS = {"FP": [1, 5, 9, 13], "BP": [2, 6, 10, 14], "SP": [3, 7, 11, 15]}
PRESSURE = CATS["FP"] + CATS["BP"] + CATS["SP"]
SCORES = ["cost", "l1", "dom_cat_up", "dom_comp_up", "max_cat_up", "target_up"]


def best_inc(c, S, A, pre, exclude=None):
    """Scalar increase rule from c: argmax gain/area_rel among CPI-improving +1 moves."""
    cpi0, a0 = S.cpi(c), A(c)
    inc = [(n, cn) for n in ALL_PARAMS if n != exclude for cn in [_shift(c, n, +1)] if cn is not None]
    pre([cn for _, cn in inc])
    cand = []
    for n, cn in inc:
        a = A(cn)
        if a is None:
            continue
        g = (cpi0 - S.cpi(cn)) / cpi0
        if g > 0:
            cand.append((n, cn, g, max(a - a0, 1e-6) / a0))
    if not cand:
        return None
    n, cn, g, ar = max(cand, key=lambda t: t[2] / t[3])
    return {"name": n, "cfg": cn, "gain": g, "eff": g / ar, "best_gain": max(t[2] for t in cand)}


def auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return None
    r = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort").astype(float)
    for v in np.unique(s):                       # average ranks over ties
        m = s == v; r[m] = r[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--src", default="results_v4_1007")
    ap.add_argument("--bo-queries", type=int, default=500)
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--max-label", type=int, default=15, help="label at most this many candidates per step (largest area saved)")
    ap.add_argument("--tau", type=float, default=0.003)
    ap.add_argument("--mcpat-workers", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(HERE, "reclaim_label"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.bench}.json")
    if os.path.exists(path):
        sys.exit(f"exists, refusing to overwrite: {path}")

    root = os.path.dirname(HERE)
    src = json.load(open(os.path.join(root, args.src, args.bench, "bo_v4_mscs_s42.json")))
    idx = [i for i, (t, q) in enumerate(zip(src["tags"], src["queries"]))
           if t.startswith("bo") and q <= args.bo_queries]
    P = np.c_[np.array(src["cpis"])[idx], np.array(src["areas"])[idx]]
    front = sorted((idx[j] for j in np.where(pareto_mask(P))[0]), key=lambda i: src["cpis"][i])
    pick = np.unique(np.linspace(0, len(front) - 1, min(args.starts, len(front))).round().astype(int))
    starts = [src["configs"][front[j]] for j in pick]

    S = SurrogateWL(DEFAULT_CKPT, [args.bench], "18k", device="cpu", max_slices=50)
    A = Area()
    pre = lambda cs: prefetch_areas(A, cs, args.mcpat_workers)
    t0 = time.time()
    print(f"[{args.bench}] BO front {len(front)} pts, {len(starts)} starts x {args.steps} steps", flush=True)

    rows, chosen = [], []
    for si, c in enumerate(starts):
        for st in range(args.steps):
            up = best_inc(c, S, A, pre)
            if up is None:
                break
            name, cnew = up["name"], up["cfg"]
            cpi1, a1, m1 = S.cpi(cnew), A(cnew), S.mscs(cnew)
            nxt0 = best_inc(cnew, S, A, pre)                     # next increase if no reclaim
            cat1 = {k: m1[v].sum() for k, v in CATS.items()}
            dom_cat = max(cat1, key=cat1.get)
            dom_comp = PRESSURE[int(np.argmax(m1[PRESSURE]))]
            if nxt0 is not None:
                r = m1 - S.mscs(nxt0["cfg"])
                tgt = PRESSURE[int(np.argmax(r[PRESSURE]))]
            else:
                tgt = dom_comp
            dec = [(n, k, cd) for n in ALL_PARAMS if n != name for k in (1, 4)
                   for cd in [_shift(cnew, n, -k)] if cd is not None]
            pre([cd for _, _, cd in dec])
            acc = []
            for n, k, cd in dec:
                a = A(cd)
                if a is None or a >= a1:
                    continue
                cost = (S.cpi(cd) - cpi1) / cpi1
                if cost < args.tau:
                    acc.append((n, k, cd, cost, a1 - a))
            if not acc:
                c = cnew
                continue
            take = max(acc, key=lambda t: t[4])
            acc.sort(key=lambda t: -t[4])
            for n, k, cd, cost, saved in acc[:args.max_label]:
                md = S.mscs(cd); d = md - m1
                catd = {kk: d[v].sum() for kk, v in CATS.items()}
                nx = best_inc(cd, S, A, pre)
                rows.append({"start": si, "step": st, "inc": name, "param": n, "stride": k,
                             "cost": cost, "area_saved": saved, "taken": n == take[0] and k == take[1],
                             "l1": float(np.abs(d).sum()), "dom_cat": dom_cat,
                             "dom_cat_up": float(catd[dom_cat]), "dom_comp_up": float(d[dom_comp]),
                             "max_cat_up": float(max(catd.values())), "target_up": float(d[tgt]),
                             "next": nx and nx["name"], "next_gain": nx and nx["gain"],
                             "next0": nxt0 and nxt0["name"], "readd": bool(nx and nx["name"] == n)})
            last = [r for r in rows if r["taken"] and r["start"] == si and r["step"] == st]
            chosen += last
            c = take[2]
            print(f"  start {si} step {st}: +{name} -{take[0]}x{take[1]} acc={len(acc)} "
                  f"readd(taken)={last[0]['readd'] if last else None} q={S.n_queries} [{time.time()-t0:.0f}s]", flush=True)
        A.flush()

    summ = {}
    for sub, R in [("cost_ge0", [r for r in rows if r["cost"] >= 0]), ("all", rows)]:
        y = [r["readd"] for r in R]
        summ[sub] = {"n": len(R), "readd_rate": float(np.mean(y)) if R else None,
                     "auc": {s: auc(y, [r[s] for r in R]) for s in SCORES} if R else {}}
    summ["taken"] = {"n": len(chosen), "readd_rate": float(np.mean([r["readd"] for r in chosen])) if chosen else None}
    json.dump({"bench": args.bench, "args": vars(args), "summary": summ, "rows": rows,
               "wall": time.time() - t0, "queries": S.n_queries}, open(path, "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)
    print(f"[{args.bench}] done -> {path} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
