#!/usr/bin/env python3
"""
Offline metrics for results/bounds_<bench>.json (check_bounds.py). No queries.

Bound for direction d at point p (relative CPI gain):
  mscs     sum over the classes d can touch of max_s share_c,s   (TOUCH below; capped at 1-Base)
  mscs_lb  same with min_s (pessimistic)
  scalar   1 - Base = 1 - 1/(CPI*W_min) for EVERY direction: needs only CPI -> what scalar knows
MSCS has value here only where it beats `scalar`.

  1. validity   best realized gain in d (all evaluated configs) > bound ?   (+ attribution: which
                classes actually fell, to tell mapping errors from bound failures)
  2. pruning    optimistic corner (cpi_p (1-bound), area_p + smallest step of d) has HVI 0 vs the
                current front -> d pruned at 0 queries; false prune = realized HVI > 0
  3. allocation (a) per point, pick the direction with max optimistic HVI: hit rate of the truly best
                direction vs chance and vs "always the most frequent best direction";
                (b) across points: Spearman(max_d optimistic HVI, max_d realized HVI), mscs vs scalar
"""
import os, glob, json
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, os.path.dirname(HERE))
from common import hypervolume

TOUCH = {"FE": ["FP"], "L1": ["FP", "BP"], "BE": ["BP"], "SP": ["SP", "FP"], "ALL": ["FP", "BP", "SP"]}
DIRS = list(TOUCH)


def bound(pt, d, kind):
    sh = pt["shares"]
    cap = 1 - min(sh["Base"])
    if kind == "scalar":
        return 1 - 1 / (pt["cpi"] * pt["w_min"])
    f = max if kind == "mscs" else min
    return min(cap, sum(f(sh[c]) for c in TOUCH[d]))


def hvi(F, pts, ref):
    if not len(pts):
        return 0.0
    return hypervolume(np.vstack([F, np.atleast_2d(pts)]), ref) - hypervolume(F, ref)


def build(z):
    rows = []
    ref = z["ref"]
    for pt in z["points"]:
        F = np.array(z["fronts"][pt["front"]])
        for d in DIRS:
            ev = pt["dirs"][d]
            steps = [e["area"] - pt["area"] for e in ev if e["tag"] != "max" and e["area"] > pt["area"]]
            dmin = min(steps) if steps else 0.0
            gains = [(pt["cpi"] - e["cpi"]) / pt["cpi"] for e in ev]
            best = int(np.argmax(gains)) if ev else None
            # attribution of the best realized move: drop of each class's max-stage share x CPI
            attr = None
            if best is not None:
                e = ev[best]
                attr = {c: (pt["cpi"] * max(pt["shares"][c]) - e["cpi"] * max(e["shares"][c])) / pt["cpi"]
                        for c in ["FP", "BP", "SP"]}
            r = {"bench": z["bench"], "front": pt["front"], "idx": pt["idx"], "dir": d,
                 "best_gain": max(gains) if ev else None, "attr": attr, "n_eval": len(ev),
                 "real_hvi": hvi(F, [(e["cpi"], e["area"]) for e in ev], ref)}
            for k in ["mscs", "mscs_lb", "scalar"]:
                b = bound(pt, d, k)
                r[f"b_{k}"] = b
                r[f"opt_{k}"] = hvi(F, [(pt["cpi"] * (1 - b), pt["area"] + dmin)], ref)
            rows.append(r)
    return rows


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
    R = []
    for f in sorted(glob.glob(os.path.join(d, "bounds_*.json"))):
        R += build(json.load(open(f)))
    print(f"=== {len(R)} (point x direction) rows, benches: {sorted({r['bench'] for r in R})} ===")
    print(f"TOUCH = {TOUCH}")
    for fr in ["bo500", "final"]:
        rr = [r for r in R if r["front"] == fr and r["n_eval"] > 0]
        print(f"\n##### front = {fr}  ({len({(r['bench'], r['idx']) for r in rr})} points)")
        print("  1. validity: best realized gain vs bound (violation = gain > bound)")
        print("     dir    n   gain   b_mscs  viol   b_lb   gain<lb   b_scalar viol_s   gain/b_mscs   attribution of best move (FP/BP/SP)")
        for d in DIRS:
            x = [r for r in rr if r["dir"] == d]
            g = np.array([r["best_gain"] for r in x]); bm = np.array([r["b_mscs"] for r in x])
            bl = np.array([r["b_mscs_lb"] for r in x]); bs = np.array([r["b_scalar"] for r in x])
            at = np.array([[r["attr"][c] for c in ["FP", "BP", "SP"]] for r in x])
            print(f"     {d:4s} {len(x):4d}  {g.mean():+.3f}  {bm.mean():.3f}  {np.mean(g > bm):.2f}  {bl.mean():.3f}  "
                  f"{np.mean(g < bl):.2f}      {bs.mean():.3f}    {np.mean(g > bs):.2f}     "
                  f"{np.median(g / np.maximum(bm, 1e-9)):+.2f}          {at.mean(0).round(3)}")
        print("  2. pruning (optimistic corner HVI == 0)")
        for k in ["mscs", "scalar"]:
            for d in DIRS:
                x = [r for r in rr if r["dir"] == d]
                pr = [r for r in x if r[f"opt_{k}"] <= 0]
                print(f"     {k:6s} {d:4s} pruned {len(pr):3d}/{len(x):3d}  false {sum(r['real_hvi'] > 0 for r in pr):2d}  "
                      f"queries saved {sum(r['n_eval'] for r in pr):4d}  (dirs with realized HVI>0: {sum(r['real_hvi'] > 0 for r in x)})")
        print("  3a. per point: does max optimistic HVI pick the truly best direction (FE/L1/BE/SP)?")
        single = [d for d in DIRS if d != "ALL"]
        pts = {}
        for r in rr:
            if r["dir"] in single:
                pts.setdefault((r["bench"], r["idx"]), []).append(r)
        good = {k: v for k, v in pts.items() if max(r["real_hvi"] for r in v) > 0}
        truth = {k: max(v, key=lambda r: r["real_hvi"])["dir"] for k, v in good.items()}
        maj = max(single, key=lambda d: sum(t == d for t in truth.values()))
        for k in ["mscs", "mscs_lb", "scalar"]:
            hit = np.mean([max(v, key=lambda r: r[f"opt_{k}"])["dir"] == truth[key] for key, v in good.items()])
            print(f"     {k:8s} hit {hit:.2f}")
        print(f"     always '{maj}' hit {np.mean([t == maj for t in truth.values()]):.2f}   chance {1/len(single):.2f}   "
              f"(n={len(good)}; best-dir counts {dict((d, sum(t == d for t in truth.values())) for d in single)})")
        print("  3b. across points: Spearman(max_d optimistic HVI, max_d realized HVI)")
        for k in ["mscs", "scalar"]:
            a = [max(r[f"opt_{k}"] for r in v) for v in pts.values()]
            b = [max(r["real_hvi"] for r in v) for v in pts.values()]
            s = spearmanr(a, b)
            print(f"     {k:6s} rho={s.correlation:.2f} (p={s.pvalue:.3f}, n={len(a)})")
    print("\n  per bench (bo500): violation mscs / pruned mscs vs scalar / false prunes mscs")
    for b in sorted({r["bench"] for r in R}):
        x = [r for r in R if r["bench"] == b and r["front"] == "bo500" and r["n_eval"] > 0]
        pm = [r for r in x if r["opt_mscs"] <= 0]; ps = [r for r in x if r["opt_scalar"] <= 0]
        print(f"    {b:16s} viol {np.mean([r['best_gain'] > r['b_mscs'] for r in x]):.2f}  pruned {len(pm)} vs {len(ps)} "
              f"/ {len(x)}  false {sum(r['real_hvi'] > 0 for r in pm)}")


if __name__ == "__main__":
    main()
