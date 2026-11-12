#!/usr/bin/env python3
"""
Offline analysis of precheck_raw.json.  Score = how good a move is (higher = better).
Methods
  scalar        Delta CPI (4-stage mean CPI)
  stage_med     median over the 4 stages of the per-stage Delta CPI
  stage_agree   mean per-stage Delta CPI x fraction of stages that agree on its sign
  dom_share     reduction of the dominant type (4-stage mean of UNWEIGHTED shares)      [paper rule]
  dom_share_w   same with CPI-weighted shares
  dom_cyc       reduction of the dominant type in CYCLES (cpi_stage x share_w, 4-stage mean)
  eyer_cyc      dominant type in cycles at its own stage (FP/SP@Decode, BP@Commit)
Views
  A  agreement with the TEACHER'S Delta-CPI ranking (PRISM 100%)
  B  label stability: method at r% vs the SAME method computed by PRISM 100%
  C  decision value: teacher Delta-CPI of the move each method ranks first, / best teacher Delta-CPI
  D  dominant-type agreement with the teacher, and how often it is BP
"""
import os, json, itertools
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
FP, BP, SP = [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]
TYPES = {"FP": FP, "BP": BP, "SP": SP}
OWN = {"FP": 1, "BP": 14, "SP": 3}


def dominant(p, field="share"):
    v = np.array(p[field]); return max(TYPES, key=lambda t: v[TYPES[t]].mean())


def scores(pb, pm, method):
    if method == "scalar":
        return np.array([(pb["cpi"] - q["cpi"]) / pb["cpi"] for q in pm])
    if "cpi_stage" not in pb:
        return None
    sb = np.array(pb["cpi_stage"]); D = np.array([(sb - np.array(q["cpi_stage"])) / sb for q in pm])  # (n,4)
    if method == "stage_med":
        return np.median(D, 1)
    if method == "stage_agree":
        return D.mean(1) * np.abs(np.sign(D).mean(1))
    f = {"dom_share": "share", "dom_share_w": "share_w", "dom_cyc": "cyc", "eyer_cyc": "cyc"}[method]
    t = dominant(pb, "share_w")
    if method == "dom_share":
        t = dominant(pb, "share")
    vb = np.array(pb[f])
    if method == "eyer_cyc":
        j = OWN[t]; return np.array([vb[j] - np.array(q[f])[j] for q in pm])
    return np.array([np.mean(vb[TYPES[t]] - np.array(q[f])[TYPES[t]]) for q in pm])


def spear(a, b):
    return spearmanr(a, b).correlation if np.std(a) > 0 and np.std(b) > 0 else np.nan


def main():
    recs = json.load(open(os.path.join(HERE, "precheck_raw.json")))
    models = list(recs[0]["pred"])
    methods = ["scalar", "stage_med", "stage_agree", "dom_share", "dom_share_w", "dom_cyc", "eyer_cyc"]
    T = "PRISM_100"
    res = {}
    for m in models:
        for meth in methods:
            A, B, Cv, top3 = [], [], [], []
            for r in recs:
                p = r["pred"][m]; tp = r["pred"][T]
                s = scores(p[0], p[1:], meth)
                if s is None:
                    continue
                t_cpi = scores(tp[0], tp[1:], "scalar")
                t_same = scores(tp[0], tp[1:], meth)
                A.append(spear(s, t_cpi)); B.append(spear(s, t_same))
                best = t_cpi.max()
                Cv.append(t_cpi[np.argmax(s)] / best if best > 0 else np.nan)
                top3.append(float(np.argmax(t_cpi) in np.argsort(-s)[:3]))
            if A:
                res[(m, meth)] = (np.nanmean(A), np.nanmean(B), np.nanmean(Cv), np.mean(top3))
    print("A = Spearman with teacher Delta-CPI ranking | B = Spearman with the SAME method at PRISM 100% "
          "(label stability) | C = teacher Delta-CPI of the chosen move / best | top3 = teacher-best in top-3")
    print(f"  {'model':12s} {'method':12s} {'A':>6s} {'B':>6s} {'C':>6s} {'top3':>6s}")
    for m in models:
        for meth in methods:
            if (m, meth) in res:
                a, b, c, t3 = res[(m, meth)]
                print(f"  {m:12s} {meth:12s} {a:6.2f} {b:6.2f} {c:6.2f} {t3:6.2f}")
        print()
    print("D  dominant type at the base config (share_w): agreement with teacher / share of BP")
    for m in models:
        if "share_w" not in recs[0]["pred"][m][0]:
            continue
        ag = np.mean([dominant(r["pred"][m][0], "share_w") == dominant(r["pred"][T][0], "share_w") for r in recs])
        bp = np.mean([dominant(r["pred"][m][0], "share_w") == "BP" for r in recs])
        print(f"  {m:12s} agree {ag:.2f}  BP {bp:.2f}")
    print("\nE  size of the signal: teacher |Delta CPI| of the 16 moves (median over bases): "
          f"best {np.median([scores(r['pred'][T][0], r['pred'][T][1:], 'scalar').max() for r in recs])*100:.2f}%  "
          f"median {np.median([np.median(np.abs(scores(r['pred'][T][0], r['pred'][T][1:], 'scalar'))) for r in recs])*100:.3f}%")


if __name__ == "__main__":
    main()
