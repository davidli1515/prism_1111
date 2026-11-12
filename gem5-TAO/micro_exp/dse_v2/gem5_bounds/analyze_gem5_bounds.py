#!/usr/bin/env python3
"""
Analysis of run_gem5_bounds.py results (gem5 truth) against PRISM predictions.
PRISM CPI / stacks for every simulated config are recomputed with the DSE surrogate (CPU, cheap).

  0. stack fidelity     base configs: |PRISM share - gem5 share| per class x stage; CPI error;
                        gem5 monotonic ordering (FP,SP fall / BP rises along the pipeline)
  1. bound coverage     (Eyerman Fig. 2) real gem5 gain of each maxed direction vs the interval
                        [min_s, max_s] of the touched classes, from the gem5 stack and from the
                        PRISM stack; error = 0 inside the interval, else distance to it.
                        "PRISM replaces the simulator" <=> PRISM error ~= gem5-stack error
  2. what-if gain       PRISM gain (CPI difference) vs gem5 gain: MAE, Spearman, sign agreement
  3. attribution        drop of (CPI x max-stage share) per class after each maxed move, gem5 vs
                        PRISM: do FP and BP fall together (position-based accounting)?

Usage: python analyze_gem5_bounds.py [--out gem5_bounds] [--no-prism]
"""
import os, sys, json, glob, argparse
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DSE_V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DSE_V2, "mscs_guide")); sys.path.insert(0, DSE_V2)
from summarize_bounds import TOUCH

CATS = ["Base", "FP", "BP", "SP"]
STAGES = ["Dec", "Disp", "Iss", "Com"]
F28 = (["dec_base", "dec_fe", "dec_be", "dec_sq"]
       + ["ren_base", "ren_fe", "ren_block", "ren_reg", "ren_serialize", "ren_sq"]
       + ["iew_base", "iew_fe", "iew_block", "iew_iqfull", "iew_lsqfull", "iew_sq"]
       + ["iq_base", "iq_fe", "iq_depend", "iq_mem", "iq_fu", "iq_sq"]
       + ["com_base", "com_fe", "com_fu", "com_mem", "com_depend", "com_sq"])


def shares(m16):
    m = np.asarray(m16, float)
    return {c: np.array([m[4 * s + k] for s in range(4)]) for k, c in enumerate(CATS)}


def interval(sh, d):
    return (sum(sh[c].min() for c in TOUCH[d]), min(1 - sh["Base"].min(), sum(sh[c].max() for c in TOUCH[d])))


def err(g, lo, hi):
    return 0.0 if lo <= g <= hi else (lo - g if g < lo else g - hi)


def prism_eval(runs):
    from refine_v3 import SurrogateWL
    from check_bounds import stack
    from common import DEFAULT_CKPT
    out, cache = {}, {}
    for b in sorted({r["bench"] for r in runs}):
        S = SurrogateWL(DEFAULT_CKPT, [b], "18k", device="cpu", max_slices=50)
        for r in runs:
            if r["bench"] == b:
                cpi, sh = stack(S, r["config"])
                out[(b, r["idx"], r["tag"])] = {"cpi": cpi, "sh": {c: np.array(v) for c, v in sh.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(args.out, "runs", "*", "*", "result.json"))]
    runs = [r for r in runs if r["error"] is None]
    print(f"{len(runs)} successful gem5 runs")
    if not runs:
        return
    P = prism_eval(runs)
    R = {(r["bench"], r["idx"], r["tag"]): r for r in runs}
    points = sorted({(r["bench"], r["idx"]) for r in runs if r["tag"] == "base"})

    # 0. stack fidelity
    print("\n=== 0. stack fidelity on base configs (n=%d) ===" % len(points))
    E, cerr, mono = [], [], []
    for b, i in points:
        g = shares(R[(b, i, "base")]["mscs16"]); p = P[(b, i, "base")]["sh"]
        E.append([[abs(p[c][s] - g[c][s]) for s in range(4)] for c in CATS])
        cerr.append(abs(P[(b, i, "base")]["cpi"] - R[(b, i, "base")]["cpi"]) / R[(b, i, "base")]["cpi"])
        mono.append([all(np.diff(g["FP"]) <= 0), all(np.diff(g["BP"]) >= 0), all(np.diff(g["SP"]) <= 0)])
    E = np.array(E).mean(0)
    print("  mean |PRISM - gem5| share   " + "  ".join(f"{s:>5s}" for s in STAGES))
    for k, c in enumerate(CATS):
        print(f"    {c:4s}                      " + "  ".join(f"{e:.3f}" for e in E[k]))
    print(f"  CPI MAPE {100*np.mean(cerr):.2f}%   gem5 monotonic FP/BP/SP: {np.mean(mono, 0).round(2)}")

    # 1-3 per direction
    rows = []
    for b, i in points:
        base, pb = R[(b, i, "base")], P[(b, i, "base")]
        gsh, psh = shares(base["mscs16"]), pb["sh"]
        for d in TOUCH:
            vs = [t for (bb, ii, t) in R if bb == b and ii == i and t.startswith(d) and t != "base"]
            if not vs:
                continue
            gg = {t: (base["cpi"] - R[(b, i, t)]["cpi"]) / base["cpi"] for t in vs}
            t = max(gg, key=gg.get)
            pg = (pb["cpi"] - P[(b, i, t)]["cpi"]) / pb["cpi"]
            glo, ghi = interval(gsh, d); plo, phi = interval(psh, d)
            ga = {c: (base["cpi"] * gsh[c].max() - R[(b, i, t)]["cpi"] * shares(R[(b, i, t)]["mscs16"])[c].max()) / base["cpi"]
                  for c in ["FP", "BP", "SP"]}
            pa = {c: (pb["cpi"] * psh[c].max() - P[(b, i, t)]["cpi"] * P[(b, i, t)]["sh"][c].max()) / pb["cpi"]
                  for c in ["FP", "BP", "SP"]}
            rows.append({"b": b, "i": i, "d": d, "g": gg[t], "p": pg, "glo": glo, "ghi": ghi, "plo": plo, "phi": phi,
                         "eg": err(gg[t], glo, ghi), "ep": err(gg[t], plo, phi),
                         "es": err(gg[t], 0.0, 1 - 1 / (base["cpi"] * min(base["config"][k] for k in
                                   ["fetchWidth", "decodeWidth", "dispatchWidth", "commitWidth"]))),
                         "ga": ga, "pa": pa})
    print("\n=== 1. bound coverage of the real gem5 gain (Eyerman Fig. 2) ===")
    print("  dir   n   gem5 gain  gem5 [lo,hi]      PRISM [lo,hi]     cover gem5/PRISM/scalar   mean err gem5/PRISM/scalar")
    for d in TOUCH:
        x = [r for r in rows if r["d"] == d]
        if not x:
            continue
        m = lambda k: np.mean([r[k] for r in x])
        cov = lambda k: np.mean([r[k] == 0 for r in x])
        print(f"  {d:4s} {len(x):3d}  {m('g'):+.3f}     [{m('glo'):.3f},{m('ghi'):.3f}]   [{m('plo'):.3f},{m('phi'):.3f}]   "
              f"{cov('eg'):.2f}/{cov('ep'):.2f}/{cov('es'):.2f}            {m('eg'):.3f}/{m('ep'):.3f}/{m('es'):.3f}")
    print("\n=== 2. what-if gain: PRISM vs gem5 ===")
    g = np.array([r["g"] for r in rows]); p = np.array([r["p"] for r in rows])
    print(f"  n={len(rows)}  MAE {np.mean(np.abs(g - p)):.3f}  Spearman {spearmanr(g, p).correlation:.2f}  "
          f"sign agreement {np.mean(np.sign(g) == np.sign(p)):.2f}")
    for d in TOUCH:
        x = [r for r in rows if r["d"] == d]
        if len(x) > 2:
            print(f"    {d:4s} gem5 {np.mean([r['g'] for r in x]):+.3f}  PRISM {np.mean([r['p'] for r in x]):+.3f}  "
                  f"Spearman {spearmanr([r['g'] for r in x], [r['p'] for r in x]).correlation:.2f}")
    print("\n=== 3. attribution of each maxed move: drop of CPI x max-stage share (FP / BP / SP) ===")
    for d in TOUCH:
        x = [r for r in rows if r["d"] == d]
        if x:
            print(f"  {d:4s} gem5 " + " / ".join(f"{np.mean([r['ga'][c] for r in x]):+.3f}" for c in ["FP", "BP", "SP"])
                  + "    PRISM " + " / ".join(f"{np.mean([r['pa'][c] for r in x]):+.3f}" for c in ["FP", "BP", "SP"]))
    json.dump(rows, open(os.path.join(args.out, "analysis_rows.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
