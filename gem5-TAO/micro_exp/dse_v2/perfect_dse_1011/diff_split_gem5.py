#!/usr/bin/env python3
"""
Difference-based split of the stall stack, PRISM vs gem5 (2026-10-11).  Reads gem5_bounds/runs
(read only); PRISM stacks via stack_eval.StackEval (same surrogate as gem5_bounds).

For every BO-front point p with gem5 runs of the base config and of resource groups maxed
(gem5_bounds tags: FE0 fetch buffer + L1I assoc, L10 L1 size, BE0 window + FUs + L1D/L2, SP0 other
branch predictor, ALL0/ALL1 every non-width resource maxed, with either predictor):
  comp(x)[c, s]   = stall of class c (FP, BP, SP) at stage s, in CPI units (gem5: CPI * share)
  relief_d        = comp(base) - comp(group d maxed)            what group d removes, per class/stage
  residual        = comp(best ALL config)                       what no resource in the space removes
Checks (PRISM vs gem5):
  1. total gain of each group (scalar level, what Concorde could also give)
  2. relief per class and stage
  3. residual per class and stage, and the relievable fraction of each class
  4. interaction: sum of single-group gains vs the ALL gain (hidden > 0, overlapping < 0)
Output: results/diff_split_gem5.txt / .json   (refuses to overwrite)
"""
import os, sys, json, glob, collections
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stack_eval import StackEval, DSE_V2

RUNS = os.path.join(DSE_V2, "gem5_bounds", "runs")
CLS, STG = ["FP", "BP", "SP"], ["Dec", "Disp", "Iss", "Com"]
GROUPS = ["FE", "L1", "BE", "SP"]


def gem5_comp(r):
    m = np.array(r["mscs16"]).reshape(4, 4)          # [stage, cat]
    return r["cpi"] * m[:, 1:].T, r["cpi"] * m[:, 0]  # (3,4), (4,)


def main():
    out_txt = os.path.join(HERE, "results", "diff_split_gem5.txt")
    if os.path.exists(out_txt):
        sys.exit(f"exists, refusing to overwrite: {out_txt}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    runs = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(RUNS, "*", "*", "result.json")):
        r = json.load(open(f))
        if r.get("error") is None:
            runs[(r["bench"], r["idx"])][r["tag"]] = r
    pts = {k: v for k, v in runs.items() if "base" in v}
    rows = []
    for b in sorted({k[0] for k in pts}):
        E = StackEval([b])
        keys = [k for k in pts if k[0] == b]
        E.enable_fast([pts[keys[0]]["base"]["config"]])
        for k in keys:
            R = pts[k]
            P = {t: E.eval(r["config"], b) for t, r in R.items()}          # (cpi, comp(3,4), base(4))
            G = {t: (r["cpi"],) + gem5_comp(r) for t, r in R.items()}
            row = {"bench": b, "idx": k[1], "tags": sorted(R)}
            for name, S in (("gem5", G), ("prism", P)):
                cpi0, c0, _ = S["base"]
                d = {"cpi": cpi0, "stall": c0.tolist()}
                for g in GROUPS:
                    t = next((t for t in S if t.startswith(g) and t[len(g):].isdigit()), None)
                    if t:
                        d[f"gain_{g}"] = (cpi0 - S[t][0]) / cpi0
                        d[f"relief_{g}"] = (c0 - S[t][1]).tolist()
                alls = [t for t in S if t.startswith("ALL")]
                if alls:
                    ta = min(alls, key=lambda t: S[t][0])
                    d["gain_ALL"] = (cpi0 - S[ta][0]) / cpi0
                    d["residual"] = S[ta][1].tolist()
                row[name] = d
            rows.append(row)
        print(f"  {b}: {len(keys)} points", flush=True)

    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))
    P(f"points: {len(rows)} (12 benchmarks x ~10 BO-front points)")
    P("\n1. total gain of each resource group (scalar level): gem5 vs PRISM")
    for g in GROUPS + ["ALL"]:
        xs = [(r["gem5"][f"gain_{g}"], r["prism"][f"gain_{g}"]) for r in rows if f"gain_{g}" in r["gem5"]]
        if xs:
            a = np.array(xs)
            P(f"   {g:3s} n={len(a):3d}  gem5 {100*a[:,0].mean():+5.1f}%  PRISM {100*a[:,1].mean():+5.1f}%  "
              f"MAE {100*np.abs(a[:,0]-a[:,1]).mean():4.1f}pp  Spearman {spearmanr(a[:,0], a[:,1]).correlation:.2f}")
    P("\n2. relief per class (CPI units, normalized by base CPI), stages Dec/Disp/Iss/Com: gem5 | PRISM | Spearman")
    for g in GROUPS:
        sel = [r for r in rows if f"relief_{g}" in r["gem5"]]
        if not sel:
            continue
        rg = np.array([np.array(r["gem5"][f"relief_{g}"]) / r["gem5"]["cpi"] for r in sel])
        rp = np.array([np.array(r["prism"][f"relief_{g}"]) / r["prism"]["cpi"] for r in sel])
        for c in range(3):
            rho = [spearmanr(rg[:, c, s], rp[:, c, s]).correlation for s in range(4)]
            P(f"   {g:3s} {CLS[c]}  gem5 {np.round(rg[:, c].mean(0), 3)}  PRISM {np.round(rp[:, c].mean(0), 3)}  "
              f"rho {np.round(rho, 2)}")
    sel = [r for r in rows if "residual" in r["gem5"]]
    rg = np.array([np.array(r["gem5"]["residual"]) / r["gem5"]["cpi"] for r in sel])
    rp = np.array([np.array(r["prism"]["residual"]) / r["prism"]["cpi"] for r in sel])
    sg = np.array([np.array(r["gem5"]["stall"]) / r["gem5"]["cpi"] for r in sel])
    sp = np.array([np.array(r["prism"]["stall"]) / r["prism"]["cpi"] for r in sel])
    P(f"\n3. residual after maxing every non-width resource (n={len(sel)}), share of base CPI, stages Dec/Disp/Iss/Com")
    for c in range(3):
        rho = [spearmanr(rg[:, c, s], rp[:, c, s]).correlation for s in range(4)]
        P(f"   {CLS[c]} base stall gem5 {np.round(sg[:, c].mean(0), 3)} PRISM {np.round(sp[:, c].mean(0), 3)}")
        P(f"      residual   gem5 {np.round(rg[:, c].mean(0), 3)} PRISM {np.round(rp[:, c].mean(0), 3)}  rho {np.round(rho, 2)}")
        fg = 1 - rg[:, c, [1, 3]].sum(1) / np.maximum(sg[:, c, [1, 3]].sum(1), 1e-9)
        fp = 1 - rp[:, c, [1, 3]].sum(1) / np.maximum(sp[:, c, [1, 3]].sum(1), 1e-9)
        P(f"      relievable fraction (Disp+Com) gem5 {fg.mean():.2f} PRISM {fp.mean():.2f}  Spearman {spearmanr(fg, fp).correlation:.2f}")
    P("\n4. interaction: sum of single-group gains - ALL gain (>0 overlapping, <0 hidden)")
    ia = [[sum(r[s].get(f"gain_{g}", 0) for g in GROUPS) - r[s]["gain_ALL"] for s in ("gem5", "prism")]
          for r in sel]
    a = np.array(ia)
    P(f"   gem5 mean {100*a[:,0].mean():+.1f}pp  PRISM {100*a[:,1].mean():+.1f}pp  Spearman {spearmanr(a[:,0], a[:,1]).correlation:.2f}  "
      f"same sign {np.mean(np.sign(a[:,0]) == np.sign(a[:,1])):.2f}  hidden(gem5) {np.mean(a[:,0] < -0.01):.2f}")
    open(out_txt, "w").write("\n".join(L) + "\n")
    json.dump(rows, open(out_txt.replace(".txt", ".json"), "w"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
