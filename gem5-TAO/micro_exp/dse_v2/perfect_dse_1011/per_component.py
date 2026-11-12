#!/usr/bin/env python3
"""
Per-component difference decomposition of the stall stack (2026-10-11).

For each benchmark, the real-CPI best design at each area budget (from the rescored archives,
stacks/<bench>.npz) is decomposed with one finite difference per component (= per DSE parameter):
  theta_p          = theta with parameter p set to its maximum (projected to a valid config)
  relief_p[c, s]   = stall(theta)[c, s] - stall(theta_p)[c, s]     class c, stage s, CPI units
  gain_p           = (CPI(theta) - CPI(theta_p)) / CPI(theta)
  all              = every parameter at its maximum (bp_type: both values tried, better kept)
  residual[c, s]   = stall(all)                                     what no component removes
  interaction      = sum_p gain_p - gain_all   (< 0: hidden / masked components, > 0: overlapping)
Output: results/per_component.txt / .json  (refuses to overwrite)
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stack_eval import StackEval, DSE_V2
from common import ALL_PARAMS, get_param_values, config_key
from dse_engine import _project_to_valid
from analyze import load, BUDGETS

BENCHES = ["600.perlbench_s", "602.gcc_s", "605.mcf_s", "619.lbm_s", "620.omnetpp_s", "621.wrf_s",
           "623.xalancbmk_s", "631.deepsjeng_s", "638.imagick_s", "641.leela_s", "644.nab_s", "648.exchange2_s"]
CLS = ["FP", "BP", "SP"]


def maxed(c, p):
    vals = get_param_values(p)
    best = max(vals) if p != "bp_type" else None
    if p == "bp_type":
        return [_project_to_valid(dict(c, bp_type=v)) for v in vals if v != c["bp_type"]]
    cn = _project_to_valid(dict(c, **{p: best}))
    return [cn] if config_key(cn) != config_key(c) else []


def all_maxed(c):
    cn = dict(c)
    for p in ALL_PARAMS:
        if p != "bp_type":
            cn[p] = max(get_param_values(p))
    cn = _project_to_valid(cn)
    return [_project_to_valid(dict(cn, bp_type=v)) for v in get_param_values("bp_type")]


def main():
    out = os.path.join(HERE, "results", "per_component.txt")
    if os.path.exists(out):
        sys.exit(f"exists, refusing to overwrite: {out}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows = []
    for b in BENCHES:
        D = load(b)
        A, cpi = D["area"], D["cpi"][:, 0]
        designs = {}
        for B in BUDGETS:
            m = A <= B
            if m.any():
                i = int(np.where(m)[0][np.argmin(cpi[m])])
                designs.setdefault(i, []).append(B)
        E = StackEval([b]); E.enable_fast([D["configs"][next(iter(designs))]])
        for i, budgets in designs.items():
            c = D["configs"][i]
            c0, s0, _ = E.eval(c, b)
            r = {"bench": b, "budgets": budgets, "area": float(A[i]), "cpi": c0, "stall": s0.tolist(), "comp": {}}
            for p in ALL_PARAMS:
                cands = maxed(c, p)
                if not cands:
                    continue
                ev = min((E.eval(x, b) + (x,) for x in cands), key=lambda t: t[0])
                n_chg = sum(ev[3][k] != c[k] for k in c)
                r["comp"][p] = {"gain": (c0 - ev[0]) / c0, "relief": (s0 - ev[1]).tolist(), "n_changed": n_chg}
            ev = min((E.eval(x, b) for x in all_maxed(c)), key=lambda t: t[0])
            r["gain_all"] = (c0 - ev[0]) / c0
            r["residual"] = ev[1].tolist()
            rows.append(r)
        print(f"  {b}: {len(designs)} designs, {E.n} evaluations", flush=True)

    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))
    P(f"designs: {len(rows)} (best real-CPI design per area budget, 12 benchmarks)")
    P("\n1. per-component gain (maxing that component alone), mean over designs; relief at Disp / Com per class (share of CPI)")
    for p in ALL_PARAMS:
        g = [(r["comp"][p], r["cpi"]) for r in rows if p in r["comp"]]
        if not g:
            continue
        gain = np.array([x["gain"] for x, _ in g])
        rel = np.array([np.array(x["relief"]) / cp for x, cp in g])
        P(f"   {p:16s} n={len(g):3d} gain {100*gain.mean():+6.2f}% (max {100*gain.max():5.1f}%)  "
          + "  ".join(f"{CLS[k]} {rel[:, k, 1].mean():+.3f}/{rel[:, k, 3].mean():+.3f}" for k in range(3))
          + f"  multi-param projection {np.mean([x['n_changed'] > 1 for x, _ in g]):.2f}")
    P("\n2. tunable vs residual (all components maxed), share of CPI, stages Dec/Disp/Iss/Com")
    st = np.array([np.array(r["stall"]) / r["cpi"] for r in rows]); rs = np.array([np.array(r["residual"]) / r["cpi"] for r in rows])
    for k in range(3):
        P(f"   {CLS[k]} stall {np.round(st[:, k].mean(0), 3)}  residual {np.round(rs[:, k].mean(0), 3)}")
    P(f"   gain all maxed {100*np.mean([r['gain_all'] for r in rows]):.1f}%")
    P("\n3. interaction: sum of single-component gains - all-maxed gain")
    ia = np.array([sum(x["gain"] for x in r["comp"].values()) - r["gain_all"] for r in rows])
    P(f"   mean {100*ia.mean():+.1f}pp  hidden (<-1pp) {np.mean(ia < -0.01):.2f}  overlapping (>+1pp) {np.mean(ia > 0.01):.2f}")
    P("\n4. per benchmark: top-3 components by gain at the 40 mm^2 design; residual FP@Disp / BP@Com")
    for r in rows:
        if 40 in r["budgets"]:
            top = sorted(r["comp"].items(), key=lambda t: -t[1]["gain"])[:3]
            P(f"   {r['bench']:16s} cpi {r['cpi']:.3f} | " + ", ".join(f"{p} {100*x['gain']:+.1f}%" for p, x in top)
              + f" | all {100*r['gain_all']:+.1f}% | residual FP@Disp {r['residual'][0][1]/r['cpi']:.2f} BP@Com {r['residual'][1][3]/r['cpi']:.2f}")
    open(out, "w").write("\n".join(L) + "\n")
    json.dump(rows, open(out.replace(".txt", ".json"), "w"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
