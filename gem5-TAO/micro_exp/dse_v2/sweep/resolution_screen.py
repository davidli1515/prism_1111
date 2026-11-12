#!/usr/bin/env python3
"""
Item 0 - resolution screen: which parameters does the surrogate actually resolve at SWEEP scale?

For every base config (6 benchmarks x N_BASE, 17-param space) and every free parameter p:
  dP = (CPI_PRISM(p=min) - CPI_PRISM(p=max)) / CPI_PRISM(base)      (relative, sign = benefit of max)
  dC = same with Concorde (independent model, 100% labels)
Per parameter (fixed rule, decided before running):
  effect      = median |dP|
  noise       = median |dP - dC|          (disagreement of two full-label models = noise floor proxy)
  sign_agree  = fraction of bases where sign(dP) == sign(dC) among |dP| >= 0.5%
  MEASURABLE  iff effect >= 1%  and  effect >= 2 x noise
Also records how often validity rules move coupled parameters at the endpoints.
Raw data -> results/resolution_raw.json, table -> results/resolution.txt (refuse to overwrite).
"""
import os, sys, json, time, argparse
from multiprocessing import get_context
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sweep_common as SC

EFFECT_MIN, SNR_MIN, SIGN_MIN_ABS = 0.01, 2.0, 0.005


def run_bench(args):
    bench, n_base, seed = args
    M = SC.Models(bench)
    rows = []
    for bi, c in enumerate(SC.bases(n_base, seed)):
        b = M(c)
        for p in SC.PARAMS:
            e = SC.endpoints(c, p)
            if e is None:
                continue
            lo, hi, info = e
            L, H = M(lo), M(hi)
            rows.append({"bench": bench, "base": bi, "param": p, "cur": c[p], **info,
                         "dP": (L["cpi_p"] - H["cpi_p"]) / b["cpi_p"], "dC": (L["cpi_c"] - H["cpi_c"]) / b["cpi_c"],
                         "dcyc": (L["cyc"] - H["cyc"]).tolist(), "cpi_base": b["cpi_p"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=12)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    out = os.path.join(HERE, "results"); os.makedirs(out, exist_ok=True)
    raw_p, txt_p = os.path.join(out, "resolution_raw.json"), os.path.join(out, "resolution.txt")
    for p in (raw_p, txt_p):
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    t0 = time.time()
    with get_context("spawn").Pool(len(SC.BENCHES)) as pool:
        rows = [r for rs in pool.map(run_bench, [(b, args.n_base, args.seed) for b in SC.BENCHES]) for r in rs]
    json.dump(rows, open(raw_p, "w"))
    lines = [f"resolution screen: {len(SC.BENCHES)} benchmarks x {args.n_base} bases, min/max sweep per parameter "
             f"[{time.time()-t0:.0f}s]",
             f"rule: MEASURABLE iff median|dP| >= {EFFECT_MIN:.0%} and median|dP| >= {SNR_MIN} x median|dP-dC|",
             f"  {'param':16s} {'n':>4s} {'median|dP|':>10s} {'p90|dP|':>8s} {'noise':>7s} {'SNR':>5s} "
             f"{'sign agree':>10s} {'coupled%':>8s}  verdict"]
    measurable = []
    for p in SC.PARAMS:
        x = [r for r in rows if r["param"] == p]
        if not x:
            lines.append(f"  {p:16s} no valid sweep"); continue
        dP = np.array([r["dP"] for r in x]); dC = np.array([r["dC"] for r in x])
        eff, noise = np.median(np.abs(dP)), np.median(np.abs(dP - dC))
        big = np.abs(dP) >= SIGN_MIN_ABS
        sa = np.mean(np.sign(dP[big]) == np.sign(dC[big])) if big.any() else np.nan
        cp = np.mean([len(r["coupled"]) > 0 for r in x])
        ok = eff >= EFFECT_MIN and eff >= SNR_MIN * noise
        measurable += [p] if ok else []
        lines.append(f"  {p:16s} {len(x):4d} {100*eff:9.2f}% {100*np.percentile(np.abs(dP),90):7.2f}% "
                     f"{100*noise:6.2f}% {eff/max(noise,1e-9):5.1f} {sa:10.2f} {100*cp:7.0f}%  "
                     f"{'MEASURABLE' if ok else '-'}")
    lines.append(f"\nmeasurable ({len(measurable)}): {measurable}")
    open(txt_p, "w").write("\n".join(lines) + "\n")
    json.dump({"measurable": measurable, "rule": {"effect_min": EFFECT_MIN, "snr_min": SNR_MIN}},
              open(os.path.join(out, "measurable.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
