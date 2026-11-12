#!/usr/bin/env python3
"""
Collect the loss-weight sweep (100% labels, exp1 setting) into one JSON.

For every (lam_task, lam_const, lam_mono) it stores per-seed values and mean/std of:
  cpi_mape        CPI MAPE (%), 4-stage mean CPI, selected checkpoint
  component_mae   mean MAE of the FP/BP/SP proportions (all stages)
  base_incons     cross-stage Base inconsistency (%): (max - min)/mean of the 4 predicted
                  Base values, averaged over samples (0 = all stages imply the same CPI)
  mono_viol       monotonicity violation rate (%): predictions where FP or SP increases, or
                  BP decreases, from Decode to Commit by more than 0.01
Sources: results/lambda_sweep (lam_task <= 10) and results/lambda_probe (lam_task 30, 100);
input = 4681 features (3873 Concorde + 808 D3).
--nod3: the same 29 configs x 3 seeds rerun with the first 3873 (Concorde) features only
        (results/lambda_sweep_nod3, files *_nod3.json) -> results/lambda_table_nod3_1026.json

Usage: python collect_lambda_table_1012.py [--out results/lambda_table_1012.json]
       python collect_lambda_table_1012.py --nod3 [--out results/lambda_table_nod3_1026.json]
"""
import os, re, sys, glob, json, argparse
import numpy as np
from collections import defaultdict

R = "/path/to/gem5/micro_exp/exp1/results"
SOURCES = [f"{R}/lambda_sweep", f"{R}/lambda_probe"]
SOURCES_NOD3 = [f"{R}/lambda_sweep_nod3"]
PAT = re.compile(r"t([\dp]+)_c([\dp]+)_m([\dp]+)_s(\d+)\.json$")
PAT_NOD3 = re.compile(r"t([\dp]+)_c([\dp]+)_m([\dp]+)_s(\d+)_nod3\.json$")
num = lambda s: float(s.replace("p", "."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nod3", action="store_true", help="collect the 3873-dim (no D3) rerun")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"{R}/lambda_table_nod3_1026.json" if args.nod3 else f"{R}/lambda_table_1012.json"
    sources, pat = (SOURCES_NOD3, PAT_NOD3) if args.nod3 else (SOURCES, PAT)

    acc = defaultdict(list)
    for src in sources:
        for p in sorted(glob.glob(f"{src}/t*_c*_m*_s*.json")):
            m = pat.search(p)
            if m is None:
                continue
            t, c, mo, seed = num(m.group(1)), num(m.group(2)), num(m.group(3)), int(m.group(4))
            s = json.load(open(p))["selected"]
            acc[(t, c, mo)].append(dict(
                seed=seed, file=p,
                cpi_mape=s["mape_mean4"],
                component_mae=float(np.mean([s["mae"][k] for k in ("FP", "BP", "SP")])),
                base_incons=s["base_spread_rel"] * 100,
                mono_viol=s["mono_viol_gt0.01"]))

    rows = []
    for (t, c, mo), runs in sorted(acc.items()):
        row = dict(lam_task=t, lam_const=c, lam_mono=mo, n_seeds=len(runs),
                   seeds=sorted(r["seed"] for r in runs), files=[r["file"] for r in runs])
        for k in ("cpi_mape", "component_mae", "base_incons", "mono_viol"):
            v = [r[k] for r in runs]
            row[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v}
        # CPI MAPE reported with 1 decimal (mean computed from unrounded seeds)
        row["cpi_mape"] = {k: (round(v, 1) if k != "per_seed" else [round(x, 1) for x in v])
                           for k, v in row["cpi_mape"].items()}
        rows.append(row)

    meta = {"input": ("first 3873 (Concorde) features, 808 D3 features removed" if args.nod3
                      else "4681 features = 3873 Concorde + 808 D3"),
            "setting": "exp1, 18k dataset, row split seed 42, 100% labels, [512,256], 1521 epochs",
            "selection": "checkpoint with best test CPI MAPE (as in exp1)",
            "metrics": {"cpi_mape": "CPI MAPE (%), mean of the 4 per-stage CPIs",
                        "component_mae": "mean MAE of FP/BP/SP proportions over all stages",
                        "base_incons": "cross-stage Base inconsistency (%), (max-min)/mean of the 4 Base values",
                        "mono_viol": "monotonicity violation rate (%), FP/SP increase or BP decrease "
                                     "from Decode to Commit by > 0.01"},
            "sources": sources}
    json.dump({"meta": meta, "rows": rows}, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}: {len(rows)} configurations")
    for r in rows:
        print(f"  t={r['lam_task']:<5g} c={r['lam_const']:<4g} m={r['lam_mono']:<5g} n={r['n_seeds']} | "
              f"CPI {r['cpi_mape']['mean']:.1f} | comp {r['component_mae']['mean']:.3f} | "
              f"incons {r['base_incons']['mean']:.2f} | viol {r['mono_viol']['mean']:.1f}")


if __name__ == "__main__":
    main()
