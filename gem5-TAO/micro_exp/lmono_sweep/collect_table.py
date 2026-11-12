#!/usr/bin/env python3
import os, re, sys, glob, json, argparse
import numpy as np
from collections import defaultdict

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SOURCES = [R]
SOURCES_NOD3 = [R]
PAT = re.compile(r"t([\dp]+)_c([\dp]+)_m([\dp]+)_s(\d+)\.json$")
PAT_NOD3 = re.compile(r"t([\dp]+)_c([\dp]+)_m([\dp]+)_s(\d+)_nod3\.json$")
num = lambda s: float(s.replace("p", "."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nod3", action="store_true", help="collect the 3873-dim (no D3) rerun")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"{R}/lambda_table_nod3.json" if args.nod3 else f"{R}/lambda_table.json"
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

        row["cpi_mape"] = {k: (round(v, 1) if k != "per_seed" else [round(x, 1) for x in v])
                           for k, v in row["cpi_mape"].items()}
        rows.append(row)

    meta = {"input": ("first 3873 (Concorde) features, 808 D3 features removed" if args.nod3
                      else "4681 features = 3873 Concorde + 808 D3"),
            "setting": "exp1, 18k dataset, row split seed 42, 100% labels, [512,256], 1521 epochs",
            "selection": "checkpoint with best validation CPI MAPE (1/16 of train rows held out); test evaluated once", "mono_form": "hinge (relu)",
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
