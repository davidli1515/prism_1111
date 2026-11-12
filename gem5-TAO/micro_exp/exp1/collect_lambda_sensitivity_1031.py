#!/usr/bin/env python3
import os, json, glob
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "results", "lambda_sweep_nod3_1026")
OUT = os.path.join(HERE, "results", "lambda_sensitivity_nod3_1031.json")
METRICS = {"cpi_mape": lambda s: s["mape_mean4"],
           "component_mae": lambda s: float(np.mean([s["mae"][k] for k in ("FP", "BP", "SP")])),
           "base_incons": lambda s: s["base_spread_rel"] * 100,
           "mono_viol": lambda s: s["mono_viol_gt0.01"]}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists, refusing to overwrite")
    acc = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(SWEEP, "t*_nod3.json"))):
        j = json.load(open(f))
        acc[(j["lam_task"], j["lam_const"], j["lam_mono"])].append((j["seed"], j["selected"], f))
    rows = []
    for (t, c, m), runs in sorted(acc.items()):
        runs = sorted(runs, key=lambda r: r[0])
        row = {"lam_task": t, "lam_const": c, "lam_mono": m, "seeds": [r[0] for r in runs],
               "files": [os.path.relpath(r[2], HERE) for r in runs]}
        for k, fn in METRICS.items():
            v = [float(fn(r[1])) for r in runs]
            row[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v}
        rows.append(row)
    meta = {"setting": "exp1 = V2 at 100% labels: 18k row split seed 42, 217,476 labeled rows, no unlabeled "
                       "rows, 14,499 test rows, [512,256], dropout 0.1, 1521 epochs",
            "input": "first 3873 (Concorde) features, 808 D3 features removed",
            "selection": "checkpoint with best test CPI MAPE",
            "chosen": {"lam_task": 10, "lam_const": 1, "lam_mono": 0},
            "metrics": {"cpi_mape": "CPI MAPE (%), mean of the 4 per-stage CPIs",
                        "component_mae": "mean MAE of the FP/BP/SP proportions over all stages",
                        "base_incons": "cross-stage Base inconsistency (%), (max-min)/mean of the 4 predicted "
                                       "Base values, averaged over samples",
                        "mono_viol": "monotonicity violation rate (%), FP or SP increases or BP decreases from "
                                     "Decode to Commit by > 0.01"},
            "std": "population std (np.std) over seeds",
            "source": os.path.relpath(SWEEP, HERE)}
    json.dump({"meta": meta, "rows": rows}, open(OUT, "w"), indent=2)
    print(f"Saved {OUT}: {len(rows)} configurations, seeds per config = {sorted({len(r['seeds']) for r in rows})}")


if __name__ == "__main__":
    main()
