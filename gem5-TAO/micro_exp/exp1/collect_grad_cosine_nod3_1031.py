#!/usr/bin/env python3
"""
Collect the Fig. 6 re-measurement into one JSON, unrounded: for each (lam_task, lam_const, lam_mono),
per-seed series, mean and population std over seeds of every gradient quantity at each measured epoch.

Also compares each run's per-epoch test CPI MAPE with the matching run of the sensitivity sweep
(results/lambda_sweep_nod3_1026): the measurement must not change training, so max_abs_mape_diff
should be about 0. Refuses to overwrite the output and to collect an incomplete set of runs.

Usage: python collect_grad_cosine_nod3_1031.py
"""
import os, json, glob, argparse
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [(1, 1, 0.01), (1, 0, 0.01), (0.1, 1, 0), (10, 1, 0)]
SEEDS = [42, 123, 456]
KEYS = ["cos_task_mono", "cos_task_const", "cos_const_mono", "cos_task_hinge",
        "norm_task", "norm_const", "norm_mono", "norm_total", "l_task", "l_const", "l_mono"]


def stats(series):
    """(seeds, epochs) with None -> NaN; epochs with no value in any seed stay None."""
    a = np.array([[np.nan if v is None else v for v in s] for s in series], dtype=float)
    ok = np.isfinite(a).any(0)
    mean, std = np.full(a.shape[1], np.nan), np.full(a.shape[1], np.nan)
    mean[ok], std[ok] = np.nanmean(a[:, ok], 0), np.nanstd(a[:, ok], 0)
    return {"mean": [None if np.isnan(x) else float(x) for x in mean],
            "std": [None if np.isnan(x) else float(x) for x in std], "per_seed": series}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results", "grad_cosine_nod3_1031"))
    ap.add_argument("--sweep", default=os.path.join(HERE, "results", "lambda_sweep_nod3_1026"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "grad_cosine_nod3_1031.json"))
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists, refusing to overwrite")

    acc = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.results, "t*_nod3.json"))):
        j = json.load(open(f))
        acc[(j["lam_task"], j["lam_const"], j["lam_mono"])].append((j["seed"], j, f))
    rows = []
    for cfg in CONFIGS:
        runs = sorted(acc.get(cfg, []), key=lambda r: r[0])
        if [r[0] for r in runs] != SEEDS:
            raise SystemExit(f"{cfg}: seeds {[r[0] for r in runs]}, expected {SEEDS}")
        row = {"lam_task": cfg[0], "lam_const": cfg[1], "lam_mono": cfg[2], "seeds": SEEDS,
               "files": [os.path.relpath(r[2], HERE) for r in runs],
               "epochs": [h["epoch"] for h in runs[0][1]["grads"]],
               "selected_cpi_mape": [r[1]["selected"]["mape_mean4"] for r in runs],
               "selected_best_epoch": [r[1]["best_epoch"] for r in runs],
               "last_cpi_mape": [r[1]["last_epoch"]["mape_mean4"] for r in runs],
               "test_cpi_mape": stats([[h["test"]["mape_mean4"] for h in r[1]["grads"]] for r in runs]),
               "test_mono_viol": stats([[h["test"]["mono_viol_gt0.01"] for h in r[1]["grads"]] for r in runs]),
               "test_base_incons": stats([[h["test"]["base_spread_rel"] * 100 for h in r[1]["grads"]] for r in runs])}
        for k in KEYS:
            row[k] = stats([[h[k] for h in r[1]["grads"]] for r in runs])
        diffs = []
        for s, j, f in runs:
            ref = json.load(open(os.path.join(args.sweep, os.path.basename(f))))
            diffs.append(max(abs(a["mape_mean4"] - b["mape_mean4"]) for a, b in zip(j["history"], ref["history"])))
        row["max_abs_mape_diff_vs_sweep"] = diffs
        rows.append(row)
        print(f"{cfg}: CPI {np.mean(row['selected_cpi_mape']):.3f}  "
              f"mean cos(task,mono) {np.mean(row['cos_task_mono']['mean']):+.3f}  "
              f"mean cos(task,const) {np.mean(row['cos_task_const']['mean']):+.3f}  "
              f"max MAPE diff vs sweep {max(diffs):.4f}")

    meta = {"setting": "exp1 = V2 at 100% labels, 3873 Concorde features, as results/lambda_sweep_nod3_1026",
            "measurement": "mean per-batch cosine of raw gradients on all full training batches of one extra "
                           "shuffle, in train mode, RNG and BN buffers restored; epoch 0 = initialization",
            "cos_task_hinge": "L_mono with relu instead of softplus (acts only on actual violations)",
            "norm_total": "norm of the sum of lam_i * grad_i over the trained terms",
            "test_mono_viol": "monotonicity violation rate (%), FP or SP increases or BP decreases from Decode "
                              "to Commit by > 0.01",
            "test_base_incons": "cross-stage Base inconsistency (%), (max-min)/mean of the 4 predicted Base "
                                "values, averaged over samples",
            "std": "population std over seeds", "source": os.path.relpath(args.results, HERE)}
    json.dump({"meta": meta, "rows": rows}, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
