#!/usr/bin/env python3

import os, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SWEEP = os.path.join(RES, "lambda_sensitivity_nod3_1031.json")
OLD_256 = os.path.join(RES, "exp1_v3_76", "exp1_256_128_183k")
NEW_256 = os.path.join(RES, "loss_ablation_256_nod3_1101")
OUT = os.path.join(RES, "loss_ablation_nod3_1101.json")
SEEDS = [42, 123, 456]
VARIANTS = [("V2 (full)", (1, 1, 0.01), "gem_full", "t1_c1_m0p01"),
            ("V2 (L_const)", (1, 1, 0), "gem_const", "t1_c1_m0"),
            ("V2 (L_mono)", (1, 0, 0.01), "gem_mono", "t1_c0_m0p01")]


def stats(v):
    return {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": [float(x) for x in v]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s256", choices=["old", "new"], default="old")
    args = ap.parse_args()

    sweep = {(r["lam_task"], r["lam_const"], r["lam_mono"]): r for r in json.load(open(SWEEP))["rows"]}
    rows = []
    for label, cfg, old, tag in VARIANTS:
        if args.s256 == "old":
            f = os.path.join(OLD_256, f"{old}.json")
            s256, files = stats([json.load(open(f))["overall_mape"]]), [f]
        else:
            files = [os.path.join(NEW_256, f"{tag}_s{s}_nod3.json") for s in SEEDS]
            missing = [f for f in files if not os.path.exists(f)]
            if missing:
                raise SystemExit(f"missing runs: {missing}")
            s256 = stats([json.load(open(f))["selected"]["mape_mean4"] for f in files])
        rows.append({"label": label, "lam_task": cfg[0], "lam_const": cfg[1], "lam_mono": cfg[2],
                     "s512": sweep[cfg]["cpi_mape"], "s256": s256,
                     "s256_files": [os.path.relpath(f, HERE) for f in files]})

    meta = {"metric": "CPI MAPE (%), test set, mean of the 4 per-stage CPIs",
            "s512": "V2, 100% labels, 3873 features, 3 seeds, best test MAPE checkpoint, lam values as in each row",
            "s256_source": args.s256,
            "s256": ("old Fig. 5 values: 4681 features, 1 run, lam_task 0.1 with L_const and 1 without; not the "
                     "lam values in each row" if args.s256 == "old" else
                     "same setting as s512, [256,128]"),
            "std": "population std over seeds"}
    json.dump({"meta": meta, "rows": rows}, open(OUT, "w"), indent=2)
    print(f"Saved {OUT} (s256 = {args.s256})")
    for r in rows:
        print(f"  {r['label']:<14} [256,128]={r['s256']['mean']:.3f}  [512,256]={r['s512']['mean']:.3f}")


if __name__ == "__main__":
    main()
