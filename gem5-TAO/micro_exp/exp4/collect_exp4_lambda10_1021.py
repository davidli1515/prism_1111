#!/usr/bin/env python3
"""
Collect exp4 label efficiency at lam_task=10, lam_const=1, lam_mono=0 into one JSON.

New runs (3 seeds 42/123/456):
  V2, V3 : results_lambda10_main_1012/{v2,v3}_r{pct}_t10_c1_m0_u1_s{seed}.json   (5-90%)
  PRISM  : results_lambda10_prism_1012/prism_r{pct}_t10_c1_m0_u1_s{seed}.json     (5-100%)
Baselines (results_77, seed 42 only, CPI MAPE only):
  Concorde, V1, and the paper's lam_task=0.1 V2 / V3 / PRISM.

Per (variant, ratio) the JSON stores per-seed values, mean and std of
  cpi_mape       CPI MAPE (%), 4-stage mean CPI, selected checkpoint (best test MAPE)
  component_mae  mean MAE of the FP/BP/SP proportions over all stages
  base_incons    cross-stage Base inconsistency (%): (max - min)/mean of the 4 predicted
                 Base values, averaged over samples (0 = all stages imply the same CPI)
  mono_viol      monotonicity violation rate (%): predictions where FP or SP increases, or
                 BP decreases, from Decode to Commit by more than 0.01
plus the source files.

--nod3: the no-D3 rerun (input = first 3873 Concorde features, as the Concorde baseline):
  V1 / V2 / V3 : results_lambda10_nod3_main_1025 (V1 = L_task only, lam_task 1)
  PRISM        : results_lambda10_nod3_prism_1025
  plus the 4681-dim lam_task=10 runs as V2_gem / V3_gem / PRISM_gem for comparison.

Usage: python collect_exp4_lambda10_1021.py [--out results_lambda10_summary_1021.json]
       python collect_exp4_lambda10_1021.py --nod3 [--out results_lambda10_nod3_summary_1025.json]
"""
import os, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = {"V2": ("results_lambda10_main_1012", "v2"),
       "V3": ("results_lambda10_main_1012", "v3"),
       "PRISM": ("results_lambda10_prism_1012", "prism")}
OLD = {"Concorde": "results_77/exp4_18k_concorde",
       "V1": "results_77/exp4_18k_vanilla",
       "V2_lam0.1": "results_77/exp4_18k_supervised_only",
       "V3_lam0.1": "results_77/exp4_18k",
       "PRISM_lam0.1": "results_77/exp4_18k_extra_unlabeled"}
SEEDS = [42, 123, 456]
PCTS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
METRICS = {
    "cpi_mape": lambda s: s["mape_mean4"],
    "component_mae": lambda s: float(np.mean([s["mae"][k] for k in ("FP", "BP", "SP")])),
    "base_incons": lambda s: s["base_spread_rel"] * 100,
    "mono_viol": lambda s: s["mono_viol_gt0.01"],
}


def stats(v):
    return {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": [float(x) for x in v]}


NEW_NOD3 = {"V1": ("results_lambda10_nod3_main_1025", "v1", "t1_c0_m0_u1", "_nod3"),
            "V2": ("results_lambda10_nod3_main_1025", "v2", "t10_c1_m0_u1", "_nod3"),
            "V3": ("results_lambda10_nod3_main_1025", "v3", "t10_c1_m0_u1", "_nod3"),
            "PRISM": ("results_lambda10_nod3_prism_1025", "prism", "t10_c1_m0_u1", "_nod3"),
            "V2_gem": ("results_lambda10_main_1012", "v2", "t10_c1_m0_u1", ""),
            "V3_gem": ("results_lambda10_main_1012", "v3", "t10_c1_m0_u1", ""),
            "PRISM_gem": ("results_lambda10_prism_1012", "prism", "t10_c1_m0_u1", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nod3", action="store_true", help="collect the 3873-dim (no D3) rerun")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(HERE, "results_lambda10_nod3_summary_1025.json" if args.nod3
                                else "results_lambda10_summary_1021.json")
    new = NEW_NOD3 if args.nod3 else {k: (d, pre, "t10_c1_m0_u1", "") for k, (d, pre) in NEW.items()}
    old = {k: v for k, v in OLD.items() if not (args.nod3 and k == "V1")}
    if args.nod3:
        old["V1_gem"] = OLD["V1"]

    variants = {}
    for name, (d, pre, w, suf) in new.items():
        rows = {}
        for p in PCTS:
            files = [os.path.join(HERE, d, f"{pre}_r{p}_{w}_s{s}{suf}.json") for s in SEEDS]
            files = [f for f in files if os.path.exists(f)]
            if not files:
                continue
            sel = [json.load(open(f))["selected"] for f in files]
            rows[str(p)] = {"n_seeds": len(sel), "seeds": [int(f.rsplit("_s", 1)[1].split("_")[0].split(".")[0]) for f in files],
                            **{k: stats([fn(s) for s in sel]) for k, fn in METRICS.items()},
                            "files": files}
        lt, lc = (1, 0) if pre == "v1" else (10, 1)
        variants[name] = {"lam_task": lt, "lam_const": lc, "lam_mono": 0, "source": d,
                          "input_dim": 3873 if suf else 4681, "ratios": rows}
    for name, d in old.items():
        rows = {}
        for p in PCTS:
            f = os.path.join(HERE, d, f"exp4_{p}pct.json")
            if os.path.exists(f):
                rows[str(p)] = {"n_seeds": 1, "seeds": [42],
                                "cpi_mape": stats([json.load(open(f))["overall_mape"]]), "files": [f]}
        variants[name] = {"source": d, "note": "baseline from results_77, seed 42, CPI MAPE only",
                          "input_dim": 3873 if name == "Concorde" else 4681,
                          "ratios": rows}

    meta = {"input": ("first 3873 (Concorde) features for V1/V2/V3/PRISM; *_gem = 4681 (+808 D3)"
                      if args.nod3 else "4681 = 3873 Concorde + 808 D3 features (Concorde baseline: 3873)"),
            "setting": "exp4, 18k dataset, row split seed 42, test = 18k test set, [512,256], 1521 epochs",
            "selection": "checkpoint with best test CPI MAPE (as in the original exp4 scripts)",
            "prism_data": "V3 + 267,213 extra unlabeled rows from the 40k set "
                          "(create_dataloaders_with_extra, as train_exp4_extra_unlabeled.py)",
            "metrics": {"cpi_mape": "CPI MAPE (%), mean of the 4 per-stage CPIs",
                        "component_mae": "mean MAE of FP/BP/SP proportions over all stages",
                        "base_incons": "cross-stage Base inconsistency (%), (max-min)/mean of the 4 Base values",
                        "mono_viol": "monotonicity violation rate (%), FP/SP increase or BP decrease "
                                     "from Decode to Commit by > 0.01"}}
    json.dump({"meta": meta, "variants": variants}, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")
    print(f"{'ratio':>5} | " + " | ".join(f"{v:>13}" for v in variants))
    for p in PCTS:
        cells = []
        for v in variants.values():
            r = v["ratios"].get(str(p))
            cells.append(f"{r['cpi_mape']['mean']:6.2f}±{r['cpi_mape']['std']:.2f}" if r else f"{'-':>11}")
        print(f"{p:>4}% | " + " | ".join(f"{c:>13}" for c in cells))


if __name__ == "__main__":
    main()
