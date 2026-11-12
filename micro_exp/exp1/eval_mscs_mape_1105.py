#!/usr/bin/env python3
"""
Per-dimension MSCS error of the final configuration (V2, no-D3, lam = (10, 1, 0), 3 seeds) on the test split,
from the saved checkpoints (best test CPI MAPE), on the CPU. For each of the 16 MSCS dimensions:
  mae      mean |pred - true| (fraction of slots)
  mape     mean |pred - true| / true over samples with true > MASK (near-zero targets make MAPE undefined)
  coverage share of test samples with true > MASK (%)
  wape     sum |pred - true| / sum true (%), all samples
The recomputed CPI MAPE and component MAE are checked against each run's JSON. Refuses to overwrite the output.

Usage: CUDA_VISIBLE_DEVICES= python eval_mscs_mape_1105.py [--mono 0p01]   (lam_mono as in the run tags)
"""
import os, sys, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONCORDE_INPUT_DIM, DROPOUT, BENCHMARK_NAMES, SEED
from model import GEMModel
from dataset import load_all_benchmarks, create_exp1_dataloaders
from train_exp1_lambda_sens import evaluate, COMP
from train_exp1_cpi_ablation import DATA_GEM, DATA_CON, HIDDEN

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "results", "lambda_sweep_nod3")
SEEDS = [42, 123, 456]
MASK = 0.05
STAGES = ["Decode", "Dispatch", "Issue", "Commit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mono", default="0", help="lam_mono as written in the run tag, e.g. 0, 0p01, 0p1, 1")
    mono = ap.parse_args().mono
    OUT = os.path.join(HERE, "results", "mscs_mape_nod3_1105.json" if mono == "0" else f"mscs_mape_nod3_m{mono}_1105.json")
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists, refusing to overwrite")
    data = load_all_benchmarks(DATA_GEM, DATA_CON, BENCHMARK_NAMES)
    _, vl, _, _, _ = create_exp1_dataloaders(data, seed=SEED)
    del data
    ds = vl.dataset.__dict__
    X, M, Y, W = ds["gem_features"][:, :CONCORDE_INPUT_DIM].contiguous(), ds["mscs_target"], ds["cpi_golden"], ds["w_min"]

    per_seed = []
    for s in SEEDS:
        tag = f"t10_c1_m{mono}_s{s}_nod3"
        model = GEMModel(CONCORDE_INPUT_DIM, HIDDEN, DROPOUT)
        model.load_state_dict(torch.load(os.path.join(RUNS, f"{tag}.pt"), map_location="cpu"))
        ev, _ = evaluate(model, X, W, Y, M, full=True)
        ref = json.load(open(os.path.join(RUNS, f"{tag}.json")))["selected"]
        check = max([abs(ev["mape_mean4"] - ref["mape_mean4"])] + [abs(ev["mae"][k] - ref["mae"][k]) for k in COMP])
        print(f"  seed {s}: CPI MAPE {ev['mape_mean4']:.4f} (JSON {ref['mape_mean4']:.4f}), max diff {check:.2e}")
        with torch.no_grad():
            P = model(X)
        err = (P - M).abs()
        dims = {}
        for comp, idx in COMP.items():
            for st, d in zip(STAGES, idx):
                t, e = M[:, d], err[:, d]
                keep = t > MASK
                dims[f"{st}/{comp}"] = {"mae": e.mean().item(),
                                        "mape": (e[keep] / t[keep]).mean().item() * 100 if keep.any() else None,
                                        "coverage": keep.float().mean().item() * 100,
                                        "wape": (e.sum() / t.sum()).item() * 100}
        per_seed.append({"seed": s, "check_max_abs_diff": check, "dims": dims})

    summary = {}
    for name in per_seed[0]["dims"]:
        summary[name] = {}
        for k in ["mae", "mape", "coverage", "wape"]:
            v = [p["dims"][name][k] for p in per_seed if p["dims"][name][k] is not None]
            summary[name][k] = {"mean": float(np.mean(v)), "std": float(np.std(v))} if v else None
    meta = {"setting": f"V2, no-D3 (3873 features), lam (10, 1, {mono.replace('p', '.')}), [512,256], test split 14,499 rows, "
                       "checkpoint with best test CPI MAPE", "runs": RUNS, "mask": f"MAPE over samples with true > {MASK}",
            "std": "population std over seeds"}
    json.dump({"meta": meta, "summary": summary, "per_seed": per_seed}, open(OUT, "w"), indent=2)
    print(f"Saved {OUT}")
    print(f"  {'dim':<16}{'MAE':>8}{'MAPE%':>8}{'cov%':>8}{'WAPE%':>8}")
    for name, r in summary.items():
        mape = f"{r['mape']['mean']:.1f}" if r["mape"] else "-"
        print(f"  {name:<16}{r['mae']['mean']:>8.4f}{mape:>8}{r['coverage']['mean']:>8.1f}{r['wape']['mean']:>8.1f}")


if __name__ == "__main__":
    main()
