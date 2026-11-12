#!/usr/bin/env python3
"""
Collect the values of the original Fig. 2 (per-benchmark CPI error, average + P90) into one JSON, unrounded.
Same sources as plot 2 of plot_error_breakdown.py:
  PRISM: average and P90 from its JSON.
  Concorde, V1: average from their JSONs, P90 from inference with their checkpoints on the test split
  (that script did not save these). Inference runs on the CPU; the recomputed average is checked against the JSON.
Refuses to overwrite the output.

Usage: CUDA_VISIBLE_DEVICES= python collect_fig2_per_benchmark_1102.py
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")
from config import GEM_INPUT_DIM, CONCORDE_INPUT_DIM, DROPOUT, SEED, BENCHMARK_NAMES
from model import GEMModel, ConcordeModel, compute_cpi_mean
from dataset import load_all_benchmarks, create_exp1_dataloaders

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "fig2_per_benchmark_1102.json")
PRISM_JSON = "/scratch/gem5-TAO/micro_exp/exp4/results_77_p90/exp4_18k_extra_unlabeled/exp4_100pct.json"
RUN_DIR = "/scratch/gem5-TAO/micro_exp/exp1/results/exp1_v2_74/exp1_512_256_183k"
HIDDEN = [512, 256]
GEM_DIR, CON_DIR = "/scratch2/wl730/micro_traindata_packed/gem", "/scratch2/wl730/micro_traindata_packed/concord"


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists, refusing to overwrite")
    prism = json.load(open(PRISM_JSON))
    variants = {"PRISM": {"source": PRISM_JSON, "overall_mape": prism["overall_mape"], "overall_p90": prism["p90"],
                          "avg": prism["per_benchmark"], "p90": prism["p90_per_benchmark"]}}

    torch.manual_seed(SEED); np.random.seed(SEED)
    _, test_loader, _, _, bm_idx = create_exp1_dataloaders(load_all_benchmarks(GEM_DIR, CON_DIR, BENCHMARK_NAMES))
    concorde = ConcordeModel(CONCORDE_INPUT_DIM, HIDDEN, 0.0)
    concorde.load_state_dict(torch.load(f"{RUN_DIR}/concorde.pt", map_location="cpu")); concorde.eval()
    vanilla = GEMModel(GEM_INPUT_DIM, HIDDEN, DROPOUT)
    vanilla.load_state_dict(torch.load(f"{RUN_DIR}/gem_vanilla.pt", map_location="cpu")); vanilla.eval()
    con_pred, van_pred, gold = [], [], []
    with torch.no_grad():
        for b in test_loader:
            con_pred.append(concorde(b["concord_features"]))
            van_pred.append(compute_cpi_mean(vanilla(b["gem_features"]), b["w_min"]))
            gold.append(b["cpi_golden"])
    gold = torch.cat(gold)

    for key, name, pred in [("Concorde", "concorde", con_pred), ("V1", "gem_vanilla", van_pred)]:
        d = json.load(open(f"{RUN_DIR}/{name}.json"))
        err = torch.abs(torch.cat(pred).reshape(gold.shape) - gold) / (gold + 1e-8) * 100
        p90 = {bm: err[s:e].quantile(0.9).item() for bm, (s, e) in bm_idx.items()}
        check = max(abs(err[s:e].mean().item() - d["per_benchmark"][bm]) for bm, (s, e) in bm_idx.items())
        print(f"  {key}: max |recomputed avg - JSON avg| over benchmarks = {check:.4f}")
        variants[key] = {"source": f"{RUN_DIR}/{name}.json (+ .pt for P90)", "overall_mape": d["overall_mape"],
                         "avg": d["per_benchmark"], "p90": p90, "avg_check_max_abs_diff": check}

    meta = {"metric": "CPI error (%), test set; avg = MAPE, p90 = 90th percentile of per-sample error",
            "source": "plot 2 of plot_error_breakdown.py ([512,256], 183k)"}
    json.dump({"meta": meta, "variants": variants}, open(OUT, "w"), indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
