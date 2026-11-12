#!/usr/bin/env python3
"""
How much the MSCS error affects bottleneck identification, final configuration (V2, no-D3, lam (10, 1, 0), 3 seeds),
test split, from the saved checkpoints, on the CPU. Per stage, over the three lost-slot components FP/BP/SP:
  dominant_acc       predicted largest component == true largest (checked against the run JSON)
  acc_by_margin      dominant_acc among samples whose true top-1 minus top-2 share is in each MARGINS bin
  regret_wrong       on misidentified samples: true share of the true dominant minus true share of the predicted one
  regret_all         the same averaged over all samples (0 when correct)
  rank_exact         full FP/BP/SP order correct
  bench_dominant     per benchmark: dominant of the mean predicted MSCS == dominant of the mean true MSCS (23 benchmarks)
  bench_rank_exact   per benchmark: full order of the mean MSCS correct
Refuses to overwrite the output.

Usage: CUDA_VISIBLE_DEVICES= python eval_bottleneck_id_1105.py
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONCORDE_INPUT_DIM, DROPOUT, BENCHMARK_NAMES, SEED, FP_INDICES, BP_INDICES, SP_INDICES
from model import GEMModel
from dataset import load_all_benchmarks, create_exp1_dataloaders
from train_exp1_lambda_sens import evaluate
from train_exp1_cpi_ablation import DATA_GEM, DATA_CON, HIDDEN

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "results", "lambda_sweep_nod3")
OUT = os.path.join(HERE, "results", "bottleneck_id_nod3_1105.json")
SEEDS = [42, 123, 456]
STAGES = ["Decode", "Dispatch", "Issue", "Commit"]
MARGINS = [0.0, 0.02, 0.05, 0.10, 1.01]                       # bins of the true top-1 minus top-2 share


def stage_metrics(p, t, bm_idx):
    """p, t: (N, 3) predicted / true FP, BP, SP shares of one stage."""
    pa, ta = p.argmax(1), t.argmax(1)
    ok = pa == ta
    ts = t.sort(1, descending=True).values
    margin = ts[:, 0] - ts[:, 1]
    regret = t.gather(1, ta[:, None])[:, 0] - t.gather(1, pa[:, None])[:, 0]
    bins = []
    for lo, hi in zip(MARGINS[:-1], MARGINS[1:]):
        m = (margin >= lo) & (margin < hi)
        bins.append({"margin": [lo, hi], "share": m.float().mean().item() * 100,
                     "acc": ok[m].float().mean().item() * 100 if m.any() else None})
    bench_dom, bench_rank = [], []
    for s, e in bm_idx.values():
        pm, tm = p[s:e].mean(0), t[s:e].mean(0)
        bench_dom.append(bool(pm.argmax() == tm.argmax()))
        bench_rank.append(bool((pm.argsort() == tm.argsort()).all()))
    return {"dominant_acc": ok.float().mean().item() * 100,
            "acc_by_margin": bins,
            "regret_wrong": regret[~ok].mean().item() if (~ok).any() else 0.0,
            "regret_wrong_p90": regret[~ok].quantile(0.9).item() if (~ok).any() else 0.0,
            "regret_all": regret.mean().item(),
            "rank_exact": (p.argsort(1) == t.argsort(1)).all(1).float().mean().item() * 100,
            "bench_dominant": f"{sum(bench_dom)}/{len(bench_dom)}",
            "bench_rank_exact": f"{sum(bench_rank)}/{len(bench_rank)}"}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists, refusing to overwrite")
    data = load_all_benchmarks(DATA_GEM, DATA_CON, BENCHMARK_NAMES)
    _, vl, _, _, bm_idx = create_exp1_dataloaders(data, seed=SEED)
    del data
    ds = vl.dataset.__dict__
    X, M, Y, W = ds["gem_features"][:, :CONCORDE_INPUT_DIM].contiguous(), ds["mscs_target"], ds["cpi_golden"], ds["w_min"]

    per_seed = []
    for s in SEEDS:
        tag = f"t10_c1_m0_s{s}_nod3"
        model = GEMModel(CONCORDE_INPUT_DIM, HIDDEN, DROPOUT)
        model.load_state_dict(torch.load(os.path.join(RUNS, f"{tag}.pt"), map_location="cpu"))
        ev, _ = evaluate(model, X, W, Y, M, full=True)
        ref = json.load(open(os.path.join(RUNS, f"{tag}.json")))["selected"]
        with torch.no_grad():
            P = model(X)
        stages = {}
        for st, f, b, sp in zip(STAGES, FP_INDICES, BP_INDICES, SP_INDICES):
            stages[st] = stage_metrics(P[:, [f, b, sp]], M[:, [f, b, sp]], bm_idx)
        check = max(abs(stages[st]["dominant_acc"] - ref["dominant_acc"][i]) for i, st in enumerate(STAGES))
        print(f"  seed {s}: CPI MAPE {ev['mape_mean4']:.4f} (JSON {ref['mape_mean4']:.4f}), dominant_acc max diff {check:.2e}")
        per_seed.append({"seed": s, "stages": stages})

    meta = {"setting": "V2, no-D3 (3873 features), lam (10, 1, 0), [512,256], test split 14,499 rows, 23 benchmarks",
            "components": "FP, BP, SP of each stage (Base excluded)", "runs": RUNS,
            "regret": "true share of the true dominant minus true share of the predicted dominant (fraction of slots)"}
    json.dump({"meta": meta, "per_seed": per_seed}, open(OUT, "w"), indent=2)
    print(f"Saved {OUT}")
    for st in STAGES:
        r = [p["stages"][st] for p in per_seed]
        mean = lambda k: np.mean([x[k] for x in r])
        bins = " ".join(f"[{b['margin'][0]:.2f},{b['margin'][1]:.2f}) {np.mean([x['acc_by_margin'][i]['share'] for x in r]):.1f}%:"
                        f"{np.mean([x['acc_by_margin'][i]['acc'] for x in r]):.1f}"
                        for i, b in enumerate(r[0]["acc_by_margin"]))
        print(f"  {st:<9} dom {mean('dominant_acc'):.2f}  rank {mean('rank_exact'):.1f}  regret_wrong {mean('regret_wrong'):.4f}"
              f" (p90 {mean('regret_wrong_p90'):.4f})  regret_all {mean('regret_all'):.5f}"
              f"  bench dom {[x['bench_dominant'] for x in r]} rank {[x['bench_rank_exact'] for x in r]}")
        print(f"            share:acc by margin  {bins}")


if __name__ == "__main__":
    main()
