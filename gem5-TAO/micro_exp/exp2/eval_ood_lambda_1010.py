#!/usr/bin/env python3
"""
Four metrics on the target test set for the cross-program OOD lambda sweep
(results_ood_lambda_1010/{t*_m*}/src20/prism/<target>/n{N}_s42.pt):
  CPI MAPE (4-stage mean), MSCS component MAE (FP/BP/SP), cross-stage Base
  inconsistency ((max-min)/mean of the 4 Base values, %), monotonicity violation
  rate (FP/SP decreasing, BP increasing across stages, violation > 0.01, %).
Target test rows are rebuilt with the same window split as train_ood_lowlabel.py
and normalized with each run's saved normalizer. Inference only (CPU).
Usage: python eval_ood_lambda_1010.py
"""
import os, sys, re, glob, json, pickle
import numpy as np, torch

MICRO = "/scratch/gem5-TAO/micro_exp"
sys.path.insert(0, f"{MICRO}/exp1"); sys.path.insert(0, f"{MICRO}/window_ids")
from config import GEM_INPUT_DIM, DROPOUT, TRAIN_RATIO, D3_FEAT_END, BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES
from model import GEMModel, compute_cpi_mean
from window_split import load_window_ids, split_by_window
sys.path.insert(0, f"{MICRO}/exp2")
from train_ood_lowlabel import open_bench, take_rows, stable_hash, DATA_18K

ROOT = f"{MICRO}/exp2/results_ood_lambda_1010"
OUT = f"{ROOT}/four_metrics.json"


def target_test(name, seed=42):
    raw = open_bench(DATA_18K, name)
    rng = np.random.RandomState(seed + stable_hash(name))
    _, te = split_by_window(load_window_ids(name, raw.shape[0]), TRAIN_RATIO, rng)
    return take_rows(raw, te, D3_FEAT_END)


def four(P, T, W, Y):
    cpi = compute_cpi_mean(torch.from_numpy(P), torch.from_numpy(W)).numpy()
    base = P[:, BASE_INDICES]
    v = np.concatenate([np.diff(P[:, FP_INDICES], 1).clip(0), np.diff(P[:, SP_INDICES], 1).clip(0),
                        (-np.diff(P[:, BP_INDICES], 1)).clip(0)], 1).max(1)
    idx = np.concatenate([FP_INDICES, BP_INDICES, SP_INDICES])
    return dict(mape=float((np.abs(cpi - Y) / Y).mean() * 100),
                comp_mae=float(np.abs(P[:, idx] - T[:, idx]).mean()),
                base_incons=float(((base.max(1) - base.min(1)) / base.mean(1)).mean() * 100),
                viol=float((v > 0.01).mean() * 100))


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists; refusing to overwrite")
    tests, res = {}, {}
    for ck in sorted(glob.glob(f"{ROOT}/t*_m*/src20/prism/*/n*_s42.pt")):
        combo = ck.split("/")[-5]; tgt = ck.split("/")[-2]; n = re.search(r"/n(\d+)_s42\.pt$", ck).group(1)
        if tgt not in tests:
            tests[tgt] = target_test(tgt)
        d = tests[tgt]
        nrm = pickle.load(open(ck.replace(".pt", "_normalizer.pkl"), "rb"))
        x = torch.from_numpy(((d["x"] - nrm["mean"]) / nrm["std"]).astype(np.float32))
        m = GEMModel(GEM_INPUT_DIM, [512, 256], DROPOUT)
        m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
        with torch.no_grad():
            P = m(x).numpy()
        r = four(P, d["mscs"], d["wmin"], d["cpi"])
        saved = json.load(open(ck.replace(".pt", ".json")))["target_mape"]
        assert abs(r["mape"] - saved) < 0.05, f"{ck}: recomputed {r['mape']:.2f} vs saved {saved:.2f}"
        res[f"{combo}|{tgt}|{n}"] = r
    json.dump(res, open(OUT, "w"), indent=2)
    combos = sorted({k.split("|")[0] for k in res}, key=lambda c: (float(c.split("_")[0][1:].replace("p", ".")), c))
    print(f"{'combo':10s} {'N':>5s} | {'n':>2s} | CPI MAPE | comp MAE | Base incons % | viol %")
    for n in ["256", "1024"]:
        for c in combos:
            rs = [v for k, v in res.items() if k.startswith(c + "|") and k.endswith("|" + n)]
            f = lambda key: np.mean([r[key] for r in rs])
            print(f"{c:10s} {n:>5s} | {len(rs):2d} | {f('mape'):8.2f} | {f('comp_mae'):8.3f} | {f('base_incons'):13.2f} | {f('viol'):6.1f}")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
