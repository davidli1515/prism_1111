#!/usr/bin/env python3
"""Four target metrics for the SUPERVISED (V2) cross-program OOD lambda sweep
(results_ood_lambda_v2); same procedure as eval_ood_lambda_1010.py. Prints V2 next to V3."""
import os, sys, re, glob, json, pickle
import numpy as np, torch
sys.path.insert(0, "/path/to/gem5/micro_exp/exp2")
from eval_ood_lambda_1010 import target_test, four, GEMModel, GEM_INPUT_DIM, DROPOUT

ROOT = "/path/to/gem5/micro_exp/exp2/results_ood_lambda_v2"
OUT = f"{ROOT}/four_metrics.json"
V3 = "/path/to/gem5/micro_exp/exp2/results_ood_lambda/four_metrics.json"

if os.path.exists(OUT):
    sys.exit(f"{OUT} exists; refusing to overwrite")
tests, res = {}, {}
for ck in sorted(glob.glob(f"{ROOT}/t*_m*/src20/supervised/*/n*_s42.pt")):
    combo, tgt = ck.split("/")[-5], ck.split("/")[-2]
    n = re.search(r"/n(\d+)_s42\.pt$", ck).group(1)
    if tgt not in tests:
        tests[tgt] = target_test(tgt)
    d = tests[tgt]
    nrm = pickle.load(open(ck.replace(".pt", "_normalizer.pkl"), "rb"))
    x = torch.from_numpy(((d["x"] - nrm["mean"]) / nrm["std"]).astype(np.float32))
    m = GEMModel(GEM_INPUT_DIM, [512, 256], DROPOUT); m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
    with torch.no_grad():
        P = m(x).numpy()
    r = four(P, d["mscs"], d["wmin"], d["cpi"])
    saved = json.load(open(ck.replace(".pt", ".json")))["target_mape"]
    assert abs(r["mape"] - saved) < 0.05, f"{ck}: recomputed {r['mape']:.2f} vs saved {saved:.2f}"
    res[f"{combo}|{tgt}|{n}"] = r
json.dump(res, open(OUT, "w"), indent=2)
v3 = json.load(open(V3))
avg = lambda R, c, n, k: np.mean([v[k] for kk, v in R.items() if kk.startswith(c + "|") and kk.endswith("|" + n)])
print(f"{'combo':10s} {'N':>5s} | CPI V2 / V3 | comp MAE V2 / V3 | Base incons% V2 / V3 | viol% V2 / V3")
for n in ["256", "1024"]:
    for c in ["t0p1_m0", "t0p1_m0p01", "t10_m0", "t10_m0p01"]:
        f = lambda k, fmt: f"{avg(res, c, n, k):{fmt}} / {avg(v3, c, n, k):{fmt}}"
        print(f"{c:10s} {n:>5s} | {f('mape', '.2f')} | {f('comp_mae', '.3f')} | {f('base_incons', '.2f')} | {f('viol', '.1f')}")
print(f"Saved {OUT}")
