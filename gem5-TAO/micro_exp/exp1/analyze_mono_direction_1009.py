#!/usr/bin/env python3
"""
Do monotonicity violations create false cross-stage direction disagreements in MSCS
differences?  (inference only, CPU is enough)

Pairs: two test rows of the SAME trace window (same program region, different random
configurations; both gem5-labeled, neither seen in training). For each pair,
dTrue = MSCS_true(a) - MSCS_true(b), dPred = MSCS_pred(a) - MSCS_pred(b) over the 12
pressure components (FP/BP/SP x 4 stages).

Metrics per model:
  false_disagree : among adjacent-stage pairs (same pressure type) whose TRUE changes move
                   in the same direction (both |d| > EPS), share where the PREDICTED changes
                   move in opposite directions
  sign_agree     : sign agreement of dPred vs dTrue per component (|dTrue| > EPS)
  scancel_excess : mean (S_cancel_pred - S_cancel_true), S_cancel = sum|d| - |sum d|
  scancel_rho    : Spearman(S_cancel_pred, S_cancel_true)
  top_comp_acc   : the component with the largest |dTrue| is also the largest |dPred|
  false_disagree split by whether the pair has a predicted monotonicity violation
  (>0.01 at either endpoint) -> direct link between violations and false disagreement.

Models: exp1 lambda sweep (100% labels, 3 seeds) and exp4 low-label V3 (5%, 1 seed),
lam_task in {0.1, 1} x lam_mono in {0, 0.01}. Same test split (seed 42) for all.
"""
import os, sys, json, glob, re
import numpy as np, torch
from scipy.stats import spearmanr
from collections import defaultdict

MICRO = "/scratch/gem5-TAO/micro_exp"
sys.path.insert(0, f"{MICRO}/exp1"); sys.path.insert(0, f"{MICRO}/window_ids")
from config import GEM_INPUT_DIM, DROPOUT, FP_INDICES, BP_INDICES, SP_INDICES, BASE_INDICES
from model import GEMModel, compute_cpi_mean
from dataset import load_all_benchmarks, create_exp1_dataloaders
from window_split import load_window_ids

EPS = 0.005
OUT = f"{MICRO}/exp1/results/mono_direction_1009.json"
MODELS = {}
for t in ["0p1", "1"]:
    for m in ["0", "0p01"]:
        MODELS[("100%", t, m)] = sorted(glob.glob(f"{MICRO}/exp1/results/lambda_sweep_1004/t{t}_c1_m{m}_s*.pt"))
        MODELS[("5% V3", t, m)] = [f"{MICRO}/exp4/results_lambda_lowlabel_1004/v3_r5_t{t}_c1_m{m}_u1_s42.pt"]
TYPES = {"FP": FP_INDICES, "BP": BP_INDICES, "SP": SP_INDICES}


def test_pairs():
    """Rebuild exp1 split (seed 42) and return test tensors + same-window pairs."""
    names = sorted(f[:-4] for f in os.listdir("/scratch2/wl730/micro_traindata_packed/gem") if f.endswith(".npy"))
    data = load_all_benchmarks("/scratch2/wl730/micro_traindata_packed/gem",
                               "/scratch2/wl730/micro_traindata_packed/concord", names)
    # recover which original row each test row is (same RNG sequence as create_exp1_dataloaders)
    rng = np.random.RandomState(42); te_win = []
    for n in sorted(data):
        N = data[n]["n_samples"]; perm = rng.permutation(N); te = perm[int(N * 15 / 16):]
        te_win.append([f"{n}:{w}" for w in load_window_ids(n, N)[te]])
    _, vl, _, _, _ = create_exp1_dataloaders(data, seed=42)
    del data
    win = np.concatenate(te_win)
    groups = defaultdict(list)
    for i, w in enumerate(win): groups[w].append(i)
    pairs = np.array([(g[a], g[b]) for g in groups.values() if len(g) > 1
                      for a in range(len(g)) for b in range(a + 1, len(g))])
    ds = vl.dataset
    return ds.gem_features, ds.mscs_target.numpy(), ds.cpi_golden, ds.w_min, pairs


def violation(P):
    v = np.concatenate([np.diff(P[:, FP_INDICES], 1).clip(0), np.diff(P[:, SP_INDICES], 1).clip(0),
                        (-np.diff(P[:, BP_INDICES], 1)).clip(0)], 1)
    return v.max(1) > 0.01


def metrics(P, T, pairs):
    a, b = pairs[:, 0], pairs[:, 1]
    dP, dT = P[a] - P[b], T[a] - T[b]
    comps = np.concatenate(list(TYPES.values()))
    # false cross-stage disagreement on adjacent stages of the same type
    fd_num = np.zeros(len(pairs)); fd_den = np.zeros(len(pairs))
    for idx in TYPES.values():
        for s in range(3):
            i, j = idx[s], idx[s + 1]
            same_true = (np.abs(dT[:, i]) > EPS) & (np.abs(dT[:, j]) > EPS) & (np.sign(dT[:, i]) == np.sign(dT[:, j]))
            opp_pred = np.sign(dP[:, i]) != np.sign(dP[:, j])
            fd_den += same_true; fd_num += same_true & opp_pred
    big = np.abs(dT[:, comps]) > EPS
    sign_agree = (np.sign(dP[:, comps]) == np.sign(dT[:, comps]))[big].mean()
    sc = lambda d: np.abs(d[:, comps]).sum(1) - np.abs(d[:, comps].sum(1))
    scP, scT = sc(dP), sc(dT)
    top = (np.abs(dP[:, comps]).argmax(1) == np.abs(dT[:, comps]).argmax(1)).mean()
    viol = violation(P); vpair = viol[a] | viol[b]
    fd = lambda mask: fd_num[mask].sum() / max(fd_den[mask].sum(), 1)
    return dict(false_disagree=fd(np.ones(len(pairs), bool)) * 100, sign_agree=sign_agree * 100,
                scancel_excess=float((scP - scT).mean()), scancel_rho=float(spearmanr(scP, scT).correlation),
                top_comp_acc=top * 100, viol_rate=viol.mean() * 100, pair_viol_share=vpair.mean() * 100,
                false_disagree_viol_pairs=fd(vpair) * 100, false_disagree_clean_pairs=fd(~vpair) * 100)


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists; refusing to overwrite")
    X, T, Y, W, pairs = test_pairs()
    print(f"test rows {len(T)}, same-window pairs {len(pairs)}")
    res = {}
    for key, cks in MODELS.items():
        rows = []
        for ck in cks:
            m = GEMModel(GEM_INPUT_DIM, [512, 256], DROPOUT)
            m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
            with torch.no_grad():
                P = torch.cat([m(X[i:i + 20000]) for i in range(0, len(X), 20000)])
                mape = ((compute_cpi_mean(P, W) - Y).abs() / Y).mean().item() * 100
            saved = json.load(open(ck[:-3] + ".json"))["selected"]["mape_mean4"]
            assert abs(mape - saved) < 0.05, f"{ck}: recomputed {mape:.2f} vs saved {saved:.2f}"
            r = metrics(P.numpy(), T, pairs); r["mape"] = mape; rows.append(r)
        res["|".join(key)] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {"n_seeds": len(rows)}
        s = res["|".join(key)]
        print(f"{key[0]:6s} lam_task={key[1]:>3s} lam_mono={key[2]:>4s} (n={len(rows)}) | viol {s['viol_rate']:5.1f}% | "
              f"false-disagree {s['false_disagree']:5.1f}% (viol pairs {s['false_disagree_viol_pairs']:5.1f} / "
              f"clean {s['false_disagree_clean_pairs']:5.1f}) | sign {s['sign_agree']:5.1f}% | "
              f"S_cancel excess {s['scancel_excess']:.4f} rho {s['scancel_rho']:.3f} | top comp {s['top_comp_acc']:5.1f}%")
    json.dump({"n_pairs": int(len(pairs)), "eps": EPS, "results": res}, open(OUT, "w"), indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
