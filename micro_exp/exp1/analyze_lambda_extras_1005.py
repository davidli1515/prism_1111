#!/usr/bin/env python3
"""
Extra probes for costs of a large lam_task (inference / logs only, no training).
Checkpoints: exp1/results/lambda_sweep (lam_task 0.1,1,10) + lambda_probe
(30,100), lam_const = 1, lam_mono in {0, 0.01}, 3 seeds each, 100% labels.

  (1) physics consistency on UNSEEN SimPoint phases (7 benchmarks whose other simpoints
      share no trace window with 18k): cross-stage monotonicity violations (>0.01),
      cross-stage Base spread, decomposition MAE vs ground truth.
  (2) extreme configurations: test rows whose arch params are far from the training
      mean (mean |z| over gd[3850:3873], top 10%) vs the rest -> CPI MAPE.
  (3) training dynamics from saved histories: best epoch, last-minus-best MAPE,
      std of test MAPE over the last 100 epochs.

Output: exp1/results/lambda_extras_1005.json + printed tables.
Usage: python analyze_lambda_extras_1005.py --gpu 3
"""
import os, sys, re, glob, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MICRO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from config import (GEM_INPUT_DIM, DROPOUT, BENCHMARK_NAMES, SEED, D3_FEAT_END, MSCS_START, MSCS_END,
                    WMIN_INDICES, BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES)
from model import GEMModel, compute_cpi_mean
from dataset import load_all_benchmarks, create_exp1_dataloaders

PACKED_18K = "/data/micro_traindata_packed"
PACKED_40K = "/data/micro_traindata_packed_all/gem"
UNSEEN_DIR = os.path.join(MICRO, "window_ids", "unseen_simpoint")
UNSEEN7 = ["508.namd_r", "638.imagick_s", "607.cactuBSSN_s", "526.blender_r",
           "623.xalancbmk_s", "631.deepsjeng_s", "641.leela_s"]
CKPT_GLOBS = [f"{HERE}/results/lambda_sweep/t*_c1_m0_s*.pt",
              f"{HERE}/results/lambda_sweep/t*_c1_m0p01_s*.pt",
              f"{HERE}/results/lambda_probe/t*_c1_m*_s*.pt"]
OUT = f"{HERE}/results/lambda_extras_1005.json"
ARCH = slice(3850, 3873)


def violations(P):
    fp, bp, sp = P[:, FP_INDICES], P[:, BP_INDICES], P[:, SP_INDICES]
    v = torch.cat([(fp[:, 1:] - fp[:, :-1]).clamp(min=0), (sp[:, 1:] - sp[:, :-1]).clamp(min=0),
                   (bp[:, :-1] - bp[:, 1:]).clamp(min=0)], 1).max(1).values
    return (v > 0.01).float().mean().item() * 100


def spread(P):
    b = P[:, BASE_INDICES]
    return ((b.max(1).values - b.min(1).values) / b.mean(1)).mean().item() * 100


def decomp_mae(P, M):
    return np.mean([(P[:, I] - M[:, I]).abs().mean().item() for I in (FP_INDICES, BP_INDICES, SP_INDICES)])


def mape(p, t):
    return ((p - t).abs() / t.abs()).mean().item() * 100


@torch.no_grad()
def predict(model, X, bs=50000):
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=3)
    dev = torch.device(f"cuda:{ap.parse_args().gpu}")
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists, refusing to overwrite")

    data = load_all_benchmarks(f"{PACKED_18K}/gem", f"{PACKED_18K}/concord", BENCHMARK_NAMES)
    _, vl, gem_norm, _, _ = create_exp1_dataloaders(data, seed=SEED)
    del data
    ds = vl.dataset
    Xte, Mte, Yte, Wte = (ds.__dict__[k].to(dev) for k in ["gem_features", "mscs_target", "cpi_golden", "w_min"])
    ext = Xte[:, ARCH].abs().mean(1)                       # features are already train z-scores
    far = ext >= torch.quantile(ext, 0.9)

    U = {}
    for b in UNSEEN7:
        rows = np.asarray(np.load(f"{PACKED_40K}/{b}.npy", mmap_mode="r")[np.sort(np.load(f"{UNSEEN_DIR}/{b}.npy"))],
                          dtype=np.float32)
        U[b] = {"X": torch.from_numpy(gem_norm.transform(rows[:, :D3_FEAT_END]).astype(np.float32)).to(dev),
                "M": torch.from_numpy(rows[:, MSCS_START:MSCS_END].copy()).to(dev)}
    gt_viol = np.mean([violations(U[b]["M"]) for b in UNSEEN7])
    print(f"ground-truth violation rate on unseen simpoints: {gt_viol:.2f}%")

    res = {}
    ckpts = sorted({p for g in CKPT_GLOBS for p in glob.glob(g)})
    for ck in ckpts:
        t, m, s = re.search(r"t([\dp]+)_c1_m([\dp]+)_s(\d+)\.pt$", ck).groups()
        key = (float(t.replace("p", ".")), float(m.replace("p", ".")), int(s))
        model = GEMModel(GEM_INPUT_DIM, [512, 256], DROPOUT).to(dev)
        model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
        r = {}
        # (1) unseen simpoint physics consistency
        Pu = {b: predict(model, U[b]["X"]) for b in UNSEEN7}
        r["unseen_viol"] = np.mean([violations(Pu[b]) for b in UNSEEN7])
        r["unseen_spread"] = np.mean([spread(Pu[b]) for b in UNSEEN7])
        r["unseen_decomp_mae"] = np.mean([decomp_mae(Pu[b], U[b]["M"]) for b in UNSEEN7])
        # (2) extreme configs (seen test set)
        P = predict(model, Xte); C = compute_cpi_mean(P, Wte)
        r["far_mape"], r["near_mape"] = mape(C[far], Yte[far]), mape(C[~far], Yte[~far])
        r["seen_viol"] = violations(P)
        # (3) training dynamics
        h = json.load(open(ck[:-3] + ".json"))
        hist = np.array([e["mape_mean4"] for e in h["history"]])
        r["best_epoch"] = int(h["best_epoch"])
        r["last_minus_best"] = float(hist[-1] - hist.min())
        r["last100_std"] = float(hist[-100:].std())
        res[str(key)] = r
        print(f"{key}: unseen viol={r['unseen_viol']:.1f}% spread={r['unseen_spread']:.2f}% "
              f"decomp={r['unseen_decomp_mae']:.3f} | far={r['far_mape']:.2f} near={r['near_mape']:.2f} "
              f"| best_ep={r['best_epoch']} last-best={r['last_minus_best']:.2f} std100={r['last100_std']:.3f}",
              flush=True)
    json.dump({"gt_unseen_viol": gt_viol, "far_threshold_quantile": 0.9, "results": res},
              open(OUT, "w"), indent=2)

    print("\nlam_task lam_mono | unseen viol | unseen spread | unseen decomp | far MAPE | near MAPE | "
          "last-best | std(last100)   (mean ± std over seeds)")
    groups = {}
    for k, r in res.items():
        t, m, _ = eval(k); groups.setdefault((t, m), []).append(r)
    for (t, m) in sorted(groups):
        g = groups[(t, m)]
        f = lambda k, d=2: f"{np.mean([x[k] for x in g]):.{d}f}±{np.std([x[k] for x in g]):.{d}f}"
        print(f"{t:>6g} {m:>6g} | {f('unseen_viol', 1):>11} | {f('unseen_spread'):>13} | "
              f"{f('unseen_decomp_mae', 3):>13} | {f('far_mape'):>9} | {f('near_mape'):>9} | "
              f"{f('last_minus_best'):>9} | {f('last100_std', 3)}  n={len(g)}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
