#!/usr/bin/env python3
"""
Does L_mono improve the MSCS itself (not CPI)? Inference-only comparison of
existing exp4 checkpoints on the 18k row-split test set.

Metrics per checkpoint:
  MAPE                 CPI error (sanity check vs. saved JSON)
  MSE Base/FP/BP/SP    per-type MSE averaged over the 4 stages (x1e3)
  viol                 % test samples with any monotonicity violation
                       (FP or SP increasing / BP decreasing from Decode to Commit)
  dom@stage            % correct dominant stall type (argmax of FP/BP/SP) per stage
  rank                 % correct full FP/BP/SP ordering, averaged over stages

Normalizer reproduced exactly via exp1 create_exp1_dataloaders(seed) (same split
and fit as exp4 for 18k-only variants).

Usage: python eval_mono_mscs.py --gpu 1
"""
import os, sys, json, argparse
import numpy as np, torch

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")
from config import SEED, BENCHMARK_NAMES, GEM_INPUT_DIM, DROPOUT, BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES
from model import GEMModel, compute_cpi_mean
from dataset import load_all_benchmarks, create_exp1_dataloaders

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {   # label -> (dir, ckpt prefix)
    "V2 supervised":      ("results_run2/exp4_18k_supervised_only/checkpoints", "supervised_only"),
    "V3 full":            ("results_run2/exp4_18k/checkpoints", "full"),
    "V3 no-mono (L)":     ("results_ablation/exp4_18k/checkpoints", "full"),
    "V3 no-mono (U)":     ("results_ablation/exp4_18k_const_only/checkpoints", "const_only"),
}
RATIOS = [5, 15, 20, 30, 50, 90]
STAGES = ["Dec", "Dis", "Iss", "Com"]


def metrics(P, T, cpi_p, cpi_t):
    m = {"MAPE": float((np.abs(cpi_p - cpi_t) / cpi_t * 100).mean())}
    for name, idx in (("Base", BASE_INDICES), ("FP", FP_INDICES), ("BP", BP_INDICES), ("SP", SP_INDICES)):
        m[f"MSE_{name}"] = float(((P[:, idx] - T[:, idx]) ** 2).mean() * 1e3)
    fp, bp, sp = P[:, FP_INDICES], P[:, BP_INDICES], P[:, SP_INDICES]
    viol = (np.diff(fp, axis=1) > 0).any(1) | (np.diff(sp, axis=1) > 0).any(1) | (np.diff(bp, axis=1) < 0).any(1)
    m["viol"] = float(viol.mean() * 100)
    ranks = []
    for s in range(4):
        cols = [FP_INDICES[s], BP_INDICES[s], SP_INDICES[s]]
        m[f"dom@{STAGES[s]}"] = float((P[:, cols].argmax(1) == T[:, cols].argmax(1)).mean() * 100)
        ranks.append((np.argsort(-P[:, cols], 1) == np.argsort(-T[:, cols], 1)).all(1).mean() * 100)
    m["rank"] = float(np.mean(ranks))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=os.path.join(HERE, "eval_mono_mscs.json"))
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    data = load_all_benchmarks("/data/micro_traindata_packed/gem",
                               "/data/micro_traindata_packed/concord", BENCHMARK_NAMES)
    _, test_loader, _, _, _ = create_exp1_dataloaders(data, seed=args.seed)
    del data
    ds = test_loader.dataset
    X, T = ds.gem_features, ds.mscs_target.numpy()
    W, cpi_t = ds.w_min, ds.cpi_golden.numpy()

    T_viol = ((np.diff(T[:, FP_INDICES], 1) > 0).any(1) | (np.diff(T[:, SP_INDICES], 1) > 0).any(1)
              | (np.diff(T[:, BP_INDICES], 1) < 0).any(1)).mean() * 100
    print(f"Ground-truth monotonicity violation rate: {T_viol:.2f}%")

    res = {}
    cols = ["MAPE", "MSE_Base", "MSE_FP", "MSE_BP", "MSE_SP", "viol"] + [f"dom@{s}" for s in STAGES] + ["rank"]
    print(f"\n{'variant':16s} {'ratio':>5s} " + " ".join(f"{c:>8s}" for c in cols))
    for r in RATIOS:
        for label, (d, pre) in VARIANTS.items():
            ck = os.path.join(HERE, d, f"{pre}_{r}pct.pt")
            if not os.path.exists(ck):
                continue
            model = GEMModel(GEM_INPUT_DIM, [512, 256], DROPOUT).to(dev)
            model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
            with torch.no_grad():
                P = torch.cat([model(X[s:s + 50000].to(dev)).cpu() for s in range(0, len(X), 50000)])
                cpi_p = compute_cpi_mean(P, W).numpy()
            m = metrics(P.numpy(), T, cpi_p, cpi_t)
            res[f"{label}|{r}"] = m
            print(f"{label:16s} {r:5d} " + " ".join(f"{m[c]:8.2f}" if c != 'MSE_Base' else f"{m[c]:8.3f}" for c in cols))
        print()
    json.dump({"gt_violation_pct": float(T_viol), "results": res}, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
