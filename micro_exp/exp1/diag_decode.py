#!/usr/bin/env python3
"""
Diagnose why decode-stage BP-vs-SP ranking accuracy is 34%.
Usage: python3 diag_decode.py
"""
import sys
import numpy as np, torch
sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")
from config import *
from model import *
from dataset import *

PRISM_CKPT = "/path/to/gem5/micro_exp/exp1/results/exp1_v3_run2/exp1_512_256_183k/gem_full.pt"

torch.manual_seed(SEED); np.random.seed(SEED)
all_data = load_all_benchmarks("/data/micro_traindata_packed/gem",
                                "/data/micro_traindata_packed/concord", None)
tl, vl, _, _, bm_idx = create_exp1_dataloaders(all_data)
dev = torch.device("cuda:0")

m = GEMModel(GEM_INPUT_DIM, [512,256], DROPOUT).to(dev)
m.load_state_dict(torch.load(PRISM_CKPT, map_location=dev)); m.eval()

P, T = [], []
with torch.no_grad():
    for b in vl:
        P.append(m(b["gem_features"].to(dev)).cpu())
        T.append(b["mscs_target"])
P, T = torch.cat(P).numpy(), torch.cat(T).numpy()

# Decode stage: BP=index 2, SP=index 3
bp_p, sp_p = P[:, 2], P[:, 3]
bp_t, sp_t = T[:, 2], T[:, 3]

print("=== Decode stage BP vs SP diagnosis ===\n")

# Distributions
print(f"True BP:  mean={bp_t.mean():.4f}, std={bp_t.std():.4f}, median={np.median(bp_t):.4f}")
print(f"True SP:  mean={sp_t.mean():.4f}, std={sp_t.std():.4f}, median={np.median(sp_t):.4f}")
print(f"Pred BP:  mean={bp_p.mean():.4f}, std={bp_p.std():.4f}")
print(f"Pred SP:  mean={sp_p.mean():.4f}, std={sp_p.std():.4f}")

# How often is BP > SP (true vs pred)
print(f"\nTrue: BP > SP in {(bp_t > sp_t).mean()*100:.1f}% of samples")
print(f"Pred: BP > SP in {(bp_p > sp_p).mean()*100:.1f}% of samples")

# Magnitude of the difference
diff_t = bp_t - sp_t
print(f"\nTrue (BP - SP): mean={diff_t.mean():+.4f}, |diff| mean={np.abs(diff_t).mean():.4f}")

# Ranking accuracy at different thresholds
print(f"\nRanking accuracy vs threshold on |true BP - SP|:")
for eps in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]:
    mask = np.abs(diff_t) > eps
    if mask.sum() > 0:
        acc = ((bp_p[mask] > sp_p[mask]) == (bp_t[mask] > sp_t[mask])).mean() * 100
        print(f"  |diff| > {eps:.2f}: {acc:.1f}%  ({mask.sum()} samples, {mask.mean()*100:.0f}%)")

# Are both near zero?
both_small = (bp_t < 0.05) & (sp_t < 0.05)
print(f"\nSamples where both BP,SP < 0.05: {both_small.mean()*100:.1f}%")
print(f"  Among these, ranking acc: {((bp_p[both_small] > sp_p[both_small]) == (bp_t[both_small] > sp_t[both_small])).mean()*100:.1f}%")

# Predicted collapse? Check if model always predicts BP<SP or BP>SP
print(f"\nModel prediction bias:")
print(f"  Pred always BP < SP? {(bp_p < sp_p).mean()*100:.1f}% of time predicts BP < SP")
print(f"  True  BP < SP: {(bp_t < sp_t).mean()*100:.1f}% of time")

# The key question: does model collapse BP and SP to similar values?
pred_gap = np.abs(bp_p - sp_p)
print(f"\nPredicted |BP - SP|: mean={pred_gap.mean():.4f} (vs true {np.abs(diff_t).mean():.4f})")
print(f"  If pred gap << true gap, model can't distinguish them")
