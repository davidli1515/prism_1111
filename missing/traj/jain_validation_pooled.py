#!/usr/bin/env python3
"""Jain-index error of PRISM vs gem5 on the pooled validation set of settings_validation_pooled.py (1,273 unique
designs; loader imported, unchanged). Per stage s, J_s = (sum_k c_ks)^2 / (3 sum_k c_ks^2) over FP/BP/SP stall cycles.
Reports per stage and per method: MAE, median absolute error, Spearman, Pearson; 95% bootstrap over benchmarks for MAE.
Writes gem5_val4/jain_validation_pooled.txt (never overwrites)."""
import os, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import settings_validation_pooled as P
OUT = os.path.join(HERE, "gem5_val4", "jain_validation_pooled.txt")
ST = ("Dec", "Disp", "Iss", "Com")


def jain(cpi, m):
    J = []
    for s in range(4):
        x = np.stack([cpi * m[:, 4 * s + k] for k in (1, 2, 3)], 1)
        J.append(x.sum(1) ** 2 / np.maximum(3 * (x ** 2).sum(1), 1e-12))
    return np.stack(J, 1)


def main():
    if os.path.exists(OUT):
        sys.exit(f"exists, refusing to overwrite: {OUT}")
    rows = P.load()
    Jm = jain(np.array([r["mc"] for r in rows]), np.stack([r["mm"] for r in rows]))
    Jg = jain(np.array([r["gc"] for r in rows]), np.stack([r["gm"] for r in rows]))
    L = [f"{len(rows)} designs; Jain of FP/BP/SP stall cycles per stage (range 1/3 .. 1)",
         f"{'method':8s}{'stage':>6s}{'MAE':>8s}{'medAE':>8s}{'Spearman':>10s}{'Pearson':>9s}{'mean PRISM/gem5':>18s}"]
    for name in ["Greedy", "BO", "GA", "Random", "All"]:
        ii = np.arange(len(rows)) if name == "All" else np.array([i for i, r in enumerate(rows) if r["method"] == name])
        for s in range(4):
            a, b = Jm[ii, s], Jg[ii, s]
            L.append(f"{name:8s}{ST[s]:>6s}{np.mean(abs(a-b)):8.3f}{np.median(abs(a-b)):8.3f}"
                     f"{spearmanr(a, b).correlation:10.2f}{pearsonr(a, b)[0]:9.2f}{np.mean(a):9.3f}/{np.mean(b):.3f}")
    B = sorted({r["bench"] for r in rows}); by = {b: [i for i, r in enumerate(rows) if r["bench"] == b] for b in B}
    rng = np.random.RandomState(0); bs = []
    for _ in range(1000):
        ii = np.array([i for b in rng.choice(B, len(B)) for i in by[b]]); bs.append(np.mean(abs(Jm[ii] - Jg[ii]), 0))
    bs = np.array(bs)
    L.append("All, MAE 95% bootstrap over benchmarks: " + ", ".join(
        f"{ST[s]} [{np.percentile(bs[:, s], 2.5):.3f}, {np.percentile(bs[:, s], 97.5):.3f}]" for s in range(4)))
    print("\n".join(L)); open(OUT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
