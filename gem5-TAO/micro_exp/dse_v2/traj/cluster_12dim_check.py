#!/usr/bin/env python3
"""Robustness of the six bottleneck states to the stage choice: re-cluster the greedy small-start designs with all
four stages (12-dim feature: FP/BP/SP stall cycles at Decode, Dispatch, Issue, Commit, normalised to sum 1) and
compare with the reference 6-dim states (Dispatch + Commit; results_small/state_clusters_v1.npz, nearest centroid).
Same procedure as cluster_states.py: 120k sample (seed 0), elbow (kneedle) over k = 2..14, k-means n_init 10.
Reported:
  1. elbow k of the 12-dim feature
  2. ARI(12-dim k=6, reference labels) on all designs; references: 6-dim re-clustered on a different 120k sample
     (seed 1) and 6-dim on the same sample with a different k-means seed
  3. each 12-dim cluster -> majority reference state (share), and its Dispatch BP fraction (order along the axis)
  4. PC1 explained variance, 6-dim vs 12-dim
  5. stage redundancy: Spearman between stages of the stall cycles of each category (50k sample)
Read-only. Writes results_small/cluster_12dim_check.txt (never overwrites)."""
import os, sys, glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results_small")
OUT = os.path.join(R, "cluster_12dim_check.txt")
ST, CAT = ("Dec", "Disp", "Iss", "Com"), ("FP", "BP", "SP")
L = []


def say(s=""):
    print(s, flush=True); L.append(s)


def cycles(cpi, m, stages):
    return np.stack([cpi * m[:, 4 * s + k] for s in stages for k in (1, 2, 3)], 1)


def norm(X):
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def kneedle(ks, inert):
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0]); y = (np.array(inert) - inert[-1]) / (inert[0] - inert[-1])
    return ks[int(np.argmax((1 - x) - y))]


def main():
    if os.path.exists(OUT):
        sys.exit(f"exists, refusing to overwrite: {OUT}")
    fs = sorted(glob.glob(os.path.join(R, "[0-9]*", "part_*.npz")))
    cpi = np.concatenate([np.load(f)["cpi"] for f in fs]); m = np.concatenate([np.load(f)["mscs"] for f in fs])
    X6, X12 = norm(cycles(cpi, m, (1, 3))), norm(cycles(cpi, m, (0, 1, 2, 3)))
    cent = np.load(os.path.join(R, "state_clusters_v1.npz"))["centroids"]; K = len(cent)
    ref = np.concatenate([np.argmin(((X6[i:i + 200000, None] - cent[None]) ** 2).sum(2), 1) for i in range(0, len(X6), 200000)])
    say(f"greedy small-start designs: {len(X6)}")

    rng = np.random.RandomState(0); idx = rng.choice(len(X6), 120000, replace=False)
    ks = list(range(2, 15)); inert = [KMeans(k, n_init=4, random_state=0).fit(X12[idx]).inertia_ for k in ks]
    say(f"1. elbow k, 12-dim: {kneedle(ks, inert)}  (6-dim reference: {K})")

    km12 = KMeans(K, n_init=10, random_state=0).fit(X12[idx]); l12 = km12.predict(X12)
    idx1 = np.random.RandomState(1).choice(len(X6), 120000, replace=False)
    l6b = KMeans(K, n_init=10, random_state=0).fit(X6[idx1]).predict(X6)
    l6c = KMeans(K, n_init=10, random_state=1).fit(X6[idx]).predict(X6)
    say(f"2. ARI vs reference states: 12-dim (k=6) {adjusted_rand_score(ref, l12):.3f} | "
        f"6-dim other sample {adjusted_rand_score(ref, l6b):.3f} | 6-dim other k-means seed {adjusted_rand_score(ref, l6c):.3f}")

    say("3. 12-dim clusters (sorted by Dispatch BP fraction): size, majority reference state (share of the cluster), "
        "Dispatch BP fraction of Dispatch stalls; reference states for comparison")
    dbp = X6[:, 1] / np.maximum(X6[:, :3].sum(1), 1e-12)
    rows = []
    for c in range(K):
        mk = l12 == c; h = np.bincount(ref[mk], minlength=K) / mk.sum()
        rows.append((np.median(dbp[mk]), c, mk.mean(), h.argmax(), h.max()))
    for f, c, sz, s, sh in sorted(rows):
        say(f"   C{c}: {100*sz:5.1f}% of designs -> S{s} ({100*sh:3.0f}%), Dispatch BP {100*f:4.0f}%")
    say("   reference: " + ", ".join(f"S{s} {100*np.median(dbp[ref == s]):.0f}%" for s in sorted(range(K), key=lambda s: np.median(dbp[ref == s]))))
    ct = np.zeros((K, K), int)
    for a, b in zip(l12, ref):
        ct[a, b] += 1
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-ct)
    say(f"   designs in the matched (one-to-one) cluster: {100*ct[r, c].sum()/len(ref):.1f}%")

    sidx = rng.choice(len(X6), 50000, replace=False)
    say(f"4. PC1 explained variance: 6-dim {100*PCA(2).fit(X6[sidx]).explained_variance_ratio_[0]:.0f}%, "
        f"12-dim {100*PCA(2).fit(X12[sidx]).explained_variance_ratio_[0]:.0f}%")

    say("5. Spearman between stages of stall cycles per instruction (50k designs)")
    C = cycles(cpi[sidx], m[sidx], (0, 1, 2, 3)).reshape(-1, 4, 3)
    for k, name in enumerate(CAT):
        pairs = [(0, 1), (1, 2), (2, 3), (1, 3)]
        say(f"   {name}: " + ", ".join(f"{ST[a]}-{ST[b]} {spearmanr(C[:, a, k], C[:, b, k]).correlation:.2f}" for a, b in pairs))
    sh = (C.sum(2) / np.maximum(C.sum((1, 2))[:, None], 1e-12)).mean(0)
    say("   mean share of the 12-dim stall total per stage: " + ", ".join(f"{s} {100*v:.0f}%" for s, v in zip(ST, sh)))
    open(OUT, "w").write("\n".join(L) + "\n"); print("wrote", OUT)


if __name__ == "__main__":
    main()
