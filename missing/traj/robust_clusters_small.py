#!/usr/bin/env python3
"""Bootstrap stability of the small-start reference states (results_small/state_clusters_v1.npz, k = 6), as part A
of robust_clusters.py: 30 x (100 random start ids out of the 1000 small-start walks, all 17 benchmarks) ->
elbow k, k = 6 k-means matched to the reference (Hungarian), ARI of the induced labels vs the reference labels on
ALL small-start designs. Also: overall share per state, silhouette (20k sample), PC1 variance, pooled transition
probabilities. Writes results_small/robust_clusters_small.txt (never overwrites)."""
import os, sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples
import robust_clusters as RC

HERE = RC.HERE; R = os.path.join(HERE, "results_small"); OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def main():
    out = os.path.join(R, "robust_clusters_small.txt")
    if os.path.exists(out):
        sys.exit(f"exists, refusing to overwrite: {out}")
    Z = np.load(os.path.join(R, "state_clusters_v1.npz")); ref, K = Z["centroids"], int(Z["k"])
    B = sorted(d for d in os.listdir(R) if d[0].isdigit())
    D = {b: RC.load(R, b) for b in B}
    X = np.concatenate([D[b]["X"] for b in B]); L = np.concatenate([Z[f"labels_{b}"] for b in B])
    st = {b: D[b]["start"] for b in B}
    rng = np.random.RandomState(0); reps = []
    for r in range(30):
        ids = rng.choice(1000, 100, replace=False)
        sub = np.concatenate([D[b]["X"][np.isin(st[b], ids)] for b in B])
        kk = RC.elbow(sub, r); km = KMeans(K, n_init=10, random_state=r).fit(sub); dist, perm = RC.match(km.cluster_centers_, ref)
        reps.append((kk, dist.max(), adjusted_rand_score(L, perm[km.predict(X)])))
    A = np.array(reps, float)
    ks, cnt = np.unique(A[:, 0].astype(int), return_counts=True)
    say(f"small-start reference: {len(X)} designs, k = {K}")
    say(f"bootstrap (30 x 100 starts): elbow k {dict(zip(ks.tolist(), cnt.tolist()))}; worst matched centroid distance "
        f"median {np.median(A[:,1]):.3f} (max {A[:,1].max():.3f}); ARI vs reference median {np.median(A[:,2]):.3f} (min {A[:,2].min():.3f})")
    sidx = rng.choice(len(X), 20000, replace=False)
    say(f"silhouette (reference labels, 20k sample): {silhouette_samples(X[sidx], L[sidx]).mean():.2f}")
    say(f"PC1 / PC2 variance (fit on 15k sample): " + ", ".join(f"{v:.3f}" for v in PCA(2).fit(X[rng.choice(len(X), 15000, replace=False)]).explained_variance_ratio_))
    say("share per state: " + ", ".join(f"S{s} {100*np.mean(L == s):.1f}%" for s in range(K)))
    say("within-stage shares of each centroid (Dispatch FP/BP/SP | Commit FP/BP/SP), %:")
    for s, c in enumerate(ref):
        d, m = c[:3] / c[:3].sum(), c[3:] / c[3:].sum()
        say(f"  S{s}: " + " ".join(f"{100*v:5.1f}" for v in d) + " | " + " ".join(f"{100*v:5.1f}" for v in m))
    T = np.zeros((K, K))
    for b in B:
        lab = Z[f"labels_{b}"]; same = st[b][1:] == st[b][:-1]; a, c = lab[:-1][same], lab[1:][same]; ch = a != c
        np.add.at(T, (a[ch], c[ch]), 1)
    P = T / np.maximum(T.sum(1, keepdims=True), 1)
    say("transition probabilities (row = from, among state changes), %:")
    for i in range(K):
        say(f"  S{i}: " + " ".join(f"{100*v:5.0f}" for v in P[i]) + f"   (n = {int(T[i].sum())})")
    open(out, "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
