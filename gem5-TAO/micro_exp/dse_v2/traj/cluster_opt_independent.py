#!/usr/bin/env python3
"""Independent bottleneck-state clustering of the designs evaluated by each search method, compared with the
reference states (results/state_clusters_v1.npz: greedy, 1000 random starts, k = 6). Same feature, elbow rule and
matching as cluster_states.py / robust_clusters.py (read-only imports of nothing; logic copied):
  feature  = stall cycles {FP, BP, SP} x {Dispatch, Commit}, normalised to sum 1
  elbow    = kneedle over k = 2..14 on a sample (<= 120k designs, k-means n_init 3)
  k = 6    = independent k-means (n_init 10) matched to the reference centroids (Hungarian, L2)
Per method: elbow k; matched centroid distance per state; ARI between the independent k = 6 labels and the
nearest-reference-centroid labels on all designs; silhouette (20k sample) with reference labels; distance to the
nearest reference centroid (median / 95th percentile) next to the reference data's own; share of designs per state.
Datasets: Greedy small starts (results_small), BO (results_bo), GA (results_ga), Random 100k (results_rw).
Writes results_opt_clusters/independent_clusters.txt; refuses to overwrite."""
import os, sys, glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__)); OUTD = os.path.join(HERE, "results_opt_clusters")
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def feat(cpi, m):
    X = np.stack([cpi * m[:, 4 * s + k] for s in (1, 3) for k in (1, 2, 3)], 1)
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def load(root, pat):
    fs = sorted(glob.glob(os.path.join(HERE, root, "[0-9]*", pat)))
    return np.concatenate([feat(np.load(f)["cpi"], np.load(f)["mscs"]) for f in fs])


def elbow(X, rs=0):
    ks = list(range(2, 15)); inert = [KMeans(k, n_init=3, random_state=rs).fit(X).inertia_ for k in ks]
    x = (np.array(ks) - 2) / 12; y = (np.array(inert) - inert[-1]) / (inert[0] - inert[-1])
    return ks[int(np.argmax((1 - x) - y))]


def match(C, ref):
    D = np.sqrt(((C[:, None] - ref[None]) ** 2).sum(-1)); r, c = linear_sum_assignment(D)
    perm = np.empty(len(ref), int); perm[r] = c
    return D[r, c], perm


def nearest(X, C):
    d = np.sqrt(((X[:, None] - C[None]) ** 2).sum(-1)); return d.argmin(1), d.min(1)


def main():
    out = os.path.join(OUTD, "independent_clusters.txt")
    if os.path.exists(out):
        sys.exit(f"exists, refusing to overwrite: {out}")
    os.makedirs(OUTD, exist_ok=True)
    Z = np.load(os.path.join(HERE, "results", "state_clusters_v1.npz")); ref = Z["centroids"]; K = int(Z["k"])
    rng = np.random.RandomState(0)
    Xr = load("results", "part_*.npz"); _, dr = nearest(Xr, ref)
    say(f"reference: greedy random starts, {len(Xr)} designs, k = {K}; "
        f"distance to nearest centroid median {np.median(dr):.3f}, p95 {np.percentile(dr, 95):.3f}")
    sets = [("Greedy (small starts)", "results_small", "part_*.npz"), ("BO", "results_bo", "seed_*.npz"),
            ("GA", "results_ga", "seed_*.npz"), ("Random (100k)", "results_rw", "part_*.npz")]
    for name, root, pat in sets:
        X = load(root, pat)
        samp = X[rng.choice(len(X), min(120000, len(X)), replace=False)]
        kk = elbow(samp)
        km = KMeans(K, n_init=10, random_state=0).fit(samp); dist, perm = match(km.cluster_centers_, ref)
        Lref, dmin = nearest(X, ref); Lind = perm[km.predict(X)]
        sidx = rng.choice(len(X), min(20000, len(X)), replace=False)
        say(f"\n== {name}: {len(X)} designs")
        say(f"   elbow k = {kk}")
        say("   k=6 centroids matched to reference (L2): " + ", ".join(f"S{c} {dist[np.where(perm == c)[0][0]]:.3f}" for c in range(K)))
        say(f"   ARI(independent k=6, nearest-reference labels) = {adjusted_rand_score(Lref, Lind):.3f}")
        say(f"   silhouette with reference labels = {silhouette_samples(X[sidx], Lref[sidx]).mean():.2f}")
        say(f"   distance to nearest reference centroid: median {np.median(dmin):.3f}, p95 {np.percentile(dmin, 95):.3f}")
        say("   share per reference state: " + ", ".join(f"S{c} {100*np.mean(Lref == c):.0f}%" for c in range(K)))
    open(out, "w").write("\n".join(OUT) + "\n"); print("wrote", out)


if __name__ == "__main__":
    main()
