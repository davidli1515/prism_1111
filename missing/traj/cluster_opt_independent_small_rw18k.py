#!/usr/bin/env python3
# COPY of cluster_opt_independent_small.py with random search = first 18k designs per benchmark (results_rw18k).
"""cluster_opt_independent.py with the SMALL-start reference states (results_small/state_clusters_v1.npz, greedy
small starts) instead of the random-start ones; datasets BO, GA, Random 100k. Same feature, elbow, matching and
metrics (functions imported, unchanged). Writes results_opt_clusters/independent_clusters_small_rw18k.txt; never overwrites."""
import os, sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples
import cluster_opt_independent as C

OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def main():
    out = os.path.join(C.OUTD, "independent_clusters_small_rw18k.txt")
    if os.path.exists(out):
        sys.exit(f"exists, refusing to overwrite: {out}")
    Z = np.load(os.path.join(C.HERE, "results_small", "state_clusters_v1.npz")); ref = Z["centroids"]; K = int(Z["k"])
    rng = np.random.RandomState(0)
    Xr = C.load("results_small", "part_*.npz"); _, dr = C.nearest(Xr, ref)
    say(f"reference: greedy small starts, {len(Xr)} designs, k = {K}; "
        f"distance to nearest centroid median {np.median(dr):.3f}, p95 {np.percentile(dr, 95):.3f}")
    for name, root, pat in [("BO", "results_bo", "seed_*.npz"), ("GA", "results_ga", "seed_*.npz"), ("Random (18k)", "results_rw18k", "part_*.npz")]:
        X = C.load(root, pat)
        samp = X[rng.choice(len(X), min(120000, len(X)), replace=False)]
        kk = C.elbow(samp)
        km = KMeans(K, n_init=10, random_state=0).fit(samp); dist, perm = C.match(km.cluster_centers_, ref)
        Lref, dmin = C.nearest(X, ref); Lind = perm[km.predict(X)]
        km_e = KMeans(kk, n_init=10, random_state=0).fit(samp)
        de = np.sqrt(((km_e.cluster_centers_[:, None] - ref[None]) ** 2).sum(-1))
        sidx = rng.choice(len(X), min(20000, len(X)), replace=False)
        say(f"\n== {name}: {len(X)} designs")
        say(f"   elbow k = {kk}; its centroids -> nearest reference state: " + ", ".join(f"S{d.argmin()} ({d.min():.3f})" for d in de))
        say("   k=6 centroids matched to reference (L2): " + ", ".join(f"S{c} {dist[np.where(perm == c)[0][0]]:.3f}" for c in range(K)))
        say(f"   ARI(independent k=6, nearest-reference labels) = {adjusted_rand_score(Lref, Lind):.3f}")
        say(f"   silhouette with reference labels = {silhouette_samples(X[sidx], Lref[sidx]).mean():.2f}")
        say(f"   distance to nearest reference centroid: median {np.median(dmin):.3f}, p95 {np.percentile(dmin, 95):.3f}")
        say("   share per reference state: " + ", ".join(f"S{c} {100*np.mean(Lref == c):.0f}%" for c in range(K)))
    open(out, "w").write("\n".join(OUT) + "\n"); print("wrote", out)


if __name__ == "__main__":
    main()
