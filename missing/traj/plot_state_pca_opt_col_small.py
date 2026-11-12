#!/usr/bin/env python3
"""Half-column panel (b) next to fig_state_centroids_col_small: PCA of the small-start reference states (same fit
and sampling as plot_states_column.py on results_small), greedy designs coloured by state, reference centroids as
circles, and the independent k-means centroids of BO, GA and Random (elbow k = 5, same sampling and fit as
cluster_opt_independent_small.py) as black markers. Style of plot_states_column.py.
Writes results_opt_clusters/fig_state_pca_opt_col_small.{png,pdf}; refuses to overwrite."""
import os, sys, glob, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_states_column as S                                # rcParams, PAL, SIZE, frame (read-only)
import cluster_opt_independent as C

HERE = C.HERE; R = os.path.join(HERE, "results_small")
MK = {"BO": dict(marker="x", s=20, linewidths=1.0), "GA": dict(marker="+", s=30, linewidths=1.0),
      "Random": dict(marker="^", s=14, linewidths=0.8, facecolors="none")}
warnings.simplefilter("ignore")


def main():
    outs = [os.path.join(C.OUTD, f"fig_state_pca_opt_col_small.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    Z = np.load(os.path.join(R, "state_clusters_v1.npz")); cent, k = Z["centroids"], int(Z["k"])
    X, L = [], []                                              # as plot_states_column.py panel (b)
    for b in sorted(d for d in os.listdir(R) if d[0].isdigit()):
        fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
        X.append(C.feat(np.concatenate([np.load(f)["cpi"] for f in fs]), np.concatenate([np.load(f)["mscs"] for f in fs])))
        L.append(Z[f"labels_{b}"])
    X, L = np.concatenate(X), np.concatenate(L)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), 60000, replace=False); sidx = rng.choice(len(X), 15000, replace=False)
    pca = PCA(2).fit(X[sidx]); P, Pc = pca.transform(X[idx]), pca.transform(cent); ev = pca.explained_variance_ratio_
    Ls = L[idx]; order = rng.permutation(len(idx))
    ind, r2 = {}, np.random.RandomState(0)                     # same sequence as cluster_opt_independent_small.py
    for name, root, pat in [("BO", "results_bo", "seed_*.npz"), ("GA", "results_ga", "seed_*.npz"), ("Random", "results_rw", "part_*.npz")]:
        Xm = C.load(root, pat)
        samp = Xm[r2.choice(len(Xm), min(120000, len(Xm)), replace=False)]
        kk = C.elbow(samp)
        KMeans(k, n_init=10, random_state=0).fit(samp)
        ind[name] = pca.transform(KMeans(kk, n_init=10, random_state=0).fit(samp).cluster_centers_)
        r2.choice(len(Xm), min(20000, len(Xm)), replace=False)
        print(name, "elbow k", kk)

    fig, ax = plt.subplots(figsize=S.SIZE)
    ax.scatter(P[order, 0], P[order, 1], c=[S.PAL[i] for i in Ls[order]], s=0.6, alpha=0.25, lw=0, rasterized=True, zorder=2)
    for s in range(k):
        ax.scatter([Pc[s, 0]], [Pc[s, 1]], s=22, marker="o", facecolors=[S.PAL[s]], edgecolors="black", linewidths=0.7, zorder=4)
    for name, m in MK.items():
        ax.scatter(ind[name][:, 0], ind[name][:, 1], c="black" if "facecolors" not in m else None,
                   edgecolors="black" if "facecolors" in m else None, zorder=5, **m)
    ax.set_xlabel(f"PC1 ({100*ev[0]:.0f}%)", fontweight="bold", labelpad=2)
    ax.set_ylabel(f"PC2 ({100*ev[1]:.0f}%)", fontweight="bold", labelpad=2)
    ax.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0); S.frame(ax)
    h = [plt.Line2D([], [], marker="o", ls="", markersize=4, markerfacecolor=S.PAL[s], markeredgecolor="black",
                    markeredgewidth=0.6, label=f"S{s}") for s in range(k)]
    h += [plt.Line2D([], [], marker="x", ls="", markersize=4, color="black", markeredgewidth=1.0, label="BO"),
          plt.Line2D([], [], marker="+", ls="", markersize=5, color="black", markeredgewidth=1.0, label="GA"),
          plt.Line2D([], [], marker="^", ls="", markersize=3.5, markerfacecolor="none", markeredgecolor="black",
                     markeredgewidth=0.8, label="Random")]
    S.legend(ax, h, 3)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
