#!/usr/bin/env python3
"""PCA view of cluster_12dim_check.py (same data, seeds and k-means). Both panels show the greedy small-start designs
projected on the first two principal components of the 12-dim feature (FP/BP/SP stall cycles at all four stages,
normalised): (a) coloured by the reference 6-dim state, (b) coloured by the 12-dim cluster (k = 6; colour = majority
reference state, the second cluster with an S3 majority in a lighter red). Circles: mean position of each group.
Style of plot_states_column.py. Writes results_small/figs/fig_12dim_pca.{png,pdf}; refuses to overwrite."""
import os, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_states_column as S                                   # rcParams, PAL, frame, legend (read-only)
import cluster_12dim_check as Q

FIG = os.path.join(Q.R, "figs"); LIGHT = {3: "#E6A39B", 0: "#9CC3E4", 2: "#8FD3A9", 1: "#F3BB8A", 4: "#C39BD3", 5: "#BFC5C9"}


def main():
    outs = [os.path.join(FIG, f"fig_12dim_pca.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    fs = sorted(glob.glob(os.path.join(Q.R, "[0-9]*", "part_*.npz")))
    cpi = np.concatenate([np.load(f)["cpi"] for f in fs]); m = np.concatenate([np.load(f)["mscs"] for f in fs])
    X6, X12 = Q.norm(Q.cycles(cpi, m, (1, 3))), Q.norm(Q.cycles(cpi, m, (0, 1, 2, 3)))
    cent = np.load(os.path.join(Q.R, "state_clusters_v1.npz"))["centroids"]; K = len(cent)
    ref = np.concatenate([np.argmin(((X6[i:i + 200000, None] - cent[None]) ** 2).sum(2), 1) for i in range(0, len(X6), 200000)])
    rng = np.random.RandomState(0); idx = rng.choice(len(X6), 120000, replace=False)   # as cluster_12dim_check.py
    l12 = KMeans(K, n_init=10, random_state=0).fit(X12[idx]).predict(X12)
    maj = {c: np.bincount(ref[l12 == c], minlength=K).argmax() for c in range(K)}
    purity = {c: np.bincount(ref[l12 == c], minlength=K).max() / np.sum(l12 == c) for c in range(K)}
    col, seen = {}, set()
    for c in sorted(range(K), key=lambda c: -purity[c]):           # purer cluster keeps the full colour
        col[c] = S.PAL[maj[c]] if maj[c] not in seen else LIGHT[maj[c]]; seen.add(maj[c])

    pidx = np.random.RandomState(2).choice(len(X12), 60000, replace=False)
    pca = PCA(2).fit(X12[np.random.RandomState(3).choice(len(X12), 15000, replace=False)])
    P = pca.transform(X12[pidx]); ev = pca.explained_variance_ratio_; order = np.random.RandomState(4).permutation(len(pidx))
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.55), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, lab, cfun, tag in ((axes[0], ref, lambda g: S.PAL[g], "S"), (axes[1], l12, lambda g: col[g], "C")):
        L = lab[pidx]
        ax.scatter(P[order, 0], P[order, 1], c=[cfun(g) for g in L[order]], s=0.6, alpha=0.25, lw=0, rasterized=True, zorder=2)
        for g in range(K):
            mu = P[L == g].mean(0)
            ax.scatter([mu[0]], [mu[1]], s=22, marker="o", facecolors=[cfun(g)], edgecolors="black", linewidths=0.7, zorder=4)
        ax.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0); S.frame(ax)
    axes[0].set_title("(a) 6-dim states", fontsize=7.5, fontweight="bold", pad=3)
    axes[1].set_title("(b) 12-dim clusters", fontsize=7.5, fontweight="bold", pad=3)
    axes[0].set_ylabel(f"PC2 ({100*ev[1]:.0f}%)", fontweight="bold", labelpad=2)
    fig.text(0.5, -0.04, f"PC1 ({100*ev[0]:.0f}%) of the 12-dim feature", ha="center", va="top", fontweight="bold", fontsize=8)
    h = [plt.Line2D([], [], marker="o", ls="", markersize=4, markerfacecolor=S.PAL[s], markeredgecolor="black",
                    markeredgewidth=0.6, label=f"S{s}") for s in range(K)]
    srt = sorted(range(K), key=lambda c: (maj[c], -purity[c]))
    h2 = [plt.Line2D([], [], marker="o", ls="", markersize=4, markerfacecolor=col[c], markeredgecolor="black",
                     markeredgewidth=0.6, label=f"C (S{maj[c]})") for c in srt]
    axes[0].legend(handles=h, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=True, edgecolor="black",
                   fancybox=False, columnspacing=0.5, handlelength=1.0, handletextpad=0.2, borderpad=0.3, fontsize=6).get_frame().set_linewidth(0.8)
    axes[1].legend(handles=h2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=True, edgecolor="black",
                   fancybox=False, columnspacing=0.5, handlelength=1.0, handletextpad=0.2, borderpad=0.3, fontsize=6).get_frame().set_linewidth(0.8)
    os.makedirs(FIG, exist_ok=True)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)
    print("PC1/PC2 of 12-dim:", ev.round(3), "| cluster -> majority state:", {f"C{c}": f"S{maj[c]} ({100*purity[c]:.0f}%)" for c in range(K)})


if __name__ == "__main__":
    main()
