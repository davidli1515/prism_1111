#!/usr/bin/env python3
"""Single-panel PCA view of the 6 bottleneck states for the doc (style of window_ids/plot_unseen_simpoint.py).
Same data and transform as plot_state_scatter.py panel (a): random-start walks (results/, 540k states),
feature = 6-dim stall composition [FP,BP,SP] x [Dispatch, Commit], 60k states shown, centroids projected.
Writes results/figs/fig_state_pca.{png,pdf}; refuses to overwrite."""
import os, glob, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 600, 'axes.linewidth': 1.0,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 4, 'ytick.major.size': 4,
})
PAL = ["#2E86C1", "#E67E22", "#27AE60", "#C0392B", "#8E44AD", "#7F8C8D"]
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results"); FIG = os.path.join(R, "figs")


def main():
    outs = [os.path.join(FIG, f"fig_state_pca.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    Z = np.load(os.path.join(R, "state_clusters_v1.npz")); cent, k = Z["centroids"], int(Z["k"])
    X, L = [], []
    for b in sorted(d for d in os.listdir(R) if d[0].isdigit()):
        fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
        cpi = np.concatenate([np.load(f)["cpi"] for f in fs]); m = np.concatenate([np.load(f)["mscs"] for f in fs])
        F = np.stack([cpi * m[:, 4 * s + kk] for s in (1, 3) for kk in (1, 2, 3)], 1)
        X.append(F / F.sum(1, keepdims=True)); L.append(Z[f"labels_{b}"])
    X, L = np.concatenate(X), np.concatenate(L)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), 60000, replace=False); sidx = rng.choice(len(X), 15000, replace=False)
    pca = PCA(2).fit(X[sidx]); P, C = pca.transform(X[idx]), pca.transform(cent); ev = pca.explained_variance_ratio_
    Ls = L[idx]; order = rng.permutation(len(idx))

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.scatter(P[order, 0], P[order, 1], c=[PAL[i] for i in Ls[order]], s=1.2, alpha=0.25, lw=0, rasterized=True, zorder=2)
    for s in range(k):
        ax.scatter(C[s, 0], C[s, 1], s=55, marker="o", c=PAL[s], edgecolor="black", linewidth=1.0, zorder=4)
        ax.annotate(f"S{s}", (C[s, 0], C[s, 1]), xytext=(5, 5), textcoords="offset points", fontsize=10,
                    fontweight="bold", zorder=5)
    ax.set_xlabel(f"PC1 ({100*ev[0]:.0f}% of variance)", fontweight="bold")
    ax.set_ylabel(f"PC2 ({100*ev[1]:.0f}% of variance)", fontweight="bold")
    ax.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.0)
    h = [plt.Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor=PAL[s], markeredgecolor="black",
                    label=f"S{s} ({100*np.mean(L==s):.0f}%)") for s in range(k)]
    ax.legend(handles=h, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=True, edgecolor="black",
              fancybox=False, columnspacing=1.0, handletextpad=0.3).get_frame().set_linewidth(1.0)
    for p in outs:
        fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
