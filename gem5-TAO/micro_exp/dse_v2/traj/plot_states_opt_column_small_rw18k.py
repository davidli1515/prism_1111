#!/usr/bin/env python3
# COPY of plot_states_opt_column_small.py with random search = first 18k designs per benchmark (results_rw18k).
"""plot_states_opt_column.py with the SMALL-start reference states (results_small/state_clusters_v1.npz; PCA fitted
on the small-start designs with the same sampling): one panel per search method, designs coloured by the nearest
reference state, circles = reference centroids, crosses = independent k-means centroids at the method's elbow k
(for Greedy the reference data itself). Writes results_opt_clusters/fig_states_opt_column_small_rw18k.{png,pdf};
refuses to overwrite."""
import os, sys, glob, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_opt_independent as C

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'figure.dpi': 600, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
})
PAL = ["#2E86C1", "#E67E22", "#27AE60", "#C0392B", "#8E44AD", "#7F8C8D"]
HERE = C.HERE; R = os.path.join(HERE, "results_small")
warnings.simplefilter("ignore")


def main():
    outs = [os.path.join(C.OUTD, f"fig_states_opt_column_small_rw18k.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    ref = np.load(os.path.join(R, "state_clusters_v1.npz"))["centroids"]; K = len(ref)
    # PCA exactly as plot_state_pca.py / plot_states_column.py
    X = []
    for b in sorted(d for d in os.listdir(R) if d[0].isdigit()):
        fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
        X.append(C.feat(np.concatenate([np.load(f)["cpi"] for f in fs]), np.concatenate([np.load(f)["mscs"] for f in fs])))
    X = np.concatenate(X); rng = np.random.RandomState(0)
    rng.choice(len(X), 60000, replace=False); sidx = rng.choice(len(X), 15000, replace=False)
    pca = PCA(2).fit(X[sidx]); Pref = pca.transform(ref)
    sets = [("Greedy (reference)", "results_small", "part_*.npz", 6), ("BO", "results_bo", "seed_*.npz", 5),
            ("GA", "results_ga", "seed_*.npz", 5), ("Random", "results_rw18k", "part_*.npz", 5)]
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.9), sharex=True, sharey=True)
    r2 = np.random.RandomState(1)
    for ax, (name, root, pat, kk) in zip(axes.ravel(), sets):
        Xm = C.load(root, pat); L, _ = C.nearest(Xm, ref)
        idx = r2.choice(len(Xm), min(20000, len(Xm)), replace=False); P = pca.transform(Xm[idx])
        samp = Xm[r2.choice(len(Xm), min(120000, len(Xm)), replace=False)]
        Pind = pca.transform(KMeans(kk, n_init=10, random_state=0).fit(samp).cluster_centers_)
        ax.scatter(P[:, 0], P[:, 1], c=[PAL[i] for i in L[idx]], s=0.5, alpha=0.25, lw=0, rasterized=True, zorder=2)
        for s in range(K):
            ax.scatter([Pref[s, 0]], [Pref[s, 1]], s=20, marker="o", facecolors=[PAL[s]], edgecolors="black", linewidths=0.6, zorder=4)
        if "reference" not in name:
            ax.scatter(Pind[:, 0], Pind[:, 1], s=22, marker="x", c="black", linewidth=1.0, zorder=5)
        ax.text(0.04, 0.93, ("Greedy (ref.)" if "reference" in name else name), transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="top", ha="left")
        ax.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)
    ev = pca.explained_variance_ratio_
    h = [plt.Line2D([], [], marker="o", ls="", markersize=3.5, markerfacecolor=PAL[s], markeredgecolor="black",
                    markeredgewidth=0.5, label=f"S{s}") for s in range(K)] + \
        [plt.Line2D([], [], marker="x", ls="", markersize=4, color="black", markeredgewidth=1.0, label="Independent")]
    fig.supylabel(f"PC2 ({100*ev[1]:.0f}%)", fontweight="bold", fontsize=8, x=0.035)
    fig.tight_layout(h_pad=0.4, w_pad=0.8, rect=(0.03, 0.17, 1, 1))
    a, c = axes[1, 0].get_position(), axes[1, 1].get_position(); xc = (a.x0 + c.x1) / 2
    fig.text(xc, a.y0 - 0.075, f"PC1 ({100*ev[0]:.0f}%)", ha="center", va="top", fontweight="bold", fontsize=8)
    fig.legend(handles=h, ncol=7, loc="upper center", bbox_to_anchor=(xc, a.y0 - 0.125), frameon=True, edgecolor="black",
               fancybox=False, framealpha=1, columnspacing=0.5, handlelength=0.8, handletextpad=0.2,
               borderpad=0.3).get_frame().set_linewidth(0.8)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
