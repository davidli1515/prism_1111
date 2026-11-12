#!/usr/bin/env python3
"""Paper panels (a) and (b) redrawn with the four-stage states (results_small/state_clusters_12d.npz), same style and
size as fig_state_centroids_col_small / fig_state_pca_opt_col_small_v2:
 (a) centroid stall shares of the 12 components (FP/BP/SP colour; stage = shade/hatch, stacked Dec, Disp, Iss, Com)
 (b) greedy designs on the first two principal components of the 12-dim feature, coloured by state; circles: state
     centroids; black markers: centroids of BO, GA and random search (18k) clustered independently (elbow k).
Writes results_small/figs/fig_state_centroids_col_12d.{png,pdf} and fig_state_pca_opt_col_12d.{png,pdf};
refuses to overwrite."""
import os, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_states_column as S                                   # rcParams, COL, COL_LIGHT, SIZE, frame, legend
import cluster_states_12d as Q

FIG = os.path.join(Q.R, "figs")
PAL = ["#2E86C1", "#27AE60", "#C0392B", "#8E44AD", "#7F8C8D"]       # S0 frontend .. S2 backend, S3 squash, S4 rare
STAGE = [("Dec.", "dark", ".."), ("Disp.", "dark", None), ("Iss.", "light", None), ("Com.", "light", "///")]
MK = {"BO": dict(marker="x", s=34, linewidths=1.4), "GA": dict(marker="+", s=52, linewidths=1.4),
      "Random": dict(marker="^", s=24, linewidths=0.9, facecolors="white")}
HALO = [pe.withStroke(linewidth=2.6, foreground="white")]


def main():
    names = ("fig_state_centroids_col_12d", "fig_state_pca_opt_col_12d")
    for n in names:
        for e in ("png", "pdf"):
            if os.path.exists(os.path.join(FIG, f"{n}.{e}")):
                sys.exit(f"exists, refusing to overwrite: {n}.{e}")
    Z = np.load(Q.OUT + ".npz"); cent, k = Z["centroids"], int(Z["k"])
    plt.rcParams['hatch.linewidth'] = 0.5

    # ---------------- (a) centroids
    fig, ax = plt.subplots(figsize=S.SIZE)
    for s in range(k):
        bottom = 0
        for i, (_, shade, hatch) in enumerate(STAGE):
            for j, c in enumerate(Q.CAT):
                v = 100 * cent[s, 3 * i + j]
                ax.bar(s, v, bottom=bottom, width=0.65, zorder=3, edgecolor="black", linewidth=0.4,
                       color=S.COL[c] if shade == "dark" else S.COL_LIGHT[c], hatch=hatch)
                bottom += v
    ax.set_xticks(range(k)); ax.set_xticklabels([f"S{s}" for s in range(k)], fontweight="bold")
    ax.set_xlabel("Bottleneck state", fontweight="bold", labelpad=2)
    ax.set_ylabel("Stall share (%)", fontweight="bold", labelpad=2); ax.set_ylim(0, 100); ax.set_xlim(-0.6, k - 0.4)
    ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0); S.frame(ax)
    cat = [Patch(facecolor=S.COL[c], edgecolor="black", linewidth=0.4, label=c) for c in Q.CAT]
    stg = [Patch(facecolor="#808080" if sh == "dark" else "#D0D0D0", edgecolor="black", linewidth=0.4, hatch=h, label=n)
           for n, sh, h in STAGE]
    blank = Patch(facecolor="none", edgecolor="none", label="")
    S.legend(ax, [cat[0], stg[0], cat[1], stg[1], cat[2], stg[2], blank, stg[3]], 4)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{names[0]}.{e}"), dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig); print("saved", names[0])

    # ---------------- (b) PCA with independent centroids
    B = sorted(d for d in os.listdir(Q.R) if d[0].isdigit())
    X = np.concatenate([Q.load("results_small", b, "part_*.npz") for b in B]); L = np.concatenate([Z[f"labels_{b}"] for b in B])
    proj = lambda Y: (Y - Z["pca_mean"]) @ Z["pca_comp"].T; ev = Z["pca_ev"]
    rng = np.random.RandomState(0); idx = rng.choice(len(X), 60000, replace=False); order = rng.permutation(len(idx))
    P, Pc, Ls = proj(X[idx]), proj(cent), L[idx]
    fig, ax = plt.subplots(figsize=S.SIZE)
    ax.scatter(P[order, 0], P[order, 1], c=[PAL[i] for i in Ls[order]], s=0.6, alpha=0.15, lw=0, rasterized=True, zorder=2)
    for s in range(k):
        ax.scatter([Pc[s, 0]], [Pc[s, 1]], s=55, marker="o", facecolors=[PAL[s]], edgecolors="black", linewidths=0.9, zorder=4)
    for name, key in (("BO", "ind_BO"), ("GA", "ind_GA"), ("Random", "ind_Random-18k")):
        m = MK[name]; c = proj(Z[key])
        sc = ax.scatter(c[:, 0], c[:, 1], c="black" if "facecolors" not in m else None,
                        edgecolors="black" if "facecolors" in m else None, zorder=5, **m)
        if "facecolors" not in m:
            sc.set_path_effects(HALO)
    ax.set_xlabel(f"PC1 ({100*ev[0]:.0f}%)", fontweight="bold", labelpad=2)
    ax.set_ylabel(f"PC2 ({100*ev[1]:.0f}%)", fontweight="bold", labelpad=2)
    ax.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0); S.frame(ax)
    h = [plt.Line2D([], [], marker="o", ls="", markersize=4, markerfacecolor=PAL[s], markeredgecolor="black",
                    markeredgewidth=0.6, label=f"S{s}") for s in range(k)]
    h += [plt.Line2D([], [], marker="x", ls="", markersize=4.5, color="black", markeredgewidth=1.4, label="BO"),
          plt.Line2D([], [], marker="+", ls="", markersize=6, color="black", markeredgewidth=1.4, label="GA"),
          plt.Line2D([], [], marker="^", ls="", markersize=4, markerfacecolor="white", markeredgecolor="black",
                     markeredgewidth=0.8, label="Random")]
    S.legend(ax, h, 3)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{names[1]}.{e}"), dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    print("saved", names[1])


if __name__ == "__main__":
    main()
