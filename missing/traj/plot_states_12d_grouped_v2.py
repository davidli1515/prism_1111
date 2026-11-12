#!/usr/bin/env python3
# COPY of plot_states_12d_grouped.py with the stage note moved below the legend.
"""Panel (a) of the four-stage states (results_small/state_clusters_12d.npz) as grouped bars, same size/style as
fig_state_centroids_col_small: per state four bars (Dec, Disp, Iss, Com), each stacked FP/BP/SP as the share of that
stage's stall cycles in the state centroid. Writes results_small/figs/fig_state_centroids_col_12d_grouped.{png,pdf};
refuses to overwrite."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_states_column as S                                   # rcParams, COL, SIZE, frame, legend (read-only)
import cluster_states_12d as Q

FIG = os.path.join(Q.R, "figs"); NAME = "fig_state_centroids_col_12d_grouped_v2"


def main():
    outs = [os.path.join(FIG, f"{NAME}.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    Z = np.load(Q.OUT + ".npz"); cent, k = Z["centroids"], int(Z["k"])
    W = cent.reshape(k, 4, 3); W = W / W.sum(2, keepdims=True)
    w, gap = 0.2, 0.22; step = 4 * w + gap
    fig, ax = plt.subplots(figsize=S.SIZE)
    for s in range(k):
        for i in range(4):
            x = s * step + i * w; b = 0
            for j, c in enumerate(Q.CAT):
                v = 100 * W[s, i, j]
                ax.bar(x, v, w, bottom=b, color=S.COL[c], edgecolor="black", linewidth=0.3, zorder=3); b += v
    ax.set_xticks([s * step + 1.5 * w for s in range(k)]); ax.set_xticklabels([f"S{s}" for s in range(k)], fontweight="bold")
    ax.set_xlim(-0.5 * w - 0.1, (k - 1) * step + 3.5 * w + 0.1)
    ax.set_xlabel("Bottleneck state", fontweight="bold", labelpad=2)
    ax.set_ylabel("Share in stage (%)", fontweight="bold", labelpad=2); ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0); S.frame(ax)
    h = [Patch(facecolor=S.COL[c], edgecolor="black", linewidth=0.4, label=c) for c in Q.CAT]
    S.legend(ax, h, 3)
    ax.text(0.5, -0.56, "Bars per state: Dec, Disp, Iss, Com", transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
