#!/usr/bin/env python3
"""Figure for cluster_12dim_check.py (same data, seeds and k-means): greedy small-start designs clustered with the
6-dim feature (reference states) and with the 12-dim feature (all four stages, k = 6).
 (a) reference states S0..S5: mean FP/BP/SP share within each stage (Dec, Disp, Iss, Com) of their designs
 (b) 12-dim clusters, same view, ordered by Dispatch BP fraction and labelled by their majority reference state
 (c) row-normalised contingency: 12-dim cluster vs reference state
Style of plot_states_column.py. Writes results_small/figs/fig_12dim_check.{png,pdf}; refuses to overwrite."""
import os, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_states_column as S                                   # rcParams, COL, frame, legend (read-only)
import cluster_12dim_check as Q

FIG = os.path.join(Q.R, "figs"); ST = ("Dec", "Disp", "Iss", "Com"); CAT = ("FP", "BP", "SP")
REF_ORDER = [0, 2, 1, 3, 4, 5]


def bars(ax, comp, labels, title):
    """comp: (n, 4 stages, 3 cats) within-stage shares."""
    w, gap = 0.2, 0.25
    for i in range(len(comp)):
        for s in range(4):
            x = i * (4 * w + gap) + s * w; b = 0
            for k, c in enumerate(CAT):
                ax.bar(x, 100 * comp[i, s, k], w, bottom=b, color=S.COL[c], edgecolor="black", linewidth=0.3)
                b += 100 * comp[i, s, k]
    ax.set_xticks([i * (4 * w + gap) + 1.5 * w for i in range(len(comp))]); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100]); S.frame(ax)
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=3)


def main():
    outs = [os.path.join(FIG, f"fig_12dim_check.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    fs = sorted(glob.glob(os.path.join(Q.R, "[0-9]*", "part_*.npz")))
    cpi = np.concatenate([np.load(f)["cpi"] for f in fs]); m = np.concatenate([np.load(f)["mscs"] for f in fs])
    C12 = Q.cycles(cpi, m, (0, 1, 2, 3)); X6, X12 = Q.norm(Q.cycles(cpi, m, (1, 3))), Q.norm(C12)
    cent = np.load(os.path.join(Q.R, "state_clusters_v1.npz"))["centroids"]; K = len(cent)
    ref = np.concatenate([np.argmin(((X6[i:i + 200000, None] - cent[None]) ** 2).sum(2), 1) for i in range(0, len(X6), 200000)])
    idx = np.random.RandomState(0).choice(len(X6), 120000, replace=False)       # as cluster_12dim_check.py
    l12 = KMeans(K, n_init=10, random_state=0).fit(X12[idx]).predict(X12)
    W = C12.reshape(-1, 4, 3); W = W / np.maximum(W.sum(2, keepdims=True), 1e-12)   # within-stage shares
    dbp = X6[:, 1] / np.maximum(X6[:, :3].sum(1), 1e-12)
    corder = sorted(range(K), key=lambda c: np.median(dbp[l12 == c]))
    maj = {c: np.bincount(ref[l12 == c], minlength=K).argmax() for c in range(K)}

    fig = plt.figure(figsize=(3.4, 3.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.62, wspace=0.12)
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    bars(a, np.stack([W[ref == s].mean(0) for s in REF_ORDER]), [f"S{s}" for s in REF_ORDER], "(a) 6-dim states")
    bars(b, np.stack([W[l12 == q].mean(0) for q in corder]), [f"C{i}\n({'S%d' % maj[q]})" for i, q in enumerate(corder)],
         "(b) 12-dim clusters")
    a.set_ylabel("Share in stage (%)", fontweight="bold", labelpad=2); b.set_yticklabels([])
    for ax in (a, b):
        ax.text(0.0, -0.05, "bars: Dec, Disp, Iss, Com", transform=ax.transAxes, fontsize=5.5, va="top", color="#404040") if ax is a else None

    M = np.zeros((K, K))
    for i, q in enumerate(corder):
        h = np.bincount(ref[l12 == q], minlength=K); M[i] = 100 * h[REF_ORDER] / h.sum()
    im = c.imshow(M, cmap="Greys", vmin=0, vmax=100, aspect="auto")
    for i in range(K):
        for j in range(K):
            if M[i, j] >= 0.5:
                c.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] > 55 else "black")
    c.set_xticks(range(K)); c.set_xticklabels([f"S{s}" for s in REF_ORDER]); c.set_yticks(range(K))
    c.set_yticklabels([f"C{i} ({100*np.mean(l12 == q):.0f}%)" for i, q in enumerate(corder)], fontsize=6.5)
    c.set_xlabel("6-dim state", fontweight="bold", labelpad=2); c.set_ylabel("12-dim cluster", fontweight="bold", labelpad=2)
    c.set_title("(c) Designs of each 12-dim cluster by 6-dim state (%)", fontsize=7.5, fontweight="bold", pad=3)
    c.tick_params(length=0); S.frame(c)
    h = [Patch(facecolor=S.COL[k], edgecolor="black", linewidth=0.5, label=k) for k in CAT]
    fig.legend(handles=h, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=True, edgecolor="black",
               fancybox=False, framealpha=1, handlelength=1.1, columnspacing=0.8, borderpad=0.3).get_frame().set_linewidth(0.8)
    os.makedirs(FIG, exist_ok=True)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
