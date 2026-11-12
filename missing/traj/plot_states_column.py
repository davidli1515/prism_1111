#!/usr/bin/env python3
"""Single-column versions of fig_state_centroids and fig_state_pca (same data, transform and style as
plot_states_split.py and plot_state_pca.py), sized to sit side by side in one column (0.49\\columnwidth each)
with larger text. Writes results/figs/fig_state_centroids_col.{png,pdf} and fig_state_pca_col.{png,pdf};
refuses to overwrite."""
import os, glob, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'figure.dpi': 600, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
})
COL = {"FP": "#2E86C1", "BP": "#E67E22", "SP": "#27AE60"}
COL_LIGHT = {"FP": "#AED6F1", "BP": "#F5CBA7", "SP": "#A9DFBF"}
PAL = ["#2E86C1", "#E67E22", "#27AE60", "#C0392B", "#8E44AD", "#7F8C8D"]
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results"); FIG = os.path.join(R, "figs")
SIZE = (1.75, 1.45)                                              # axes-area figure size; legend added below


def frame(ax):
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("black"); s.set_linewidth(0.8)


def save(fig, name):
    outs = [os.path.join(FIG, f"{name}.{e}") for e in ("png", "pdf")]
    for p in outs:
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)
    plt.close(fig)


def legend(ax, h, ncol, y=-0.27):
    ax.legend(handles=h, ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, y), frameon=True, edgecolor="black",
              fancybox=False, framealpha=1, columnspacing=0.6, handlelength=1.1, handletextpad=0.3, borderpad=0.3,
              labelspacing=0.25).get_frame().set_linewidth(0.8)


def main():
    names = ("fig_state_centroids_col", "fig_state_pca_col")
    for n in names:
        for e in ("png", "pdf"):
            if os.path.exists(os.path.join(FIG, f"{n}.{e}")):
                sys.exit(f"exists, refusing to overwrite: {n}.{e}")
    Z = np.load(os.path.join(R, "state_clusters_v1.npz")); cent, k = Z["centroids"], int(Z["k"])
    S = [f"S{s}" for s in range(k)]

    # ---------------- (a) centroids
    fig, ax = plt.subplots(figsize=SIZE)
    comps = [("D", "FP"), ("D", "BP"), ("D", "SP"), ("C", "FP"), ("C", "BP"), ("C", "SP")]
    for s in range(k):
        bottom = 0
        for j, (stg, cat) in enumerate(comps):
            v = 100 * cent[s, j]
            ax.bar(s, v, bottom=bottom, width=0.65, zorder=3, edgecolor="black", linewidth=0.4,
                   color=COL[cat] if stg == "D" else COL_LIGHT[cat], hatch="///" if stg == "C" else None)
            bottom += v
    ax.set_xticks(range(k)); ax.set_xticklabels(S, fontweight="bold")
    ax.set_xlabel("Bottleneck state", fontweight="bold", labelpad=2)
    ax.set_ylabel("Stall share (%)", fontweight="bold", labelpad=2); ax.set_ylim(0, 100); ax.set_xlim(-0.6, k - 0.4)
    ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0); frame(ax)
    plt.rcParams['hatch.linewidth'] = 0.5
    h = [Patch(facecolor=COL[c], edgecolor="black", linewidth=0.4, label=f"Disp. {c}") for c in ("FP", "BP", "SP")] + \
        [Patch(facecolor=COL_LIGHT[c], edgecolor="black", linewidth=0.4, hatch="///", label=f"Com. {c}") for c in ("FP", "BP", "SP")]
    legend(ax, h, 2)
    save(fig, names[0])

    # ---------------- (b) PCA (same sampling as plot_state_pca.py)
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
    fig, ax = plt.subplots(figsize=SIZE)
    ax.scatter(P[order, 0], P[order, 1], c=[PAL[i] for i in Ls[order]], s=0.6, alpha=0.25, lw=0, rasterized=True, zorder=2)
    for s in range(k):
        ax.scatter(C[s, 0], C[s, 1], s=22, marker="o", c=PAL[s], edgecolor="black", linewidth=0.7, zorder=4)
        ax.annotate(S[s], (C[s, 0], C[s, 1]), xytext=(3, 3), textcoords="offset points", fontsize=7.5,
                    fontweight="bold", zorder=5)
    ax.set_xlabel(f"PC1 ({100*ev[0]:.0f}%)", fontweight="bold", labelpad=2)
    ax.set_ylabel(f"PC2 ({100*ev[1]:.0f}%)", fontweight="bold", labelpad=2)
    ax.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0); frame(ax)
    h = [plt.Line2D([], [], marker="o", ls="", markersize=4, markerfacecolor=PAL[s], markeredgecolor="black",
                    markeredgewidth=0.6, label=f"{S[s]} ({100*np.mean(L==s):.0f}%)") for s in range(k)]
    legend(ax, h, 2)
    save(fig, names[1])
    print("PC variance", ev)


if __name__ == "__main__":
    main()
