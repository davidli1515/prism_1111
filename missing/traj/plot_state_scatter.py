#!/usr/bin/env python3
"""Scatter view of the Pesto-style bottleneck states (results/figs/fig_state_scatter.*, never overwritten).
Points = trajectory states (random subsample of 60k of 540k), feature = the clustering feature (6-dim stall
composition). (a) PCA of the feature (centroids projected with the same transform); (b) two raw axes:
Dispatch-BP share vs Dispatch-SP share (Dispatch-FP = the rest of Dispatch); (c) Dispatch-BP vs Commit-FP
share (separates S5). Centroids are direct-labelled. Prints silhouette (sample) and per-state spread."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
FIG = os.path.join(R, "figs")
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})


def main():
    Z = np.load(os.path.join(R, "state_clusters_v1.npz"))
    cent, k = Z["centroids"], int(Z["k"])
    B = sorted(d for d in os.listdir(R) if d[0].isdigit())
    X, L = [], []
    for b in B:
        fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
        cpi = np.concatenate([np.load(f)["cpi"] for f in fs]); m = np.concatenate([np.load(f)["mscs"] for f in fs])
        F = np.stack([cpi * m[:, 4 * s + kk] for s in (1, 3) for kk in (1, 2, 3)], 1)
        X.append(F / F.sum(1, keepdims=True)); L.append(Z[f"labels_{b}"])
    X, L = np.concatenate(X), np.concatenate(L)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), 60000, replace=False); Xs, Ls = X[idx], L[idx]
    sidx = rng.choice(len(X), 15000, replace=False)
    sil = silhouette_samples(X[sidx], L[sidx])
    print(f"silhouette (15k sample): overall {sil.mean():.2f}; per state " +
          ", ".join(f"S{s} {sil[L[sidx]==s].mean():.2f}" for s in range(k)))
    for s in range(k):
        d = np.linalg.norm(X[L == s] - cent[s], axis=1)
        print(f"  S{s}: n={np.sum(L==s):6d} ({np.mean(L==s)*100:4.1f}%)  median distance to centroid {np.median(d):.3f}")
    pca = PCA(2).fit(X[sidx]); P, C = pca.transform(Xs), pca.transform(cent)
    ev = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    views = [(P, C, f"PC1 ({100*ev[0]:.0f}% var.)", f"PC2 ({100*ev[1]:.0f}% var.)", "(a) PCA of the 6-dim stall composition"),
             (Xs[:, [1, 2]], cent[:, [1, 2]], "Dispatch BP share", "Dispatch SP share", "(b) Dispatch: back-pressure vs speculation"),
             (Xs[:, [1, 3]], cent[:, [1, 3]], "Dispatch BP share", "Commit FP share", "(c) Dispatch back-pressure vs Commit starvation")]
    order = np.argsort(rng.rand(len(Xs)))
    for ax, (pts, cp, xl, yl, title) in zip(axes, views):
        ax.scatter(pts[order, 0], pts[order, 1], c=[PAL[i] for i in Ls[order]], s=1.5, alpha=0.18, lw=0, rasterized=True)
        for s in range(k):
            ax.scatter(cp[s, 0], cp[s, 1], s=90, c=PAL[s], edgecolor="white", linewidth=2, zorder=5)
            ax.annotate(f"S{s}", (cp[s, 0], cp[s, 1]), xytext=(6, 6), textcoords="offset points", fontsize=10,
                        fontweight="bold", color=INK, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(color=GRID, lw=0.6)
    handles = [plt.Line2D([], [], marker="o", ls="", color=PAL[s], label=f"S{s} ({np.mean(L==s)*100:.0f}% of states)")
               for s in range(k)]
    fig.legend(handles=handles, loc="lower center", ncol=k, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Bottleneck states of 540k trajectory states (60k shown); silhouette {sil.mean():.2f}",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"fig_state_scatter.{ext}")
        if os.path.exists(p):
            p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
