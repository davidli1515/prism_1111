#!/usr/bin/env python3
"""fig_states split into three separate figures, in the plotting style of window_ids/plot_unseen_simpoint.py
(reads results/state_clusters_v1.npz and results/<bench>/part_*.npz; writes results/figs/, never overwrites):
  fig_state_centroids    stall composition of the 6 bottleneck states
  fig_state_start_end    share of walks in each state at start and end, per benchmark
  fig_state_transitions  pooled transition probabilities between states along walks"""
import os, glob, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 600, 'axes.linewidth': 1.0,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 4, 'ytick.major.size': 4,
})
COL = {"FP": "#2E86C1", "BP": "#E67E22", "SP": "#27AE60"}
COL_LIGHT = {"FP": "#AED6F1", "BP": "#F5CBA7", "SP": "#A9DFBF"}

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
FIG = os.path.join(R, "figs")
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}


def save(fig, name):
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"{name}.{ext}")
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)
    plt.close(fig)


def frame(ax):
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("black"); s.set_linewidth(1.0)


def short(b):
    return b.split(".")[1].replace("_s", "").replace("_r", "") + ("*" if b in FP_SUITE else "")


def main():
    Z = np.load(os.path.join(R, "state_clusters_v1.npz"))
    cent, k = Z["centroids"], int(Z["k"])
    B = sorted(d for d in os.listdir(R) if d[0].isdigit())
    names = [f"S{s}" for s in range(k)]

    # ---------------- centroids
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    comps = [("D", "FP"), ("D", "BP"), ("D", "SP"), ("C", "FP"), ("C", "BP"), ("C", "SP")]
    for s in range(k):
        bottom = 0
        for j, (stg, cat) in enumerate(comps):
            v = 100 * cent[s, j]
            ax.bar(s, v, bottom=bottom, width=0.65, zorder=3, edgecolor="black", linewidth=0.5,
                   color=COL[cat] if stg == "D" else COL_LIGHT[cat], hatch="///" if stg == "C" else None)
            bottom += v
    ax.set_xticks(range(k)); ax.set_xticklabels(names, fontweight="bold")
    ax.set_ylabel("Share of stall cycles (%)", fontweight="bold"); ax.set_ylim(0, 100); ax.set_xlim(-0.6, k - 0.4)
    ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0); frame(ax)
    h = [Patch(facecolor=COL[c], edgecolor="black", linewidth=0.5, label=f"Dispatch {c}") for c in ("FP", "BP", "SP")] + \
        [Patch(facecolor=COL_LIGHT[c], edgecolor="black", linewidth=0.5, hatch="///", label=f"Commit {c}") for c in ("FP", "BP", "SP")]
    ax.legend(handles=h, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=True, edgecolor="black",
              fancybox=False, columnspacing=1.0, handlelength=1.4).get_frame().set_linewidth(1.0)
    save(fig, "fig_state_centroids")

    # ---------------- start / end shares
    rows, labels = [], []
    for b in B:
        fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
        st = np.concatenate([np.load(f)["start"] for f in fs]); lab = Z[f"labels_{b}"]
        first = np.r_[0, np.where(np.diff(st) != 0)[0] + 1]; last = np.r_[first[1:] - 1, len(st) - 1]
        rows += [np.bincount(lab[first], minlength=k) / len(first), np.bincount(lab[last], minlength=k) / len(last)]
        labels += [f"{short(b)} start", f"{short(b)} end"]
    M = 100 * np.array(rows)
    fig, ax = plt.subplots(figsize=(4.4, 8.2))
    ax.imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=100)
    ax.set_xticks(range(k)); ax.set_xticklabels(names, fontweight="bold")
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5)
    for i in range(2, len(labels), 2):
        ax.axhline(i - 0.5, color="black", lw=0.8)
    for i in range(M.shape[0]):
        for j in range(k):
            if M[i, j] >= 10:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 55 else "black")
    ax.tick_params(length=0); frame(ax)
    ax.set_xlabel("Share of walks in each state (%)", fontweight="bold", labelpad=8)
    save(fig, "fig_state_start_end")

    # ---------------- transitions
    T = np.zeros((k, k))
    for b in B:
        fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
        st = np.concatenate([np.load(f)["start"] for f in fs]); lab = Z[f"labels_{b}"]
        same = st[1:] == st[:-1]; a, c = lab[:-1][same], lab[1:][same]; ch = a != c
        np.add.at(T, (a[ch], c[ch]), 1)
    P = 100 * T / np.maximum(T.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(P, cmap="Greys", vmin=0, vmax=100)
    for i in range(k):
        for j in range(k):
            if T[i, j] > 0:
                ax.text(j, i, f"{P[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if P[i, j] > 55 else "black")
    ax.set_xticks(range(k)); ax.set_xticklabels(names, fontweight="bold")
    ax.set_yticks(range(k)); ax.set_yticklabels(names, fontweight="bold")
    ax.set_xlabel("To state", fontweight="bold"); ax.set_ylabel("From state", fontweight="bold")
    ax.tick_params(length=0); frame(ax)
    save(fig, "fig_state_transitions")


if __name__ == "__main__":
    main()
