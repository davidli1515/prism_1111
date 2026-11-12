#!/usr/bin/env python3
"""Figures from the trajectory data (writes results/figs/*.png|pdf, never overwrites):
  fig_states:  (a) centroid composition of the 6 bottleneck states, (b) share of each benchmark's states in
               each state (all / start / end), (c) pooled transition probabilities between states
  fig_entropy: Dispatch-stage stall entropy along walks, per benchmark: median and IQR over 1000 walks vs
               normalised walk progress, plus a few individual walks."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
FIG = os.path.join(R, "figs"); os.makedirs(FIG, exist_ok=True)
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
COL = {"FP": "#2a78d6", "BP": "#eb6834", "SP": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})


def save(fig, name):
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"{name}.{ext}")
        if os.path.exists(p):
            p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        print("saved", p)


def load(tag):
    fs = sorted(glob.glob(os.path.join(R, tag, "part_*.npz")))
    return {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "cpi", "mscs", "kind")}


def short(b):
    return b.split(".")[1].replace("_s", "").replace("_r", "") + ("*" if b in FP_SUITE else "")


def main():
    Z = np.load(os.path.join(R, "state_clusters_v1.npz"))
    cent, k = Z["centroids"], int(Z["k"])
    B = sorted(d for d in os.listdir(R) if d[0].isdigit())
    D = {b: load(b) for b in B}
    L = {b: Z[f"labels_{b}"] for b in B}

    # ------------------------------------------------------------ fig_states
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.7, 1.0], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    comps = [("D", "FP"), ("D", "BP"), ("D", "SP"), ("C", "FP"), ("C", "BP"), ("C", "SP")]
    for s in range(k):
        bottom = 0
        for j, (stg, cat) in enumerate(comps):
            v = 100 * cent[s, j]
            ax.bar(s, v, bottom=bottom, color=COL[cat], width=0.72, edgecolor="white", linewidth=1.0,
                   hatch="////" if stg == "C" else None, alpha=1.0 if stg == "D" else 0.55)
            bottom += v
    ax.set_xticks(range(k)); ax.set_xticklabels([f"S{s}" for s in range(k)])
    ax.set_ylabel("share of stall cycles (%)"); ax.set_ylim(0, 100)
    ax.set_title("(a) bottleneck states (centroids)", loc="left", fontsize=10, color=INK)
    leg = [Patch(facecolor=COL[c], label=f"Dispatch {c}") for c in ("FP", "BP", "SP")] + \
          [Patch(facecolor=COL[c], alpha=0.55, hatch="////", edgecolor="white", label=f"Commit {c}") for c in ("FP", "BP", "SP")]
    ax.legend(handles=leg, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    ax = fig.add_subplot(gs[1])
    rows, labels = [], []
    for b in B:
        st = D[b]["start"]; first = np.r_[0, np.where(np.diff(st) != 0)[0] + 1]; last = np.r_[first[1:] - 1, len(st) - 1]
        rows += [np.bincount(L[b][first], minlength=k) / len(first), np.bincount(L[b][last], minlength=k) / len(last)]
        labels += [f"{short(b)}  start", "end"]
    M = np.array(rows)
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k)); ax.set_xticklabels([f"S{s}" for s in range(k)])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6.5)
    for i in range(0, len(labels), 2):
        ax.axhline(i - 0.5, color="white", lw=1.5)
    for i in range(M.shape[0]):
        for j in range(k):
            if M[i, j] >= 0.15:
                ax.text(j, i, f"{100*M[i,j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.6 else INK)
    ax.set_title("(b) share of walks in each state, at start and end (* = SPECfp)", loc="left", fontsize=10, color=INK)
    ax.tick_params(length=0); [s_.set_visible(False) for s_ in ax.spines.values()]

    ax = fig.add_subplot(gs[2])
    T = np.zeros((k, k))
    for b in B:
        st, lab = D[b]["start"], L[b]
        same = st[1:] == st[:-1]; a, c = lab[:-1][same], lab[1:][same]; ch = a != c
        np.add.at(T, (a[ch], c[ch]), 1)
    Tp = T / np.maximum(T.sum(1, keepdims=True), 1)
    ax.imshow(Tp, cmap="Oranges", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if T[i, j] > 0:
                ax.text(j, i, f"{100*Tp[i,j]:.0f}\n({int(T[i,j])})", ha="center", va="center", fontsize=6,
                        color="white" if Tp[i, j] > 0.6 else INK)
    ax.set_xticks(range(k)); ax.set_xticklabels([f"S{s}" for s in range(k)]); ax.set_xlabel("to")
    ax.set_yticks(range(k)); ax.set_yticklabels([f"S{s}" for s in range(k)]); ax.set_ylabel("from")
    ax.set_title("(c) transitions: % of exits from a state\n(count)", loc="left", fontsize=10, color=INK)
    ax.tick_params(length=0); [s_.set_visible(False) for s_ in ax.spines.values()]
    save(fig, "fig_states")

    # ------------------------------------------------------------ fig_entropy
    grid = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(3, 6, figsize=(14, 7), sharex=True, sharey=True)
    rng = np.random.RandomState(1)
    for ax, b in zip(axes.ravel(), B):
        d = D[b]; st = d["start"]
        c = np.stack([d["cpi"] * d["mscs"][:, 4 + kk] for kk in (1, 2, 3)], 1)      # Dispatch FP,BP,SP cycles
        p = c / np.maximum(c.sum(1, keepdims=True), 1e-12)
        H = -(p * np.log(p + 1e-12)).sum(1) / np.log(3)
        curves = []
        for s in np.unique(st):
            h = H[st == s]
            x = np.linspace(0, 1, len(h)) if len(h) > 1 else np.array([0.0])
            curves.append(np.interp(grid, x, h) if len(h) > 1 else np.full_like(grid, h[0]))
        curves = np.array(curves)
        for i in rng.choice(len(curves), 12, replace=False):
            ax.plot(grid, curves[i], color=MUTED, lw=0.5, alpha=0.35)
        q25, q50, q75 = np.percentile(curves, [25, 50, 75], axis=0)
        ax.fill_between(grid, q25, q75, color=COL["FP"], alpha=0.2, lw=0)
        ax.plot(grid, q50, color=COL["FP"], lw=2)
        ax.set_title(short(b), fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_ylim(0, 1)
    for ax in axes.ravel()[len(B):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("walk progress (start = 0, end = 1)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Dispatch stall entropy")
    fig.suptitle("Balance of Dispatch-stage stalls (FP / BP / SP) along greedy walks: median and IQR over 1000 walks, "
                 "grey = 12 sample walks  (* = SPECfp)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig_entropy")


if __name__ == "__main__":
    main()
