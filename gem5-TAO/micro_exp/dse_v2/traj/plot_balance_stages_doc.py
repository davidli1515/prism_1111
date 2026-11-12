#!/usr/bin/env python3
"""Doc version of fig_balance_stages (small starts), in the style of window_ids/plot_unseen_simpoint.py.
Same data and metric as plot_balance_stages.py: per benchmark, median over 1000 greedy walks of the Jain index
of FP/BP/SP stall cycles at each stage, vs normalised walk progress.
Writes results_small/figs/fig_balance_stages_doc_v2.{png,pdf}; refuses to overwrite."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_balance_stages import load, jain, median_curve, short, GX, STAGES

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 10,
    'figure.dpi': 600, 'axes.linewidth': 1.0,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
})
COL = ["#2E86C1", "#E67E22", "#27AE60", "#8E44AD"]
LS = ["-", "-", "--", "-."]
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "results_small"


def main():
    fd = os.path.join(HERE, ROOT, "figs"); outs = [os.path.join(fd, f"fig_balance_stages_doc_v2.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    B = sorted(d for d in os.listdir(os.path.join(HERE, ROOT)) if d[0].isdigit())
    fig, axes = plt.subplots(3, 6, figsize=(12, 6), sharex=True, sharey=True)
    for ax, b in zip(axes.ravel(), B):
        d = load(ROOT, b)
        for s in range(4):
            ax.plot(GX, median_curve(d["start"], jain(d, s)), color=COL[s], ls=LS[s], lw=1.8 if s == 1 else 1.3, zorder=3)
        ax.text(0.04, 0.93, short(b), transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        ax.set_ylim(1 / 3, 1); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"]); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.0)
    last = axes.ravel()[len(B)]
    for ax in axes.ravel()[len(B):]:
        ax.axis("off")
    h = [plt.Line2D([], [], color=COL[s], ls=LS[s], lw=1.8, label=STAGES[s]) for s in range(4)]
    last.legend(handles=h, loc="center", frameon=True, edgecolor="black", fancybox=False).get_frame().set_linewidth(1.0)
    fig.supxlabel("Walk progress (start = 0, end = 1)", fontweight="bold", fontsize=11)
    fig.supylabel("Jain index of FP/BP/SP stall cycles", fontweight="bold", fontsize=11)
    fig.tight_layout(h_pad=0.4, w_pad=1.2)
    for p in outs:
        fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
