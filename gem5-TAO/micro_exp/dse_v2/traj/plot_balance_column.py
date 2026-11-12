#!/usr/bin/env python3
"""Single-column version of fig_balance_stages_doc_v2 (same data, metric and style) for four representative
benchmarks: per stage, median over the 1000 small-start greedy walks of the Jain index of FP/BP/SP stall cycles
vs normalised walk progress. Writes results_small/figs/fig_balance_column.{png,pdf}; refuses to overwrite."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_balance_stages import load, jain, median_curve, GX, STAGES

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'figure.dpi': 600, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
})
COL = ["#2E86C1", "#E67E22", "#27AE60", "#8E44AD"]
LS = ["-", "-", "--", "-."]
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "results_small"
BENCH = [("644.nab_s", "nab"), ("607.cactuBSSN_s", "cactuBSSN"), ("631.deepsjeng_s", "deepsjeng"), ("602.gcc_s", "gcc")]


def main():
    fd = os.path.join(HERE, ROOT, "figs"); outs = [os.path.join(fd, f"fig_balance_column.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.6), sharex=True, sharey=True)
    for ax, (b, name) in zip(axes.ravel(), BENCH):
        d = load(ROOT, b)
        for s in range(4):
            ax.plot(GX, median_curve(d["start"], jain(d, s)), color=COL[s], ls=LS[s], lw=1.5 if s == 1 else 1.1, zorder=3)
        ax.text(0.04, 0.93, name, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.set_ylim(1 / 3, 1); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([0.4, 0.6, 0.8])
        ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)
    h = [plt.Line2D([], [], color=COL[s], ls=LS[s], lw=1.5, label=STAGES[s]) for s in range(4)]
    fig.supylabel("Jain index", fontweight="bold", fontsize=8, x=0.045)
    fig.tight_layout(h_pad=0.4, w_pad=1.4, rect=(0.03, 0.17, 1, 1))
    a, b = axes[1, 0].get_position(), axes[1, 1].get_position(); xc = (a.x0 + b.x1) / 2
    fig.text(xc, a.y0 - 0.09, "Search progress (start = 0, end = 1)", ha="center", va="top", fontweight="bold", fontsize=8)
    fig.legend(handles=h, ncol=4, loc="upper center", bbox_to_anchor=(xc, a.y0 - 0.15), frameon=True, edgecolor="black",
               fancybox=False, framealpha=1, columnspacing=0.8, handlelength=1.8, handletextpad=0.4,
               borderpad=0.3).get_frame().set_linewidth(0.8)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
