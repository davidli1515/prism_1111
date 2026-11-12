#!/usr/bin/env python3
"""fig_stage_jain_opt_cpi_column_rl4 (same data; compute() imported from plot_stage_jain_opt_cpi_column_rl4.py,
read-only) with the legend of fig_balance_gem5_column_leela_rl5: one column, line and text side by side, no separator
lines, light-grey frame.
Writes <out-dir>/fig_stage_jain_opt_cpi_column_rl5.{png,pdf} (default results_rw/figs); refuses to overwrite."""
import os, sys, glob, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_stage_jain_opt_cpi_column as O
from plot_stage_jain_opt_cpi_column_rl4 import compute
from plot_stage_jain_opt_cpi_column import S, PI, PR, load, jain, median_curve, GX_G, STAGES, inc_cpi, thresh, CPI_COL

HERE = O.HERE
SHORT = ["Decode", "Dispatch", "Issue", "Commit"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=os.path.join(HERE, "results_rw", "figs"))
    fd = ap.parse_args().out_dir; os.makedirs(fd, exist_ok=True)
    outs = [os.path.join(fd, f"fig_stage_jain_opt_cpi_column_rl5.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    curves = compute()
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 1.95), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.67, bottom=0.215, top=0.99, wspace=0.15, hspace=0.07)
    print("normalised CPI first -> end; Dispatch Jain first -> end"); right = []
    for i, (ax, name) in enumerate(zip(axes.ravel(), ("Greedy", "BO", "GA", "Random"))):
        x, med, cpi = curves[name]
        for s in range(4):
            ax.plot(x, med[s], color=S.COL[s], ls=S.LS[s], lw=1.5 if s == 1 else 1.1, zorder=3)
        ax2 = ax.twinx(); right.append(ax2)
        ax2.plot(x, cpi, color=CPI_COL, ls=":", lw=1.3, zorder=4)
        ax2.set_ylim(0.55, 1.12); ax2.set_yticks([0.6, 0.8, 1.0]); ax2.set_yticklabels(["0.6", "0.8", "1"])
        ax2.tick_params(axis="y", direction="in", labelright=(i % 2 == 1), length=2.5, width=0.8)
        v = cpi[~np.isnan(cpi)]; dj = med[1][~np.isnan(med[1])]
        print(f"  {name:7s} CPI {v[0]:.2f}->{v[-1]:.2f} | Disp Jain {dj[0]:.2f}->{dj[-1]:.2f}")
        ax.text(0.04, 0.98, name, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.set_ylim(1 / 3, 0.9); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([0.4, 0.6, 0.8])
        ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
        for a_ in (ax, ax2):
            for sp in a_.spines.values():
                sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)
    a, b_, c = axes[1, 0].get_position(), axes[0, 0].get_position(), axes[1, 1].get_position()
    ym = (a.y0 + b_.y1) / 2
    fig.canvas.draw(); rd = fig.canvas.get_renderer(); x0 = axes[0, 0].yaxis.get_tightbbox(rd).x0 / fig.bbox.width
    fig.text(x0 - 0.008, ym, "Jain index", rotation=90, ha="right", va="center", fontweight="bold", fontsize=8)
    xr = right[1].yaxis.get_tightbbox(rd).x1 / fig.bbox.width
    t = fig.text(xr + 0.006, ym, "Normalized CPI", rotation=270, ha="left", va="center", fontweight="bold", fontsize=8)
    fig.text((a.x0 + c.x1) / 2, a.y0 - 0.085, "Search progress (start = 0, end = 1)", ha="center", va="top", fontweight="bold", fontsize=8)
    h = [plt.Line2D([], [], color=S.COL[s], ls=S.LS[s], lw=1.5 if s == 1 else 1.1, label=SHORT[s]) for s in range(4)]
    h += [plt.Line2D([], [], color=CPI_COL, ls=":", lw=1.3, label="CPI")]
    lg = fig.legend(handles=h, ncol=1, loc="center left", bbox_to_anchor=(t.get_window_extent(rd).x1 / fig.bbox.width + 0.012, ym),
                    frameon=True, edgecolor="#BFBFBF", fancybox=False, framealpha=1, fontsize=7.5, handlelength=1.6,
                    handletextpad=0.4, labelspacing=1.25, borderpad=0.5, borderaxespad=0)
    lg.get_frame().set_linewidth(0.5)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
