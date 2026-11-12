#!/usr/bin/env python3
"""Variant of plot_opt_traj_gem5.py (same data, same style): the gem5 and Model lines are the MEAN over benchmarks
of the representative trajectories instead of the median (a median over 17 benchmarks jumps when a different
benchmark becomes the middle one). A missing gem5 point (failed run) is linearly interpolated within its own
benchmark trajectory, so every progress point averages the same benchmarks. Median line and 25-75% band unchanged.
Writes <out-dir>/fig_opt_traj_gem5_mean.{png,pdf} (default results_rw/figs); refuses to overwrite."""
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plot_opt_traj_gem5 import population, representative, MODEL, GEM5, MED, HERE, OPT, GX, GK


def fill(row):
    ok = np.isfinite(row)
    return np.interp(GK, GK[ok], row[ok]) if ok.sum() >= 2 else np.full_like(row, np.nan)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=os.path.join(HERE, "results_rw", "figs"))
    fd = ap.parse_args().out_dir; os.makedirs(fd, exist_ok=True)
    outs = [os.path.join(fd, f"fig_opt_traj_gem5_mean.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    POP, REP = population(), representative()
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.6), sharex=True, sharey=True)
    for ax, k in zip(axes.ravel(), OPT):
        P = POP[k]; q = np.percentile(P, [25, 75], axis=0)
        ax.fill_between(GX, q[0], q[1], color=MED, alpha=0.2, lw=0, zorder=1)
        ax.plot(GX, np.median(P, 0), color=MED, lw=1.1, zorder=2)
        Mr, Gr = REP[k]
        Gf = np.array([fill(np.asarray(r, float)) for r in Gr]); keep = np.isfinite(Gf).all(1)
        mv, gv = Mr[keep].mean(0), Gf[keep].mean(0)
        ax.plot(GK, mv, color=MODEL, lw=1.3, zorder=3)
        ax.plot(GK, gv, color=GEM5, lw=1.1, ls="--", marker="o", ms=2.6, mfc=GEM5, mec="black", mew=0.4, zorder=4)
        print(f"{k:7s} median {np.median(P,0)[0]:.2f}->{np.median(P,0)[-1]:.2f} | model mean {mv[0]:.2f}->{mv[-1]:.2f} | "
              f"gem5 mean {gv[0]:.2f}->{gv[-1]:.2f}  (benchmarks: pop {len(P)}, rep {keep.sum()}, "
              f"interpolated gem5 points {int(np.sum(~np.isfinite(np.asarray(Gr, float))))})")
        ax.text(0.04, 0.93, k, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.set_ylim(0.4, 0.9); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([0.5, 0.6, 0.7, 0.8])
        ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)
    h = [plt.Line2D([], [], color=GEM5, lw=1.1, ls="--", marker="o", ms=3, mfc=GEM5, mec="black", mew=0.4, label="gem5"),
         plt.Line2D([], [], color=MODEL, lw=1.3, label="Model"),
         plt.Line2D([], [], color=MED, lw=1.1, label="Median"),
         Patch(facecolor=MED, alpha=0.2, lw=0, label="25–75%")]
    fig.supylabel("Dispatch Jain index", fontweight="bold", fontsize=8, x=0.045)
    fig.tight_layout(h_pad=0.4, w_pad=1.4, rect=(0.03, 0.17, 1, 1))
    a, c = axes[1, 0].get_position(), axes[1, 1].get_position(); xc = (a.x0 + c.x1) / 2
    fig.text(xc, a.y0 - 0.09, "Search progress (start = 0, end = 1)", ha="center", va="top", fontweight="bold", fontsize=8)
    fig.legend(handles=h, ncol=4, loc="upper center", bbox_to_anchor=(xc, a.y0 - 0.15), frameon=True, edgecolor="black",
               fancybox=False, framealpha=1, columnspacing=0.8, handlelength=1.8, handletextpad=0.4,
               borderpad=0.3).get_frame().set_linewidth(0.8)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
