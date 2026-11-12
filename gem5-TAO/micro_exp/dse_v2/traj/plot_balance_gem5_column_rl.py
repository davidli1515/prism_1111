#!/usr/bin/env python3
"""fig_balance_gem5_column_leela (nab, cactuBSSN, leela, gcc; same data and definitions as plot_balance_gem5_column.py,
imported read-only) with the legend right of the panels instead of below them and the y label next to the tick labels.
Writes <out-dir>/fig_balance_gem5_column_leela_rl.{png,pdf} (default results_small/figs); refuses to overwrite."""
import os, sys, glob, json, argparse
sys.argv, ARGV = sys.argv[:1], sys.argv[1:]             # plot_balance_gem5_column reads sys.argv at import
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plot_balance_gem5_column as P

BENCH = [("644.nab_s", "nab"), ("607.cactuBSSN_s", "cactuBSSN"), ("641.leela_s", "leela"), ("602.gcc_s", "gcc")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=os.path.join(P.R, "figs"))
    fd = ap.parse_args(ARGV).out_dir; os.makedirs(fd, exist_ok=True)
    outs = [os.path.join(fd, f"fig_balance_gem5_column_leela_rl.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    jobs = [j for pf in ("plan.json", "plan_peak.json") if os.path.exists(os.path.join(P.V, pf)) for j in json.load(open(os.path.join(P.V, pf)))]
    W = {}
    for j in jobs:
        p = os.path.join(P.V, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
        if os.path.exists(p):
            r = json.load(open(p)); W.setdefault(j["bench"], []).append((j["step"], j["walk_len"], j["model_cpi"], j["model_mscs"], r["cpi"], r["mscs16"]))
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 1.95), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.775, bottom=0.215, top=0.99, wspace=0.13, hspace=0.07)
    for ax, (b, name) in zip(axes.ravel(), BENCH):
        rows = sorted(W[b]); x = np.array([r[0] for r in rows]) / max(rows[0][1] - 1, 1)
        mv = P.jain(np.array([r[2] for r in rows]), np.array([r[3] for r in rows]))
        gv = P.jain(np.array([r[4] for r in rows]), np.array([r[5] for r in rows]))
        fs = sorted(glob.glob(os.path.join(P.R, b, "part_*.npz")))
        d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "cpi", "mscs")}
        v = P.jain(d["cpi"], d["mscs"]); cs = []
        for s in np.unique(d["start"]):
            h = v[d["start"] == s]; cs.append(np.interp(P.GX, np.linspace(0, 1, len(h)), h) if len(h) > 1 else np.full_like(P.GX, h[0]))
        cs = np.array(cs); q = np.percentile(cs, [25, 75], axis=0)
        ax.fill_between(P.GX, q[0], q[1], color=P.MED, alpha=0.2, lw=0, zorder=1)
        ax.plot(P.GX, np.median(cs, 0), color=P.MED, lw=1.1, zorder=2)
        ax.plot(x, mv, color=P.MODEL, lw=1.3, zorder=3)
        ax.plot(x, gv, color=P.GEM5, lw=1.1, ls="--", marker="o", ms=2.6, mfc=P.GEM5, mec="black", mew=0.4, zorder=4)
        print(f"{name:10s} model {mv[0]:.2f}->{mv.max():.2f}->{mv[-1]:.2f} | gem5 {gv[0]:.2f}->{gv.max():.2f}->{gv[-1]:.2f}")
        ly, lva = (0.05, "bottom") if name == "leela" else (0.93, "top")       # leela's curves fill the top-left corner
        ax.text(0.04, ly, name, transform=ax.transAxes, fontsize=8, fontweight="bold", va=lva)
        ax.set_ylim(1 / 3, 1); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([0.4, 0.6, 0.8])
        ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)
    h = [plt.Line2D([], [], color=P.GEM5, lw=1.1, ls="--", marker="o", ms=3, mfc=P.GEM5, mec="black", mew=0.4, label="gem5"),
         plt.Line2D([], [], color=P.MODEL, lw=1.3, label="Model"),
         plt.Line2D([], [], color=P.MED, lw=1.1, label="Median"),
         Patch(facecolor=P.MED, alpha=0.2, lw=0, label="25–75%")]
    a, b_, c = axes[1, 0].get_position(), axes[0, 0].get_position(), axes[1, 1].get_position()
    ym = (a.y0 + b_.y1) / 2
    fig.canvas.draw(); x0 = axes[0, 0].yaxis.get_tightbbox(fig.canvas.get_renderer()).x0 / fig.bbox.width
    fig.text(x0 - 0.008, ym, "Dispatch Jain index", rotation=90, ha="right", va="center", fontweight="bold", fontsize=8)
    fig.text((a.x0 + c.x1) / 2, a.y0 - 0.085, "Search progress (start = 0, end = 1)", ha="center", va="top", fontweight="bold", fontsize=8)
    fig.legend(handles=h, ncol=1, loc="center left", bbox_to_anchor=(c.x1 + 0.012, ym), frameon=True, edgecolor="black",
               fancybox=False, framealpha=1, handlelength=1.8, handletextpad=0.4, labelspacing=0.45, borderaxespad=0,
               borderpad=0.35).get_frame().set_linewidth(0.8)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
