#!/usr/bin/env python3
"""Single-column model-vs-gem5 balance figure for the four examples of fig_balance_column (nab, cactuBSSN,
deepsjeng, gcc), drawn as fig_jain_gem5_v2 (plot_balance_gem5_v2.py --start small, same data): Dispatch Jain
index along the representative small-start greedy walk on the model and on gem5 (gem5_val4/), with the median and
middle 50% of the 1000 small-start walks; style of plot_balance_column.py.
Optional argument <bench>:<name> replaces the third example (output suffix _<name>).
Writes results_small/figs/fig_balance_gem5_column[_<name>].{png,pdf}; refuses to overwrite."""
import os, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'figure.dpi': 600, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
})
MODEL, GEM5, MED = "#2E86C1", "#E67E22", "#7F7F7F"
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results_small"); V = os.path.join(HERE, "gem5_val4")
BENCH = [("644.nab_s", "nab"), ("607.cactuBSSN_s", "cactuBSSN"), ("631.deepsjeng_s", "deepsjeng"), ("602.gcc_s", "gcc")]
if len(sys.argv) > 1:                                   # e.g. 641.leela_s:leela -> replaces the third example
    BENCH[2] = tuple(sys.argv[1].split(":"))
GX = np.linspace(0, 1, 21)


def jain(cpi, m):
    x = np.stack([cpi * m[:, 4 + k] for k in (1, 2, 3)], 1)
    return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def main():
    fd = os.path.join(R, "figs"); suf = "" if len(sys.argv) == 1 else "_" + BENCH[2][1]
    outs = [os.path.join(fd, f"fig_balance_gem5_column{suf}.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    jobs = [j for pf in ("plan.json", "plan_peak.json") if os.path.exists(os.path.join(V, pf)) for j in json.load(open(os.path.join(V, pf)))]
    W = {}
    for j in jobs:
        p = os.path.join(V, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
        if os.path.exists(p):
            r = json.load(open(p)); W.setdefault(j["bench"], []).append((j["step"], j["walk_len"], j["model_cpi"], j["model_mscs"], r["cpi"], r["mscs16"]))
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.6), sharex=True, sharey=True)
    for ax, (b, name) in zip(axes.ravel(), BENCH):
        rows = sorted(W[b]); x = np.array([r[0] for r in rows]) / max(rows[0][1] - 1, 1)
        mv = jain(np.array([r[2] for r in rows]), np.array([r[3] for r in rows]))
        gv = jain(np.array([r[4] for r in rows]), np.array([r[5] for r in rows]))
        fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
        d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "cpi", "mscs")}
        v = jain(d["cpi"], d["mscs"]); cs = []
        for s in np.unique(d["start"]):
            h = v[d["start"] == s]; cs.append(np.interp(GX, np.linspace(0, 1, len(h)), h) if len(h) > 1 else np.full_like(GX, h[0]))
        cs = np.array(cs); q = np.percentile(cs, [25, 75], axis=0)
        ax.fill_between(GX, q[0], q[1], color=MED, alpha=0.2, lw=0, zorder=1)
        ax.plot(GX, np.median(cs, 0), color=MED, lw=1.1, zorder=2)
        ax.plot(x, mv, color=MODEL, lw=1.3, zorder=3)
        ax.plot(x, gv, color=GEM5, lw=1.1, ls="--", marker="o", ms=2.6, mfc=GEM5, mec="black", mew=0.4, zorder=4)
        print(f"{name:10s} model {mv[0]:.2f}->{mv.max():.2f}->{mv[-1]:.2f} | gem5 {gv[0]:.2f}->{gv.max():.2f}->{gv[-1]:.2f}")
        ax.text(0.04, 0.93, name, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.set_ylim(1 / 3, 1); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([0.4, 0.6, 0.8])
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
