#!/usr/bin/env python3
"""Single-column summary over all 17 benchmarks, one panel per search method (Greedy, BO, GA, Random), style of
plot_balance_column.py (model only): per stage, Jain index of FP/BP/SP stall cycles vs search progress.
  Greedy  : small-start walks (results_small); per benchmark the median over its 1000 walks (plot_balance_stages).
  BO/GA/RW: incumbent = lowest-CPI design so far with area <= A_ref (A_ref from results_ga/<bench>/meta.json),
            x = evaluation / budget; per benchmark the median over its 10 runs (RW = 10 blocks of 512 designs),
            kept where >= 5 runs have a feasible incumbent (as plot_opt_jain_all_v2.py).
Line = median over benchmarks, drawn where >= 12 benchmarks have a value.
Writes results_rw/figs/fig_stage_jain_opt_column.{png,pdf}; refuses to overwrite."""
import os, sys, glob, json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_balance_stages import load, jain, median_curve, GX as GX_G, STAGES
import plot_balance_incumbent as PI
import plot_balance_incumbent_rw as PR

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
HERE = os.path.dirname(os.path.abspath(__file__)); MIN_RUNS, MIN_BENCH = 5, 12
warnings.simplefilter("ignore")


def per_bench(C):
    """C: (runs, 4, len(x)) -> per-stage median over runs where >= MIN_RUNS runs have a value."""
    med = np.nanmedian(C, 0); med[np.sum(~np.isnan(C), 0) < MIN_RUNS] = np.nan
    return med


def across(M):
    """M: (bench, 4, len(x)) -> median over benchmarks where >= MIN_BENCH benchmarks have a value."""
    med = np.nanmedian(M, 0); med[np.sum(~np.isnan(M), 0) < MIN_BENCH] = np.nan
    return med


def main():
    fd = os.path.join(HERE, "results_rw", "figs"); outs = [os.path.join(fd, f"fig_stage_jain_opt_column.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    B = sorted(d for d in os.listdir(os.path.join(HERE, "results_ga")) if d[0].isdigit())
    aref = {b: json.load(open(os.path.join(HERE, "results_ga", b, "meta.json")))["A_ref"] for b in B}
    curves = {}
    curves["Greedy"] = (GX_G, across(np.array([[median_curve(d["start"], jain(d, s)) for s in range(4)]
                                               for d in (load("results_small", b) for b in B)])))
    for opt, name in (("bo", "BO"), ("ga", "GA")):
        M = [per_bench(np.array([PI.incumbent_curves(f, aref[b]) for f in sorted(glob.glob(os.path.join(HERE, f"results_{opt}", b, "seed_*.npz")))]))
             for b in B]
        curves[name] = (PI.GX, across(np.array(M)))
    M = []
    for b in B:
        fs = sorted(glob.glob(os.path.join(HERE, "results_rw", b, "part_*.npz")))
        z = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("idx", "cpi", "area", "mscs")}
        o = np.argsort(z["idx"]); z = {k: v[o] for k, v in z.items()}
        M.append(per_bench(np.array([PR.curves(z["cpi"][sl], z["area"][sl], z["mscs"][sl], aref[b])
                                     for sl in (slice(r * PR.BLOCK, (r + 1) * PR.BLOCK) for r in range(PR.NB))])))
    curves["Random"] = (PI.GX, across(np.array(M)))

    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.6), sharex=True, sharey=True)
    print("first -> max -> end, per stage (Dec | Disp | Iss | Com)")
    for ax, name in zip(axes.ravel(), ("Greedy", "BO", "GA", "Random")):
        x, med = curves[name]; row = []
        for s in range(4):
            ax.plot(x, med[s], color=COL[s], ls=LS[s], lw=1.5 if s == 1 else 1.1, zorder=3)
            v = med[s][~np.isnan(med[s])]; row.append(f"{v[0]:.2f}->{v.max():.2f}->{v[-1]:.2f}")
        print(f"  {name:7s} " + " | ".join(row))
        ax.text(0.04, 0.93, name, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.set_ylim(1 / 3, 0.85); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([0.4, 0.6, 0.8])
        ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)
    h = [plt.Line2D([], [], color=COL[s], ls=LS[s], lw=1.5, label=STAGES[s]) for s in range(4)]
    fig.supylabel("Jain index", fontweight="bold", fontsize=8, x=0.045)
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
