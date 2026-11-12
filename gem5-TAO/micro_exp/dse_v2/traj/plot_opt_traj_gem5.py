#!/usr/bin/env python3
"""Per-optimiser Dispatch-Jain trajectories over all 17 benchmarks, model vs gem5, style of
plot_balance_gem5_column.py (one panel per optimiser: Greedy, BO, GA, Random). x = optimisation progress
(Greedy: step / walk length of the small-start walks; BO/GA/Random: evaluations from the first incumbent to the end
of the run, incumbent = lowest-CPI design so far with area <= A_ref).
  gem5   : representative walk / run per benchmark (gem5_val4 for Greedy, gem5_val6 for BO/GA/Random), 12 progress
           points, median over benchmarks
  Model  : the model on the same designs, median over benchmarks
  Median : the model on all walks / runs (per benchmark the median trajectory), median over benchmarks
  25-75% : interquartile range over benchmarks of the per-benchmark median trajectory
Writes <out-dir>/fig_opt_traj_gem5.{png,pdf} (default results_rw/figs); refuses to overwrite."""
import os, sys, glob, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plot_stage_jain_optimizers import jain4, cat
import gem5_val6 as G6

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'figure.dpi': 600, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
})
MODEL, GEM5, MED = "#2E86C1", "#E67E22", "#7F7F7F"
HERE = os.path.dirname(os.path.abspath(__file__))
OPT = ["Greedy", "BO", "GA", "Random"]
GX, GK = np.linspace(0, 1, 21), np.linspace(0, 1, G6.K)


def jd(cpi, m):
    return jain4(np.atleast_1d(cpi), np.atleast_2d(m))[:, 1]


def population():
    """-> {opt: benchmarks x 21} per-benchmark median Dispatch-Jain trajectory (model, all walks / runs)"""
    out = {k: [] for k in OPT}
    for b in sorted(d for d in os.listdir(os.path.join(HERE, "results_rw")) if d[0].isdigit()):
        g = cat(sorted(glob.glob(os.path.join(HERE, "results_small", b, "part_*.npz"))), ("start", "cpi", "mscs", "area"))
        v = jd(g["cpi"], g["mscs"]); cs = []
        for s in np.unique(g["start"]):
            h = v[g["start"] == s]; cs.append(np.interp(GX, np.linspace(0, 1, len(h)), h) if len(h) > 1 else np.full_like(GX, h[0]))
        out["Greedy"].append(np.median(cs, 0))
        A = G6.a_ref(b)
        rw = cat(sorted(glob.glob(os.path.join(HERE, "results_rw", b, "part_*.npz"))), ("cfg", "cpi", "mscs", "area"))
        for k, rs in G6.runs_of(b, rw).items():
            cs = []
            for z in rs:
                idx = G6.incumbent_index(z["cpi"], z["area"], A)
                if (idx >= 0).any():
                    cs.append(jd(z["cpi"], z["mscs"])[G6.at_progress(idx, GX)])
            if len(cs) >= 5:
                out[k].append(np.median(cs, 0))
    return {k: np.array(v) for k, v in out.items()}


def representative():
    """-> {opt: (model benchmarks x K, gem5 benchmarks x K)}; gem5 rows are NaN where a point is missing"""
    out = {}
    V4 = os.path.join(HERE, "gem5_val4"); W = {}
    for j in json.load(open(os.path.join(V4, "plan.json"))):
        p = os.path.join(V4, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
        if os.path.exists(p):
            r = json.load(open(p)); W.setdefault(j["bench"], []).append((j["step"], j["walk_len"], j["model_cpi"], j["model_mscs"], r["cpi"], r["mscs16"]))
    M, G = [], []
    for b, rows in sorted(W.items()):
        rows = sorted(rows); x = np.array([r[0] for r in rows]) / max(rows[0][1] - 1, 1)
        M.append(np.interp(GK, x, jd(np.array([r[2] for r in rows]), np.array([r[3] for r in rows]))))
        G.append(np.interp(GK, x, jd(np.array([r[4] for r in rows]), np.array([r[5] for r in rows]))))
    out["Greedy"] = (np.array(M), np.array(G))
    jp = os.path.join(G6.OUT, "plan.json")
    jobs = json.load(open(jp)) if os.path.exists(jp) else G6.plan()
    for opt in ("BO", "GA", "Random"):
        M, G = [], []
        for b in sorted({j["bench"] for j in jobs if j["opt"] == opt}):
            js = sorted([j for j in jobs if j["opt"] == opt and j["bench"] == b], key=lambda j: j["progress"])
            M.append([jd(j["model_cpi"], j["model_mscs"])[0] for j in js])
            gr = []
            for j in js:
                p = G6.result_path(G6.OUT, j)
                gr.append(jd(*(lambda r: (r["cpi"], r["mscs16"]))(json.load(open(p))))[0] if os.path.exists(p) else np.nan)
            G.append(gr)
        out[opt] = (np.array(M), np.array(G))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=os.path.join(HERE, "results_rw", "figs"))
    fd = ap.parse_args().out_dir; os.makedirs(fd, exist_ok=True)
    outs = [os.path.join(fd, f"fig_opt_traj_gem5.{e}") for e in ("png", "pdf")]
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
        mv = np.median(Mr, 0); ax.plot(GK, mv, color=MODEL, lw=1.3, zorder=3)
        ok = np.sum(np.isfinite(Gr), 0) >= max(3, len(Gr) // 2)
        if ok.any():
            with np.errstate(all="ignore"):
                gv = np.nanmedian(Gr, 0)
            ax.plot(GK[ok], gv[ok], color=GEM5, lw=1.1, ls="--", marker="o", ms=2.6, mfc=GEM5, mec="black", mew=0.4, zorder=4)
            print(f"{k:7s} median {np.median(P,0)[0]:.2f}->{np.median(P,0)[-1]:.2f} | model {mv[0]:.2f}->{mv[-1]:.2f} | "
                  f"gem5 {gv[ok][0]:.2f}->{gv[ok][-1]:.2f}  (benchmarks: pop {len(P)}, rep {len(Mr)})")
        else:
            print(f"{k:7s} median {np.median(P,0)[0]:.2f}->{np.median(P,0)[-1]:.2f} | model {mv[0]:.2f}->{mv[-1]:.2f} | gem5 n/a")
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
