#!/usr/bin/env python3
"""Appendix single-column figure, style of plot_balance_gem5_column.py: per-stage Jain index of FP/BP/SP stalls,
start vs end of the optimisation, one panel per optimiser (Greedy, BO, GA, Random). In each panel and stage three
dumbbells (hollow = start, filled = end), median over benchmarks:
  gem5   : the representative design pair simulated in gem5 (Greedy: gem5_val4 walk, first/last state;
           BO/GA/Random: gem5_val5 run, start/end incumbent)
  Model  : the model on the same designs
  Median : the model on all runs (per benchmark the median over its walks / runs, as plot_stage_jain_optimizers.py)
Writes results_rw/figs/fig_stage_jain_optimizers_gem5.{png,pdf}; refuses to overwrite."""
import os, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_stage_jain_optimizers import jain4, cat, first_last_incumbent
import gem5_val5 as G5

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
STAGES = ["Dec", "Disp", "Iss", "Com"]
SRC = [("gem5", GEM5, -0.24), ("Model", MODEL, 0.0), ("Median", MED, 0.24)]


def jain1(c, m):
    return jain4(np.array([c]), np.array([m]))[0]


def representative():
    """-> {opt: {bench: (model_start, model_end, gem5_start, gem5_end)}} (4-vectors over stages)"""
    out = {k: {} for k in OPT}
    V4 = os.path.join(HERE, "gem5_val4"); W = {}
    for j in json.load(open(os.path.join(V4, "plan.json"))):
        p = os.path.join(V4, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
        if os.path.exists(p):
            r = json.load(open(p)); W.setdefault(j["bench"], []).append((j["step"], j["model_cpi"], j["model_mscs"], r["cpi"], r["mscs16"]))
    for b, rows in W.items():
        rows = sorted(rows); a, z = rows[0], rows[-1]
        out["Greedy"][b] = (jain1(a[1], a[2]), jain1(z[1], z[2]), jain1(a[3], a[4]), jain1(z[3], z[4]))
    jp = os.path.join(G5.OUT, "plan.json")
    jobs = json.load(open(jp)) if os.path.exists(jp) else []
    for opt in ("BO", "GA", "Random"):
        for b in sorted({j["bench"] for j in jobs if j["opt"] == opt}):
            st = next(j for j in jobs if j["bench"] == b and j["opt"] == opt and j["which"] == "start")
            en = next(j for j in jobs if j["bench"] == b and j["opt"] == opt and j["which"] == "end")
            rs, re_ = G5.gem5_of(st, jobs), G5.gem5_of(en, jobs)
            if rs is not None and re_ is not None:
                out[opt][b] = (jain1(st["model_cpi"], st["model_mscs"]), jain1(en["model_cpi"], en["model_mscs"]),
                               jain1(rs["cpi"], rs["mscs16"]), jain1(re_["cpi"], re_["mscs16"]))
    return out


def population():
    """-> {opt: (start 4-vector, end 4-vector)}: per benchmark median over walks / runs, then median over benchmarks"""
    B = sorted(d for d in os.listdir(os.path.join(HERE, "results_rw")) if d[0].isdigit())
    S = {k: [] for k in OPT}; E = {k: [] for k in OPT}
    for b in B:
        g = cat(sorted(glob.glob(os.path.join(HERE, "results_small", b, "part_*.npz"))), ("start", "cpi", "mscs", "area"))
        last = np.r_[np.where(np.diff(g["start"]) != 0)[0], len(g["start"]) - 1]; first = np.r_[0, last[:-1] + 1]
        A = float(np.median(g["area"][last])); Jg = jain4(g["cpi"], g["mscs"])
        S["Greedy"].append(np.median(Jg[first], 0)); E["Greedy"].append(np.median(Jg[last], 0))
        rw = cat(sorted(glob.glob(os.path.join(HERE, "results_rw", b, "part_*.npz"))), ("cpi", "mscs", "area"))
        runs = {"BO": [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "results_bo", b, "seed_*.npz")))],
                "GA": [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "results_ga", b, "seed_*.npz")))],
                "Random": [{k: rw[k][s * 512:(s + 1) * 512] for k in rw} for s in range(10)]}
        for k, rs in runs.items():
            st, en = [], []
            for z in rs:
                f, l = first_last_incumbent(z["cpi"], z["area"], A)
                if f >= 0:
                    J = jain4(z["cpi"][[f, l]], z["mscs"][[f, l]]); st.append(J[0]); en.append(J[1])
            if len(st) >= 5:
                S[k].append(np.median(st, 0)); E[k].append(np.median(en, 0))
    return {k: (np.median(S[k], 0), np.median(E[k], 0)) for k in OPT}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=os.path.join(HERE, "results_rw", "figs"))
    fd = ap.parse_args().out_dir; os.makedirs(fd, exist_ok=True)
    outs = [os.path.join(fd, f"fig_stage_jain_optimizers_gem5.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    REP, POP = representative(), population()
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.8), sharex=True, sharey=True)
    for ax, k in zip(axes.ravel(), OPT):
        V = np.array(list(REP[k].values())) if REP[k] else None
        vals = {"Median": POP[k]}
        if V is not None:
            vals["Model"] = (np.median(V[:, 0], 0), np.median(V[:, 1], 0))
            vals["gem5"] = (np.median(V[:, 2], 0), np.median(V[:, 3], 0))
        for name, col, dx in SRC:
            if name not in vals:
                continue
            s0, s1 = vals[name]
            for s in range(4):
                x = s + dx
                ax.plot([x, x], [s0[s], s1[s]], color=col, lw=1.2, ls="--" if name == "gem5" else "-", zorder=2)
                ax.plot(x, s0[s], "o", ms=3.0, mfc="white", mec=col, mew=0.7, zorder=3)
                ax.plot(x, s1[s], "o", ms=3.0, mfc=col, mec="black" if name == "gem5" else col, mew=0.4, zorder=4)
            print(f"{k:7s} {name:6s} " + "  ".join(f"{STAGES[s]} {s0[s]:.2f}->{s1[s]:.2f}" for s in range(4))
                  + (f"  (n={len(V)})" if name != "Median" and V is not None else ""))
        ax.text(0.04, 0.93, k, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.set_xlim(-0.5, 3.5); ax.set_xticks(range(4)); ax.set_xticklabels(STAGES)
        ax.set_ylim(1 / 3, 0.85); ax.set_yticks([0.4, 0.6, 0.8])
        ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)
    h = [plt.Line2D([], [], color=GEM5, lw=1.1, ls="--", marker="o", ms=3, mfc=GEM5, mec="black", mew=0.4, label="gem5"),
         plt.Line2D([], [], color=MODEL, lw=1.3, marker="o", ms=3, mfc=MODEL, mec=MODEL, label="Model"),
         plt.Line2D([], [], color=MED, lw=1.1, marker="o", ms=3, mfc=MED, mec=MED, label="Median"),
         plt.Line2D([], [], ls="none", marker="o", ms=3.4, mfc="white", mec="black", mew=0.6, label="start"),
         plt.Line2D([], [], ls="none", marker="o", ms=3.4, mfc="black", mec="black", label="end")]
    fig.supylabel("Jain index", fontweight="bold", fontsize=8, x=0.045)
    fig.tight_layout(h_pad=0.4, w_pad=1.4, rect=(0.03, 0.15, 1, 1))
    a, c = axes[1, 0].get_position(), axes[1, 1].get_position(); xc = (a.x0 + c.x1) / 2
    fig.text(xc, a.y0 - 0.07, "Pipeline stage", ha="center", va="top", fontweight="bold", fontsize=8)
    fig.legend(handles=h, ncol=5, loc="upper center", bbox_to_anchor=(xc, a.y0 - 0.13), frameon=True, edgecolor="black",
               fancybox=False, framealpha=1, columnspacing=0.7, handlelength=1.6, handletextpad=0.3,
               borderpad=0.3).get_frame().set_linewidth(0.8)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
