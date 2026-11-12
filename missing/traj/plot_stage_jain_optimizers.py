#!/usr/bin/env python3
"""Appendix figure: Jain index of FP/BP/SP stalls at each stage (Decode, Dispatch, Issue, Commit), start vs end of
the optimisation, for four optimisers (model only), median over the 17 benchmarks.
  Greedy  : small-start walks (results_small); start = first state, end = last state; per benchmark the median
            over its 1000 walks.
  BO/GA/RW: incumbent = lowest-CPI design so far with area <= A_ref (A_ref = median greedy end area);
            start = first incumbent of the run, end = incumbent at the end of the run; per benchmark the median
            over 10 runs (RW = 10 blocks of 512 random designs).
Dumbbell: hollow = start, filled = end, segment = change. Writes results_rw/figs/fig_stage_jain_optimizers.png|pdf
(never overwrites)."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "results_rw", "figs"); os.makedirs(FIG, exist_ok=True)
STAGES = ["Decode", "Dispatch", "Issue", "Commit"]
OPT = ["Greedy", "BO", "GA", "Random"]
COL = {"Greedy": "#2a78d6", "BO": "#eb6834", "GA": "#1baf7a", "Random": "#eda100"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e7e6e1", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": "#c9c8c1", "axes.linewidth": 0.8,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0,
                     "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})


def jain4(cpi, m):
    out = []
    for s in range(4):
        x = np.stack([cpi * m[:, 4 * s + k] for k in (1, 2, 3)], 1); out.append(x.sum(1) ** 2 / (3 * (x ** 2).sum(1)))
    return np.stack(out, 1)                                         # n x 4


def cat(fs, keys):
    return {k: np.concatenate([np.load(f)[k] for f in fs]) for k in keys}


def first_last_incumbent(cpi, area, A):
    cur, first = -1, -1
    for i in range(len(cpi)):
        if area[i] <= A and (cur < 0 or cpi[i] < cpi[cur]):
            cur = i
            if first < 0:
                first = i
    return first, cur


def main():
    B = sorted(d for d in os.listdir(os.path.join(HERE, "results_rw")) if d[0].isdigit())
    S = {k: np.full((len(B), 4), np.nan) for k in OPT}; E = {k: np.full((len(B), 4), np.nan) for k in OPT}
    for bi, b in enumerate(B):
        g = cat(sorted(glob.glob(os.path.join(HERE, "results_small", b, "part_*.npz"))), ("start", "cpi", "mscs", "area"))
        last = np.r_[np.where(np.diff(g["start"]) != 0)[0], len(g["start"]) - 1]; first = np.r_[0, last[:-1] + 1]
        A = float(np.median(g["area"][last])); Jg = jain4(g["cpi"], g["mscs"])
        S["Greedy"][bi] = np.median(Jg[first], 0); E["Greedy"][bi] = np.median(Jg[last], 0)
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
                S[k][bi] = np.median(st, 0); E[k][bi] = np.median(en, 0)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    off = {k: (i - 1.5) * 0.17 for i, k in enumerate(OPT)}
    print(f"{'':8s} " + "  ".join(f"{k:>22s}" for k in OPT))
    for s in range(4):
        row = []
        for k in OPT:
            s0, e0 = np.nanmedian(S[k][:, s]), np.nanmedian(E[k][:, s]); x = s + off[k]
            ax.plot([x, x], [s0, e0], color=COL[k], lw=2.6, solid_capstyle="round", zorder=2)
            ax.plot(x, s0, "o", ms=9, mfc=SURF, mec=COL[k], mew=2.2, zorder=3)
            ax.plot(x, e0, "o", ms=9, mfc=COL[k], mec=SURF, mew=1.6, zorder=4)
            nb = np.sum(np.isfinite(S[k][:, s]))
            row.append(f"{s0:.2f}->{e0:.2f} ({e0-s0:+.3f}, n={nb})")
        print(f"{STAGES[s]:8s} " + "  ".join(f"{r:>22s}" for r in row))
    ax.set_xticks(range(4)); ax.set_xticklabels(STAGES, fontsize=14)
    ax.set_xlim(-0.55, 3.55); ax.set_ylim(1 / 3 - 0.01, 0.8)
    ax.set_yticks([1 / 3, 0.5, 0.65, 0.8]); ax.set_yticklabels(["1/3", "0.5", "0.65", "0.8"])
    ax.set_ylabel("Jain index of FP / BP / SP stalls", fontsize=14)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], color=COL[k], lw=2.6, label=k) for k in OPT] + \
        [plt.Line2D([], [], ls="none", marker="o", ms=9, mfc=SURF, mec=INK2, mew=2.2, label="start"),
         plt.Line2D([], [], ls="none", marker="o", ms=9, mfc=INK2, mec=SURF, mew=1.6, label="end")]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=12, ncol=6, bbox_to_anchor=(0, 1.14),
              handlelength=1.4, columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"fig_stage_jain_optimizers.{ext}")
        if os.path.exists(p):
            p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor=SURF); print("saved", p)


if __name__ == "__main__":
    main()
