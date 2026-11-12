#!/usr/bin/env python3
"""One-panel summary over all 17 benchmarks: Dispatch-stall Jain of the incumbent design along BO, GA and
random-walker (RW) runs, relative to typical designs of the same area.
Per benchmark: A_ref = median end area of the greedy small-start walks; incumbent at evaluation t = the
lowest-CPI design evaluated so far with area <= A_ref (same rule for all three optimisers); value = its Jain
minus the median Jain of all 100k random designs with area within A_ref +-5% ("typical design of that area").
Per benchmark the median over 10 runs (BO / GA seeds; RW = 10 blocks of 512 designs; needs >= 5 runs with a
feasible design); line = median over benchmarks, band = interquartile range over benchmarks.
Right-hand marker: greedy end designs (median over benchmarks of the per-benchmark median).
Writes results_rw/figs/fig_opt_jain_all.png|pdf (never overwrites)."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "results_rw", "figs"); os.makedirs(FIG, exist_ok=True)
COL = {"BO": "#2a78d6", "GA": "#eb6834", "Random": "#1baf7a"}
INK, INK2, GRID, SURF, GREY = "#0b0b0b", "#52514e", "#e7e6e1", "#ffffff", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c9c8c1", "axes.linewidth": 0.8,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0,
                     "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})
TS = np.arange(4, 512, 4)


def jain(cpi, m):
    x = np.stack([cpi * m[:, 4 + k] for k in (1, 2, 3)], 1); return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def cat(fs, keys):
    return {k: np.concatenate([np.load(f)[k] for f in fs]) for k in keys}


def incumbent_curve(cpi, area, J, A):
    """Jain of the best-CPI design with area <= A among the first t evaluations, for t in TS (NaN if none)."""
    ok = area <= A; c = np.where(ok, cpi, np.inf); best = np.minimum.accumulate(c)
    idx = np.zeros(len(c), int); cur = -1
    for i in range(len(c)):
        if ok[i] and (cur < 0 or c[i] < c[cur]):
            cur = i
        idx[i] = cur
    return np.array([J[idx[t - 1]] if t - 1 < len(c) and idx[t - 1] >= 0 else np.nan for t in TS])


def main():
    B = sorted(d for d in os.listdir(os.path.join(HERE, "results_rw")) if d[0].isdigit())
    curves = {k: [] for k in COL}; greedy = []
    for b in B:
        g = cat(sorted(glob.glob(os.path.join(HERE, "results_small", b, "part_*.npz"))), ("start", "cpi", "mscs", "area"))
        last = np.r_[np.where(np.diff(g["start"]) != 0)[0], len(g["start"]) - 1]; A = float(np.median(g["area"][last]))
        rw = cat(sorted(glob.glob(os.path.join(HERE, "results_rw", b, "part_*.npz"))), ("cpi", "mscs", "area"))
        win = np.abs(rw["area"] / A - 1) <= 0.05; base = np.median(jain(rw["cpi"][win], rw["mscs"][win]))
        greedy.append(np.median(jain(g["cpi"][last], g["mscs"][last])) - base)
        runs = {"BO": [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "results_bo", b, "seed_*.npz")))],
                "GA": [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "results_ga", b, "seed_*.npz")))],
                "Random": [{k: rw[k][s * 512:(s + 1) * 512] for k in rw} for s in range(10)]}
        for k, rs in runs.items():
            C = np.array([incumbent_curve(z["cpi"], z["area"], jain(z["cpi"], z["mscs"]), A) for z in rs]) - base
            n = np.sum(np.isfinite(C), 0)
            with np.errstate(all="ignore"):
                curves[k].append(np.where(n >= 5, np.nanmedian(C, 0), np.nan))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0, color=GREY, lw=1.2, zorder=1)
    ax.text(8, 0.004, "typical design of the same area", fontsize=9.5, color=INK2, va="bottom")
    ends = []
    for k, col in COL.items():
        M = np.array(curves[k]); nb = np.sum(np.isfinite(M), 0)
        with np.errstate(all="ignore"):
            q25, q50, q75 = np.nanpercentile(M, [25, 50, 75], axis=0)
        ok = nb >= 12                                          # plot only where >= 12 of 17 benchmarks have a value
        ax.fill_between(TS[ok], q25[ok], q75[ok], color=col, alpha=0.12, lw=0, zorder=2)
        ax.plot(TS[ok], q50[ok], color=col, lw=2.2, solid_capstyle="round", zorder=3, label=k)
        ends.append((q50[ok][-1], k, col))
        print(f"{k:7s} final median {q50[ok][-1]:+.3f} (IQR {q25[ok][-1]:+.3f}..{q75[ok][-1]:+.3f}); "
              f"benchmarks > 0 at the end: {np.sum(M[:, ok][:, -1] > 0)}/{np.sum(np.isfinite(M[:, ok][:, -1]))}")
    gq = np.percentile(greedy, [25, 50, 75])
    ax.errorbar([530], [gq[1]], yerr=[[gq[1] - gq[0]], [gq[2] - gq[1]]], fmt="D", ms=7, color=INK, mec=SURF, mew=1.2,
                elinewidth=1.4, capsize=0, zorder=5, label="Greedy (end)")
    print(f"greedy  median {gq[1]:+.3f} (IQR {gq[0]:+.3f}..{gq[2]:+.3f}); benchmarks > 0: {np.sum(np.array(greedy) > 0)}/{len(greedy)}")
    ax.set_xlim(0, 550); ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_xlabel("evaluations"); ax.set_ylabel("Δ Jain of best design\n(vs typical design, same area)")
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5, ncol=2, handlelength=1.8, columnspacing=1.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"fig_opt_jain_all.{ext}")
        if os.path.exists(p):
            p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor=SURF); print("saved", p)


if __name__ == "__main__":
    main()
