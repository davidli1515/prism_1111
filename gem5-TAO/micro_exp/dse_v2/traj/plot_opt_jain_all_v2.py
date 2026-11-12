#!/usr/bin/env python3
"""One-panel summary over all 17 benchmarks (v2: greedy added, area-matched baseline per design).
y = Dispatch-stall Jain of the current design minus the Jain of TYPICAL designs of the same area
    (median Jain of the 500 random designs (results_rw, 100k) closest in log-area to that design).
x = optimisation progress, 0 = start, 1 = end of the run (efficiency is not compared).
  Greedy  : the current design of each small-start walk (results_small), x = step / walk length.
  BO/GA/RW: the incumbent = lowest-CPI design so far with area <= A_ref (A_ref = median greedy end area),
            x = evaluation / budget (BO 500, GA 512, RW = 10 blocks of 512 random designs).
Per benchmark: median over walks / runs (>= 5 runs with a value); line = median over benchmarks,
band = interquartile range over benchmarks (drawn where >= 12 benchmarks have a value).
Writes results_rw/figs/fig_opt_jain_all_v2.png|pdf (never overwrites)."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "results_rw", "figs"); os.makedirs(FIG, exist_ok=True)
COL = {"Greedy": "#2a78d6", "BO": "#eb6834", "GA": "#1baf7a", "Random": "#eda100"}
INK, INK2, GRID, SURF, ZERO = "#0b0b0b", "#52514e", "#e7e6e1", "#ffffff", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c9c8c1", "axes.linewidth": 0.8,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0,
                     "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})
GX = np.linspace(0, 1, 41)
K_NEAR = 500


def jain(cpi, m):
    x = np.stack([cpi * m[:, 4 + k] for k in (1, 2, 3)], 1); return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def cat(fs, keys):
    return {k: np.concatenate([np.load(f)[k] for f in fs]) for k in keys}


class Baseline:
    """median Jain of the K_NEAR random designs closest in log-area"""
    def __init__(self, area, J):
        o = np.argsort(area); self.la = np.log(area[o]); self.J = J[o]

    def __call__(self, a):
        a = np.atleast_1d(a); out = np.empty(len(a))
        for i, x in enumerate(np.log(a)):
            c = np.searchsorted(self.la, x); lo = max(0, c - K_NEAR); hi = min(len(self.la), c + K_NEAR)
            w = np.arange(lo, hi); w = w[np.argsort(np.abs(self.la[w] - x))[:K_NEAR]]
            out[i] = np.median(self.J[w])
        return out


def incumbent_idx(cpi, area, A):
    idx = np.full(len(cpi), -1); cur = -1
    for i in range(len(cpi)):
        if area[i] <= A and (cur < 0 or cpi[i] < cpi[cur]):
            cur = i
        idx[i] = cur
    return idx


def main():
    B = sorted(d for d in os.listdir(os.path.join(HERE, "results_rw")) if d[0].isdigit())
    curves = {k: [] for k in COL}
    for b in B:
        rw = cat(sorted(glob.glob(os.path.join(HERE, "results_rw", b, "part_*.npz"))), ("cpi", "mscs", "area"))
        base = Baseline(rw["area"], jain(rw["cpi"], rw["mscs"]))
        g = cat(sorted(glob.glob(os.path.join(HERE, "results_small", b, "part_*.npz"))), ("start", "cpi", "mscs", "area"))
        last = np.r_[np.where(np.diff(g["start"]) != 0)[0], len(g["start"]) - 1]; A = float(np.median(g["area"][last]))
        # greedy: every state of every walk
        ua, inv = np.unique(np.round(g["area"], 3), return_inverse=True)            # baseline once per distinct area
        dg = jain(g["cpi"], g["mscs"]) - base(ua)[inv]; st = g["start"]; cs = []
        for s in np.unique(st):
            h = dg[st == s]; cs.append(np.interp(GX, np.linspace(0, 1, len(h)), h) if len(h) > 1 else np.full_like(GX, h[0]))
        curves["Greedy"].append(np.median(np.array(cs), 0))
        runs = {"BO": [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "results_bo", b, "seed_*.npz")))],
                "GA": [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "results_ga", b, "seed_*.npz")))],
                "Random": [{k: rw[k][s * 512:(s + 1) * 512] for k in rw} for s in range(10)]}
        for k, rs in runs.items():
            C = []
            for z in rs:
                n = len(z["cpi"]); ii = incumbent_idx(z["cpi"], z["area"], A); t = np.clip((GX * (n - 1)).round().astype(int), 0, n - 1)
                d = jain(z["cpi"], z["mscs"]) - base(z["area"]); C.append(np.where(ii[t] >= 0, d[ii[t]], np.nan))
            C = np.array(C); n_ok = np.sum(np.isfinite(C), 0)
            with np.errstate(all="ignore"):
                curves[k].append(np.where(n_ok >= 5, np.nanmedian(C, 0), np.nan))
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.axhline(0, color=ZERO, lw=1.2, zorder=1)
    ax.text(0.99, -0.004, "typical design of the same area", fontsize=9.5, color=INK2, va="top", ha="right")
    for k, col in COL.items():
        M = np.array(curves[k]); nb = np.sum(np.isfinite(M), 0)
        with np.errstate(all="ignore"):
            q25, q50, q75 = np.nanpercentile(M, [25, 50, 75], axis=0)
        ok = nb >= 12
        ax.fill_between(GX[ok], q25[ok], q75[ok], color=col, alpha=0.10, lw=0, zorder=2)
        ax.plot(GX[ok], q50[ok], color=col, lw=2.4 if k == "Greedy" else 2.0, solid_capstyle="round", zorder=3, label=k)
        e = M[:, ok][:, -1]; s0 = M[:, ok][:, 0]
        print(f"{k:7s} start {np.nanmedian(s0):+.3f} -> end {np.nanmedian(e):+.3f} (IQR {np.nanpercentile(e,25):+.3f}..{np.nanpercentile(e,75):+.3f}); "
              f"end > 0 in {np.sum(e > 0)}/{np.sum(np.isfinite(e))} benchmarks; first plotted x = {GX[ok][0]:.2f}")
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["start", "", "", "", "end"])
    ax.set_xlabel("optimisation progress"); ax.set_ylabel("Δ Jain vs typical design\nof the same area")
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5, ncol=4, handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"fig_opt_jain_all_v2.{ext}")
        if os.path.exists(p):
            p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor=SURF); print("saved", p)


if __name__ == "__main__":
    main()
