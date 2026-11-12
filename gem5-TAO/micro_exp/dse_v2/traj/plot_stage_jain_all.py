#!/usr/bin/env python3
"""One panel, all 17 benchmarks: Jain index of FP/BP/SP stall cycles at each pipeline stage (Decode, Dispatch,
Issue, Commit) along the small-start greedy walks.
  lines  = model: per benchmark the median over its 1000 walks (resampled on walk progress), then the median
           over the 17 benchmarks; band = interquartile range over benchmarks.
  dots   = gem5 on the representative walk of each benchmark (gem5_val4, 12 states; peak states excluded so
           every walk is sampled the same way), resampled on progress, median over the 17 benchmarks.
Writes results_small/figs/fig_stage_jain_all_v2.png|pdf (v1 = earlier style) (never overwrites)."""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R, V = os.path.join(HERE, "results_small"), os.path.join(HERE, "gem5_val4")
FIG = os.path.join(R, "figs"); os.makedirs(FIG, exist_ok=True)
STAGES = ["Decode", "Dispatch", "Issue", "Commit"]
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e7e6e1", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": "#c9c8c1", "axes.linewidth": 0.8,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0,
                     "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})
GX = np.linspace(0, 1, 41)
GD = np.linspace(0, 1, 5)                       # where gem5 dots are drawn


def jain(cpi, m, s):
    x = np.stack([cpi * m[:, 4 * s + k] for k in (1, 2, 3)], 1); return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def main():
    B = sorted(d for d in os.listdir(R) if d[0].isdigit())
    model = np.zeros((4, len(B), len(GX)))
    for bi, b in enumerate(B):
        fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
        d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "cpi", "mscs")}
        st = d["start"]
        for s in range(4):
            v = jain(d["cpi"], d["mscs"], s); cs = []
            for w in np.unique(st):
                h = v[st == w]; cs.append(np.interp(GX, np.linspace(0, 1, len(h)), h) if len(h) > 1 else np.full_like(GX, h[0]))
            model[s, bi] = np.median(np.array(cs), 0)
    W = {}
    for j in json.load(open(os.path.join(V, "plan.json"))):
        p = os.path.join(V, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
        if os.path.exists(p):
            r = json.load(open(p)); W.setdefault(j["bench"], []).append((j["step"], j["walk_len"], r["cpi"], r["mscs16"]))
    gem = np.full((4, len(B), len(GD)), np.nan)
    for bi, b in enumerate(B):
        rows = sorted(W.get(b, []))
        if len(rows) < 2:
            continue
        x = np.array([r[0] for r in rows]) / max(rows[0][1] - 1, 1)
        c = np.array([r[2] for r in rows]); m = np.array([r[3] for r in rows])
        for s in range(4):
            gem[s, bi] = np.interp(GD, x, jain(c, m, s))
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for s in range(4):
        q25, q50, q75 = np.percentile(model[s], [25, 50, 75], axis=0)
        ax.fill_between(GX, q25, q75, color=COL[s], alpha=0.10, lw=0, zorder=1)
        ax.plot(GX, q50, color=COL[s], lw=2.6, solid_capstyle="round", zorder=3)
        g = np.nanmedian(gem[s], 0)
        ax.plot(GD, g, color=COL[s], lw=1.3, alpha=0.9, zorder=4)
        ax.plot(GD, g, ls="none", marker="o", ms=9, mfc=COL[s], mec=SURF, mew=1.8, zorder=5)
        ax.text(1.02, q50[-1], STAGES[s], color=INK, fontsize=14, va="center", ha="left", fontweight="bold")
        print(f"{STAGES[s]:8s} model {q50[0]:.3f} -> {q50[-1]:.3f} ({q50[-1]-q50[0]:+.3f}) | gem5 {g[0]:.3f} -> {g[-1]:.3f} ({g[-1]-g[0]:+.3f})")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(1 / 3 - 0.01, 0.8)
    ax.set_yticks([1 / 3, 0.5, 0.65, 0.8]); ax.set_yticklabels(["1/3", "0.5", "0.65", "0.8"])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["start", "", "", "", "end"])
    ax.set_xlabel("optimisation progress", fontsize=14); ax.set_ylabel("Jain index of FP / BP / SP stalls", fontsize=14)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], color=INK2, lw=2.6, label="model"),
         plt.Line2D([], [], color=INK2, lw=1.3, marker="o", ms=9, mec=SURF, mew=1.8, label="gem5"),
         plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.15, lw=0, label="model, middle 50% of benchmarks")]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=13, bbox_to_anchor=(0, 1.16), ncol=3, handlelength=1.8,
              columnspacing=1.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"fig_stage_jain_all_v2.{ext}")
        if os.path.exists(p):
            p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor=SURF); print("saved", p)


if __name__ == "__main__":
    main()
