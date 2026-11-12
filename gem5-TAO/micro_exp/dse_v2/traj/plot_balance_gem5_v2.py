#!/usr/bin/env python3
"""Redesigned model-vs-gem5 balance figure (one representative walk per benchmark).
  python plot_balance_gem5_v2.py --start small|random
Panels sorted by the model's change along the walk (largest first); each panel: IQR band of the 1000 walks
(context), the representative walk on the model (blue) and on gem5 (orange), and the start->end change of
both. Gray = median / middle 50% of the 1000 walks; legend in the 18th slot. Writes <results>/figs/fig_{entropy,jain}_gem5_v2.png|pdf
(never overwrites)."""
import os, glob, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
MODEL, GEM5, BAND, MED = "#2a78d6", "#eb6834", "#8a8983", "#8a8983"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8983", "#e7e6e1", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9c8c1", "axes.linewidth": 0.8,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0,
                     "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})
GX = np.linspace(0, 1, 21)


def x3(cpi, m):
    return np.stack([cpi * m[:, 4 + k] for k in (1, 2, 3)], 1)


def entropy(x):
    p = x / x.sum(1, keepdims=True); return -(p * np.log(p + 1e-12)).sum(1) / np.log(3)


def jain(x):
    return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def short(b):
    return b.split(".")[1].replace("_s", "").replace("_r", "") + ("*" if b in FP_SUITE else "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", choices=["small", "random"], default="small")
    args = ap.parse_args()
    R = os.path.join(HERE, "results_small" if args.start == "small" else "results")
    V = os.path.join(HERE, "gem5_val4" if args.start == "small" else "gem5_val3")
    FIG = os.path.join(R, "figs"); os.makedirs(FIG, exist_ok=True)
    jobs = [j for pf in ("plan.json", "plan_peak.json") if os.path.exists(os.path.join(V, pf)) for j in json.load(open(os.path.join(V, pf)))]
    W = {}
    for j in jobs:
        p = os.path.join(V, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
        if os.path.exists(p):
            r = json.load(open(p)); W.setdefault(j["bench"], []).append((j["step"], j["walk_len"], j["model_cpi"], j["model_mscs"], r["cpi"], r["mscs16"]))
    B = sorted(W)
    for metric, fn, ylo, ylab in (("entropy", entropy, 0.0, "Dispatch stall entropy"), ("jain", jain, 1 / 3, "Dispatch stall Jain index")):
        P = {}
        for b in B:
            rows = sorted(W[b]); x = np.array([r[0] for r in rows]) / max(rows[0][1] - 1, 1)
            mv = fn(x3(np.array([r[2] for r in rows]), np.array([r[3] for r in rows])))
            gv = fn(x3(np.array([r[4] for r in rows]), np.array([r[5] for r in rows])))
            fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
            d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "cpi", "mscs")}
            v = fn(x3(d["cpi"], d["mscs"])); st = d["start"]; cs = []
            for s in np.unique(st):
                h = v[st == s]; cs.append(np.interp(GX, np.linspace(0, 1, len(h)), h) if len(h) > 1 else np.full_like(GX, h[0]))
            P[b] = dict(x=x, mv=mv, gv=gv, q=np.percentile(np.array(cs), [25, 75], axis=0), med=np.median(np.array(cs), 0))
        order = sorted(B, key=lambda b: -(P[b]["mv"][-1] - P[b]["mv"][0]))
        fig, axes = plt.subplots(3, 6, figsize=(14, 7.2), sharex=False, sharey=False)
        for ax, b in zip(axes.ravel(), order):
            q = P[b]
            ax.fill_between(GX, q["q"][0], q["q"][1], color=BAND, alpha=0.16, lw=0)
            ax.plot(GX, q["med"], color=MED, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(q["x"], q["mv"], color=MODEL, lw=2, solid_capstyle="round", zorder=3)
            ax.plot(q["x"], q["gv"], color=GEM5, lw=2, solid_capstyle="round", zorder=4)
            ax.plot(q["x"], q["gv"], ls="none", marker="o", ms=4.5, mfc=GEM5, mec=SURF, mew=1.2, zorder=5)
            dm, dg = q["mv"][-1] - q["mv"][0], q["gv"][-1] - q["gv"][0]
            ax.set_title(short(b), fontsize=12, color=INK, loc="left", pad=3, fontweight="bold")
            yy = lambda v: (np.interp(GX, q["x"], v) - ylo) / (1 - ylo)                       # data in axes coords
            pts = np.r_[np.c_[GX, yy(q["mv"])], np.c_[GX, yy(q["gv"])]]
            corners = [(0.98, 0.04, "right", "bottom"), (0.98, 0.96, "right", "top"), (0.02, 0.04, "left", "bottom"), (0.02, 0.96, "left", "top")]
            hit = lambda cx, cy: np.sum((np.abs(pts[:, 0] - (cx - 0.2 if cx > 0.5 else cx + 0.2)) < 0.24) & (np.abs(pts[:, 1] - (cy - 0.07 if cy > 0.5 else cy + 0.07)) < 0.11))
            cx, cy, ha, va = min(corners, key=lambda c: hit(c[0], c[1]))
            ax.text(cx, cy, f"Δ model {dm:+.2f}\nΔ gem5 {dg:+.2f}", transform=ax.transAxes, ha=ha, va=va, fontsize=8.5,
                    color=INK2, linespacing=1.3, bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.85), zorder=6)
            ax.set_ylim(ylo, 1.0); ax.set_xlim(-0.03, 1.03)
            ax.set_yticks([ylo, (ylo + 1) / 2, 1.0]); ax.set_yticklabels([f"{ylo:.2g}", f"{(ylo+1)/2:.2g}", "1"])
            ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["start", "", "end"])
            ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylab, fontsize=10.5)
        axes.ravel()[-1].axis("off")
        h = [plt.Line2D([], [], color=GEM5, lw=2.5, marker="o", ms=6, mfc=GEM5, mec=SURF, mew=1.2, label="gem5"),
             plt.Line2D([], [], color=MODEL, lw=2.5, label="model"),
             plt.Line2D([], [], color=MED, lw=2.5, label="model, median of\n1000 walks"),
             plt.Rectangle((0, 0), 1, 1, color=BAND, alpha=0.25, lw=0, label="model, middle 50%\nof 1000 walks")]
        axes.ravel()[-1].legend(handles=h, loc="center left", frameon=False, fontsize=12, handlelength=2.4, labelspacing=0.9,
                                title=("small-design starts" if args.start == "small" else "random starts") + "\n* = SPECfp",
                                title_fontsize=11, alignment="left")
        fig.tight_layout(h_pad=1.6, w_pad=1.0)
        for ext in ("png", "pdf"):
            p = os.path.join(FIG, f"fig_{metric}_gem5_v2.{ext}")
            if os.path.exists(p):
                p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
            fig.savefig(p, bbox_inches="tight", facecolor=SURF); print("saved", p)


if __name__ == "__main__":
    main()
