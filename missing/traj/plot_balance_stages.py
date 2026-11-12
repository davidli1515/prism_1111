#!/usr/bin/env python3
"""Jain balance index of FP/BP/SP stall cycles at EACH pipeline stage (Decode, Dispatch, Issue, Commit) along
greedy walks: per benchmark, the median over 1000 walks of each stage, on one panel.
Writes <results_dir>/figs/fig_balance_stages.png|pdf for small starts (results_small) and random starts
(results); never overwrites. Prints start/end values per stage."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
STAGES = ["Decode", "Dispatch", "Issue", "Commit"]
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})
GX = np.linspace(0, 1, 21)


def load(root, b):
    fs = sorted(glob.glob(os.path.join(HERE, root, b, "part_*.npz")))
    return {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "cpi", "mscs")}


def jain(d, s):
    x = np.stack([d["cpi"] * d["mscs"][:, 4 * s + k] for k in (1, 2, 3)], 1)
    return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def median_curve(st, v):
    cs = []
    for s in np.unique(st):
        h = v[st == s]
        cs.append(np.interp(GX, np.linspace(0, 1, len(h)), h) if len(h) > 1 else np.full_like(GX, h[0]))
    return np.median(np.array(cs), 0)


def short(b):
    return b.split(".")[1].replace("_s", "").replace("_r", "") + ("*" if b in FP_SUITE else "")


def main():
    for root, label in (("results_small", "small starts"), ("results", "random starts")):
        B = sorted(d for d in os.listdir(os.path.join(HERE, root)) if d[0].isdigit())
        fig, axes = plt.subplots(3, 6, figsize=(14, 7), sharex=True, sharey=True)
        print(f"\n== {label}: Jain start->end per stage (Dec | Disp | Iss | Com)")
        for ax, b in zip(axes.ravel(), B):
            d = load(root, b); row = []
            for s in range(4):
                m = median_curve(d["start"], jain(d, s))
                ax.plot(GX, m, color=COL[s], lw=2 if s == 1 else 1.6)
                row.append(f"{m[0]:.2f}->{m[-1]:.2f}")
            ax.set_title(short(b), fontsize=9, color=INK, loc="left")
            ax.grid(axis="y", color=GRID, lw=0.6); ax.set_ylim(1 / 3, 1)
            print(f"  {short(b):11s} " + " | ".join(row))
        for ax in axes.ravel()[len(B):]:
            ax.axis("off")
        h = [plt.Line2D([], [], color=COL[s], lw=2, label=STAGES[s]) for s in range(4)]
        axes.ravel()[-1].legend(handles=h, loc="center", frameon=False, fontsize=9, title="stage (median of 1000 walks)")
        for ax in axes[-1]:
            ax.set_xlabel("walk progress (start = 0, end = 1)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Jain index of FP/BP/SP", fontsize=8)
        fig.suptitle(f"Stall balance at each pipeline stage along greedy walks ({label}; 1/3 = one stall type, "
                     f"1 = all equal; * = SPECfp)", fontsize=10, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fd = os.path.join(HERE, root, "figs"); os.makedirs(fd, exist_ok=True)
        for ext in ("png", "pdf"):
            p = os.path.join(fd, f"fig_balance_stages.{ext}")
            if os.path.exists(p):
                p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
            fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
