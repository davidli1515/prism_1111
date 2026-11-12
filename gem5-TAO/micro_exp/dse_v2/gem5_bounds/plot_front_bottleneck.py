#!/usr/bin/env python3
"""
Bottleneck migration along the Pareto front: Dispatch-stage MSCS shares (Base/FP/BP/SP) of every
validated BO-front point, PRISM prediction (left bar) next to gem5 measurement (right, hatched).
Data: gem5_bounds/runs/*/p*_base/result.json; PRISM recomputed with the DSE surrogate.
Usage: python plot_front_bottleneck.py [--stage 1] [--benches ...]
"""
import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DSE_V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DSE_V2, "mscs_guide")); sys.path.insert(0, DSE_V2)

CATS = ["Base", "FP", "BP", "SP"]
COL = {"Base": "#2a78d6", "FP": "#eb6834", "BP": "#1baf7a", "SP": "#eda100"}
STAGES = ["Decode", "Dispatch", "Issue", "Commit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--benches", nargs="+", default=["602.gcc_s", "605.mcf_s", "641.leela_s", "638.imagick_s"])
    args = ap.parse_args()
    from refine_v3 import SurrogateWL
    from check_bounds import stack
    from common import DEFAULT_CKPT, Area

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 10, 'axes.labelsize': 11, 'axes.linewidth': 1.2,
                         'xtick.direction': 'in', 'ytick.direction': 'in'})
    fig, axes = plt.subplots(1, len(args.benches), figsize=(4.2 * len(args.benches), 3.4), sharey=True)
    A = Area(); s = args.stage
    for ax, b in zip(np.atleast_1d(axes), args.benches):
        runs = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "runs", b, "p*_base", "result.json"))]
        S = SurrogateWL(DEFAULT_CKPT, [b], "18k", device="cpu", max_slices=50)
        pts = []
        for r in runs:
            _, sh = stack(S, r["config"])
            g = np.array(r["mscs16"])
            pts.append((A(r["config"]), [sh[c][s] for c in CATS], [g[4 * s + k] for k in range(4)]))
        pts.sort(key=lambda t: t[0])
        x = np.arange(len(pts)); w = 0.4
        for side, off, hatch in [(1, -w / 2, None), (2, w / 2, "////")]:
            bottom = np.zeros(len(pts))
            for k, c in enumerate(CATS):
                h = np.array([p[side][k] for p in pts])
                ax.bar(x + off, h, w * 0.92, bottom=bottom, color=COL[c], edgecolor="white", linewidth=0.8,
                       hatch=hatch, label=c if (side == 1 and ax is np.atleast_1d(axes)[0]) else None)
                bottom += h
        ax.set_xticks(x); ax.set_xticklabels([f"{p[0]:.0f}" for p in pts], fontsize=8)
        ax.set_xlabel("Front point area (mm$^2$)")
        ax.set_title(b, fontweight="bold")
        ax.set_ylim(0, 1); ax.grid(axis="y", ls=":", alpha=0.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_ylabel(f"{STAGES[s]}-stage share")
    from matplotlib.patches import Patch
    h, l = ax0.get_legend_handles_labels()
    h += [Patch(facecolor="white", edgecolor="gray"), Patch(facecolor="white", edgecolor="gray", hatch="////")]
    l += ["PRISM (left)", "gem5 (right)"]
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(HERE, f"fig_front_bottleneck_{STAGES[s].lower()}.{ext}")
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("saved", p)


if __name__ == "__main__":
    main()
