#!/usr/bin/env python3
"""BO and GA counterpart of fig_balance_stages_doc_v2 (same style). BO and GA do not walk, so the path is the
INCUMBENT: after each evaluation, the best-CPI design evaluated so far with area <= A_ref (A_ref = GA's area
budget = median end area of the greedy small-start walks, read from results_ga/<bench>/meta.json). Per stage,
Jain index of FP/BP/SP stall cycles of the incumbent vs evaluation progress (0..1), median over the 10 seeds;
evaluations before the first feasible design are left blank.
Writes results_<opt>/figs/fig_balance_incumbent_<opt>.{png,pdf} for opt in bo, ga; refuses to overwrite."""
import os, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_balance_stages import short, STAGES

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 10,
    'figure.dpi': 600, 'axes.linewidth': 1.0,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
})
COL = ["#2E86C1", "#E67E22", "#27AE60", "#8E44AD"]
LS = ["-", "-", "--", "-."]
HERE = os.path.dirname(os.path.abspath(__file__))
GX = np.linspace(0, 1, 101)


def jain_rows(cpi, m, s):
    x = np.stack([cpi * m[:, 4 * s + k] for k in (1, 2, 3)], 1)
    return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def incumbent_curves(f, aref):
    z = np.load(f); cpi, area, m = z["cpi"], z["area"], z["mscs"]
    n = len(cpi); inc = np.full(n, -1); best, bi = np.inf, -1
    for i in range(n):
        if area[i] <= aref and cpi[i] < best:
            best, bi = cpi[i], i
        inc[i] = bi
    out = np.full((4, len(GX)), np.nan)
    pos = np.minimum((GX * (n - 1)).round().astype(int), n - 1)
    for s in range(4):
        J = jain_rows(cpi, m, s)
        v = np.where(inc[pos] >= 0, J[np.maximum(inc[pos], 0)], np.nan)
        out[s] = v
    return out


def main():
    B = sorted(d for d in os.listdir(os.path.join(HERE, "results_ga")) if d[0].isdigit())
    aref = {b: json.load(open(os.path.join(HERE, "results_ga", b, "meta.json")))["A_ref"] for b in B}
    for opt, label in (("bo", "BO"), ("ga", "GA")):
        root = os.path.join(HERE, f"results_{opt}"); fd = os.path.join(root, "figs"); os.makedirs(fd, exist_ok=True)
        outs = [os.path.join(fd, f"fig_balance_incumbent_{opt}.{e}") for e in ("png", "pdf")]
        if any(os.path.exists(p) for p in outs):
            sys.exit(f"exists, refusing to overwrite: {outs}")
        fig, axes = plt.subplots(3, 6, figsize=(12, 6), sharex=True, sharey=True)
        print(f"== {label}: incumbent Jain first-feasible -> end, per stage (Dec | Disp | Iss | Com)")
        for ax, b in zip(axes.ravel(), B):
            C = np.array([incumbent_curves(f, aref[b]) for f in sorted(glob.glob(os.path.join(root, b, "seed_*.npz")))])
            with np.errstate(all="ignore"):
                med = np.nanmedian(C, 0)
            row = []
            for s in range(4):
                ax.plot(GX, med[s], color=COL[s], ls=LS[s], lw=1.8 if s == 1 else 1.3, zorder=3)
                ok = ~np.isnan(med[s]); row.append(f"{med[s][ok][0]:.2f}->{med[s][ok][-1]:.2f}" if ok.any() else "-")
            print(f"  {short(b):11s} " + " | ".join(row))
            ax.text(0.04, 0.93, short(b), transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
            ax.set_ylim(1 / 3, 1); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
            ax.set_yticks([0.4, 0.6, 0.8, 1.0])
            ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.0)
        last = axes.ravel()[len(B)]
        for ax in axes.ravel()[len(B):]:
            ax.axis("off")
        h = [plt.Line2D([], [], color=COL[s], ls=LS[s], lw=1.8, label=STAGES[s]) for s in range(4)]
        last.legend(handles=h, loc="center", frameon=True, edgecolor="black", fancybox=False).get_frame().set_linewidth(1.0)
        fig.supxlabel(f"{label} evaluation progress (first = 0, last = 1)", fontweight="bold", fontsize=11)
        fig.supylabel("Jain index of FP/BP/SP stall cycles (incumbent)", fontweight="bold", fontsize=11)
        fig.tight_layout(h_pad=0.4, w_pad=1.2)
        for p in outs:
            fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)
        plt.close(fig)


if __name__ == "__main__":
    main()
