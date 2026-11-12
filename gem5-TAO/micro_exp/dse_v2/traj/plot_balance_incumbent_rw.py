#!/usr/bin/env python3
"""Random-walker counterpart of plot_balance_incumbent.py (same style, same incumbent definition).
Runs = the first 10 consecutive blocks of 512 designs of results_rw/<bench> (the RW-512 runs of analyze_ga_rw.py,
same budget as GA). Incumbent = best-CPI design so far with area <= A_ref (A_ref from results_ga/<bench>/meta.json);
per stage, Jain index of FP/BP/SP stall cycles of the incumbent vs evaluation progress, median over the 10 runs.
Writes results_rw/figs/fig_balance_incumbent_rw.{png,pdf}; refuses to overwrite."""
import os, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_balance_incumbent as PI                       # rcParams, colours, GX, jain_rows (read-only)
from plot_balance_stages import short, STAGES

HERE = os.path.dirname(os.path.abspath(__file__)); BLOCK, NB = 512, 10


def curves(cpi, area, m, aref):
    n = len(cpi); inc = np.full(n, -1); best, bi = np.inf, -1
    for i in range(n):
        if area[i] <= aref and cpi[i] < best:
            best, bi = cpi[i], i
        inc[i] = bi
    pos = np.minimum((PI.GX * (n - 1)).round().astype(int), n - 1)
    return np.array([np.where(inc[pos] >= 0, PI.jain_rows(cpi, m, s)[np.maximum(inc[pos], 0)], np.nan) for s in range(4)])


def main():
    fd = os.path.join(HERE, "results_rw", "figs"); os.makedirs(fd, exist_ok=True)
    outs = [os.path.join(fd, f"fig_balance_incumbent_rw.{e}") for e in ("png", "pdf")]
    if any(os.path.exists(p) for p in outs):
        sys.exit(f"exists, refusing to overwrite: {outs}")
    B = sorted(d for d in os.listdir(os.path.join(HERE, "results_ga")) if d[0].isdigit())
    fig, axes = plt.subplots(3, 6, figsize=(12, 6), sharex=True, sharey=True)
    print("== RW-512: incumbent Jain first-feasible -> end, per stage (Dec | Disp | Iss | Com)")
    for ax, b in zip(axes.ravel(), B):
        aref = json.load(open(os.path.join(HERE, "results_ga", b, "meta.json")))["A_ref"]
        fs = sorted(glob.glob(os.path.join(HERE, "results_rw", b, "part_*.npz")))
        z = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("idx", "cpi", "area", "mscs")}
        o = np.argsort(z["idx"]); z = {k: v[o] for k, v in z.items()}
        C = np.array([curves(z["cpi"][sl], z["area"][sl], z["mscs"][sl], aref)
                      for sl in (slice(r * BLOCK, (r + 1) * BLOCK) for r in range(NB))])
        with np.errstate(all="ignore"):
            import warnings; warnings.simplefilter("ignore")
            med = np.nanmedian(C, 0)
        row = []
        for s in range(4):
            ax.plot(PI.GX, med[s], color=PI.COL[s], ls=PI.LS[s], lw=1.8 if s == 1 else 1.3, zorder=3)
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
    h = [plt.Line2D([], [], color=PI.COL[s], ls=PI.LS[s], lw=1.8, label=STAGES[s]) for s in range(4)]
    last.legend(handles=h, loc="center", frameon=True, edgecolor="black", fancybox=False).get_frame().set_linewidth(1.0)
    fig.supxlabel("Random search evaluation progress (first = 0, last = 1)", fontweight="bold", fontsize=11)
    fig.supylabel("Jain index of FP/BP/SP stall cycles (incumbent)", fontweight="bold", fontsize=11)
    fig.tight_layout(h_pad=0.4, w_pad=1.2)
    for p in outs:
        fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
