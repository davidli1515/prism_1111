#!/usr/bin/env python3
"""fig_stage_jain_opt_column plus the CPI of the same designs on a right axis: per panel, the normalised CPI
(current design / incumbent CPI divided by the CPI at the first point of the curve: greedy start, first feasible
incumbent), same aggregation (median over walks / runs, then over benchmarks, same thresholds).
Writes results_rw/figs/fig_stage_jain_opt_cpi_column.{png,pdf}; refuses to overwrite."""
import os, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_stage_jain_opt_column as S                      # rcParams, COL, LS, per_bench, across (read-only)
from plot_balance_stages import load, jain, median_curve, GX as GX_G, STAGES
import plot_balance_incumbent as PI
import plot_balance_incumbent_rw as PR

HERE = S.HERE; CPI_COL = "black"


def inc_cpi(cpi, area, aref, gx):
    """incumbent CPI / first feasible incumbent CPI at progress gx (nan before the first feasible design)."""
    n = len(cpi); out = np.full(n, np.nan); best = np.inf
    for i in range(n):
        if area[i] <= aref and cpi[i] < best:
            best = cpi[i]
        out[i] = best if np.isfinite(best) else np.nan
    pos = np.minimum((gx * (n - 1)).round().astype(int), n - 1)
    ok = np.where(~np.isnan(out))[0]
    return out[pos] / out[ok[0]] if len(ok) else out[pos]


def thresh(C, k):
    med = np.nanmedian(C, 0); med[np.sum(~np.isnan(C), 0) < k] = np.nan
    return med


def main():
    fd = os.path.join(HERE, "results_rw", "figs"); outs = [os.path.join(fd, f"fig_stage_jain_opt_cpi_column.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    B = sorted(d for d in os.listdir(os.path.join(HERE, "results_ga")) if d[0].isdigit())
    aref = {b: json.load(open(os.path.join(HERE, "results_ga", b, "meta.json")))["A_ref"] for b in B}
    curves = {}
    J, Cc = [], []
    for b in B:
        d = load("results_small", b)
        J.append([median_curve(d["start"], jain(d, s)) for s in range(4)])
        first = {s: d["cpi"][np.argmax(d["start"] == s)] for s in np.unique(d["start"])}
        Cc.append(median_curve(d["start"], d["cpi"] / np.array([first[s] for s in d["start"]])))
    curves["Greedy"] = (GX_G, S.across(np.array(J)), np.median(np.array(Cc), 0))
    for opt, name in (("bo", "BO"), ("ga", "GA")):
        J, Cc = [], []
        for b in B:
            fs = sorted(glob.glob(os.path.join(HERE, f"results_{opt}", b, "seed_*.npz")))
            J.append(S.per_bench(np.array([PI.incumbent_curves(f, aref[b]) for f in fs])))
            Cc.append(thresh(np.array([inc_cpi(np.load(f)["cpi"], np.load(f)["area"], aref[b], PI.GX) for f in fs]), S.MIN_RUNS))
        curves[name] = (PI.GX, S.across(np.array(J)), thresh(np.array(Cc), S.MIN_BENCH))
    J, Cc = [], []
    for b in B:
        fs = sorted(glob.glob(os.path.join(HERE, "results_rw", b, "part_*.npz")))
        z = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("idx", "cpi", "area", "mscs")}
        o = np.argsort(z["idx"]); z = {k: v[o] for k, v in z.items()}
        sls = [slice(r * PR.BLOCK, (r + 1) * PR.BLOCK) for r in range(PR.NB)]
        J.append(S.per_bench(np.array([PR.curves(z["cpi"][sl], z["area"][sl], z["mscs"][sl], aref[b]) for sl in sls])))
        Cc.append(thresh(np.array([inc_cpi(z["cpi"][sl], z["area"][sl], aref[b], PI.GX) for sl in sls]), S.MIN_RUNS))
    curves["Random"] = (PI.GX, S.across(np.array(J)), thresh(np.array(Cc), S.MIN_BENCH))

    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.6), sharex=True, sharey=True)
    print("normalised CPI first -> end; Dispatch Jain first -> end")
    for i, (ax, name) in enumerate(zip(axes.ravel(), ("Greedy", "BO", "GA", "Random"))):
        x, med, cpi = curves[name]
        for s in range(4):
            ax.plot(x, med[s], color=S.COL[s], ls=S.LS[s], lw=1.5 if s == 1 else 1.1, zorder=3)
        ax2 = ax.twinx()
        ax2.plot(x, cpi, color=CPI_COL, ls=":", lw=1.3, zorder=4)
        ax2.set_ylim(0.55, 1.12); ax2.set_yticks([0.6, 0.8, 1.0]); ax2.set_yticklabels(["0.6", "0.8", "1"])
        ax2.tick_params(axis="y", direction="in", labelright=(i % 2 == 1), length=2.5, width=0.8)
        v = cpi[~np.isnan(cpi)]; dj = med[1][~np.isnan(med[1])]
        print(f"  {name:7s} CPI {v[0]:.2f}->{v[-1]:.2f} | Disp Jain {dj[0]:.2f}->{dj[-1]:.2f}")
        ax.text(0.04, 0.93, name, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.set_ylim(1 / 3, 0.9); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([0.4, 0.6, 0.8])
        ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
        for a in (ax, ax2):
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)
    h = [plt.Line2D([], [], color=S.COL[s], ls=S.LS[s], lw=1.5, label=STAGES[s]) for s in range(4)] + \
        [plt.Line2D([], [], color=CPI_COL, ls=":", lw=1.3, label="Norm. CPI")]
    fig.supylabel("Jain index", fontweight="bold", fontsize=8, x=0.045)
    fig.tight_layout(h_pad=0.4, w_pad=0.6, rect=(0.03, 0.2, 0.93, 1))
    a, c = axes[1, 0].get_position(), axes[1, 1].get_position(); xc = (a.x0 + c.x1) / 2
    fig.text(c.x1 + 0.085, (a.y0 + axes[0, 1].get_position().y1) / 2, "Normalized CPI", rotation=270, ha="center",
             va="center", fontweight="bold", fontsize=8)
    fig.text(xc, a.y0 - 0.09, "Search progress (start = 0, end = 1)", ha="center", va="top", fontweight="bold", fontsize=8)
    fig.legend(handles=h, ncol=3, loc="upper center", bbox_to_anchor=(xc, a.y0 - 0.15), frameon=True, edgecolor="black",
               fancybox=False, framealpha=1, columnspacing=0.8, handlelength=1.8, handletextpad=0.4,
               borderpad=0.3).get_frame().set_linewidth(0.8)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
