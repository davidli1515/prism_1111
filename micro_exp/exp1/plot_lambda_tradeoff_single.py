#!/usr/bin/env python3
"""
Single-panel summary of the lam_task trade-off (lam_const = 1, 100% labels, 3 seeds).

All metrics are indexed to their maximum over the plotted settings (common 0-1 base, one axis):
  decomposition error (MAE of FP/BP/SP)      lam_mono = 0.01        falls with lam_task
  monotonicity violations (% samples >0.01)  lam_mono = 0.01 solid  falls with lam_task
                                             lam_mono = 0    dashed (gap = L_mono guard)
  cross-stage inconsistency (Base spread)    lam_mono = 0.01        rises with lam_task (cost)
  CPI MAPE                                   lam_mono = 0.01        flat
Raw values are printed next to the end points.

Usage: python plot_lambda_tradeoff_single.py [--labels 100|5]
"""
import os, json, glob, argparse
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 600, 'axes.linewidth': 1.0,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 4, 'ytick.major.size': 4,
})

MICRO = "/path/to/gem5/micro_exp"
SRC = {"100": (f"{MICRO}/exp1/results/lambda_sweep/*.json", lambda j: j["lam_const"] == 1, "100% labels, 3 seeds"),
       "5":   (f"{MICRO}/exp4/results_lambda_lowlabel/v3_r5_*.json", lambda j: True, "5% labels, semi-supervised, 1 seed")}
OUT = f"{MICRO}/exp1/results/lambda_tradeoff_single_{{}}_1004.png"
# validated categorical slots 1-4 (contrast WARN -> every line is direct-labeled + own marker)
C_DEC, C_VIOL, C_INC, C_CPI = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#222222", "#707070"

M = {"decomp": lambda s: np.mean([s["mae"][k] for k in ("FP", "BP", "SP")]),
     "viol":   lambda s: s["mono_viol_gt0.01"],
     "incons": lambda s: s["base_spread_rel"] * 100,
     "cpi":    lambda s: s["mape_mean4"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--labels", choices=list(SRC), default="100")
    a = ap.parse_args()
    pattern, filt, subtitle = SRC[a.labels]
    acc = defaultdict(list)
    for f in glob.glob(pattern):
        j = json.load(open(f))
        if filt(j):
            acc[(j["lam_task"], j["lam_mono"])].append(j["selected"])
    ts = sorted({t for t, m in acc if m == 0.01})
    val = lambda key, m: np.array([np.mean([M[key](s) for s in acc[(t, m)]]) for t in ts])

    series = [  # (values, normalizer, color, marker, linestyle, label, unit fmt)
        (val("decomp", 0.01), None, C_DEC, "o", "-", "Decomposition error", "{:.3f}"),
        (val("viol", 0.01), "viol", C_VIOL, "s", "-", "Monotonicity violations", "{:.1f}%"),
        (val("viol", 0), "viol", C_VIOL, "s", "--", "Violations without $L_{mono}$", "{:.1f}%"),
        (val("incons", 0.01), None, C_INC, "^", "-", "Cross-stage inconsistency", "{:.2f}%"),
    ]
    viol_max = max(series[1][0].max(), series[2][0].max()) or 1.0

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.fill_between(ts, series[1][0] / viol_max, series[2][0] / viol_max,
                    color=C_VIOL, alpha=0.12, lw=0, zorder=1)
    for v, norm, c, mk, ls, lab, f in series:
        y = v / (viol_max if norm == "viol" else v.max())
        mfc = "white" if ls == "--" else c
        ax.plot(ts, y, color=c, lw=2, ls=ls, marker=mk, ms=7, mfc=mfc, mec=c if ls == "--" else "white",
                mew=1.2, zorder=3, label=lab)
    gx = (ts[0] * ts[1]) ** 0.5
    gy = (series[1][0][:2].mean() + series[2][0][:2].mean()) / 2 / viol_max
    ax.annotate("$L_{mono}$ guard", (gx, gy), fontsize=9, color=INK, ha="center", va="center")

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ts))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ts]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(ts[0] / 1.3, ts[-1] * 1.3)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("$\\lambda_{task}$ (log scale)", fontweight="bold")
    ax.set_ylabel("Relative to max over $\\lambda_{task}$", fontweight="bold")
    ax.set_title(f"Loss-weight trade-off ({subtitle})", fontweight="bold", pad=8)
    ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=True, edgecolor="black",
              fancybox=False, framealpha=1.0, fontsize=9, handlelength=4.5).get_frame().set_linewidth(1.0)
    rng = lambda v, f: f"{f.format(v[0])} \u2192 {f.format(v[-1])}"
    cpi = val("cpi", 0.01)
    fig.text(0.5, -0.04, "Raw values, $\\lambda_{task}$ = " + f"{ts[0]:g} \u2192 {ts[-1]:g}: "
             f"decomposition error {rng(series[0][0], '{:.3f}')}; violations {rng(series[1][0], '{:.1f}%')} "
             f"(without $L_{{mono}}$: {rng(series[2][0], '{:.1f}%')});\n"
             f"inconsistency {rng(series[3][0], '{:.2f}%')}; CPI MAPE {cpi.min():.2f}-{cpi.max():.2f}% throughout. "
             "$\\lambda_{const}$=1, $\\lambda_{mono}$=0.01 unless noted.",
             ha="center", fontsize=8.5, color=MUTED)
    plt.tight_layout()
    out = OUT.format(a.labels)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists, refusing to overwrite")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", dpi=600)
    print(f"Saved {out} (+ .pdf)")


if __name__ == "__main__":
    main()
