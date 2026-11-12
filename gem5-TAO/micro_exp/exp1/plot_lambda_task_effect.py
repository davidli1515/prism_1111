#!/usr/bin/env python3
"""
Effect of lam_task and lam_mono (lam_const = 1) on the MSCS decomposition.

  cols : decomposition error (mean MAE of FP/BP/SP) | monotonicity violations
         (% test samples with a cross-stage violation > 0.01) | cross-stage inconsistency
         (mean relative spread of the 4 predicted Base values)
  x    : lam_task (log scale); one line per lam_mono (ordinal blue ramp + marker)
  rows : 100% labels (exp1/results/lambda_sweep_1004, mean of 3 seeds)
         5% labels, semi-supervised V3 (exp4/results_lambda_lowlabel_1004, 1 seed)

Usage: python plot_lambda_task_effect.py
"""
import os, json, glob
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

MICRO = "/scratch/gem5-TAO/micro_exp"
EXP1_DIR = f"{MICRO}/exp1/results/lambda_sweep_1004"
LOWL_DIR = f"{MICRO}/exp4/results_lambda_lowlabel_1004"
OUTPUT = f"{MICRO}/exp1/results/lambda_task_effect_3col_1004.png"

# lam_mono is ordinal -> one-hue ramp (validated: ordinal, light mode) + marker shape
MONO_STYLE = {0: ("#86b6ef", "o"), 0.01: ("#3987e5", "s"), 0.1: ("#1c5cab", "^"), 1: ("#0d366b", "D")}
MUTED = "#707070"
METRICS = [("decomp", "Decomposition error\n(MAE of FP/BP/SP)"),
           ("viol", "Monotonicity violations\n(% samples, >0.01)"),
           ("spread", "Cross-stage inconsistency\n(Base spread, %)")]


def metric(s, key):
    if key == "viol":
        return s["mono_viol_gt0.01"]
    if key == "spread":
        return s["base_spread_rel"] * 100
    return np.mean([s["mae"][k] for k in ("FP", "BP", "SP")])


def load(files, filt):
    acc = defaultdict(list)
    for f in files:
        j = json.load(open(f))
        if filt(j):
            acc[(j["lam_task"], j["lam_mono"])].append(j["selected"])
    return acc


def main():
    rows = [
        ("100% labels (3 seeds)", load(glob.glob(f"{EXP1_DIR}/*.json"), lambda j: j["lam_const"] == 1)),
        ("5% labels, semi-supervised (1 seed)",
         load(glob.glob(f"{LOWL_DIR}/v3_r5_*.json"), lambda j: True)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4), sharex=True)
    for r, (title, acc) in enumerate(rows):
        for c, (key, ylabel) in enumerate(METRICS):
            ax = axes[r, c]
            for m, (col, mk) in MONO_STYLE.items():
                ts = sorted(t for (t, mm) in acc if mm == m)
                if not ts:
                    continue
                ys = [np.mean([metric(s, key) for s in acc[(t, m)]]) for t in ts]
                ax.plot(ts, ys, color=col, lw=2, zorder=2)
                ax.scatter(ts, ys, s=46, color=col, marker=mk, edgecolor="white",
                           linewidth=1.2, zorder=3, label=f"$\\lambda_{{mono}}$={m:g}")
            ax.set_xscale("log")
            ticks = [0.1, 0.3, 1, 3, 10]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_ylim(bottom=0)
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_title(f"({'abcdef'[3*r+c]}) {title}", fontweight="bold", pad=8, fontsize=11)
            ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
            for s in ax.spines.values():
                s.set_color("black"); s.set_linewidth(1.0)
            if r == 1:
                ax.set_xlabel("$\\lambda_{task}$ (log scale)", fontweight="bold")
    axes[0, 0].legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False,
                      framealpha=1.0).get_frame().set_linewidth(1.0)
    fig.text(0.5, -0.01, "$\\lambda_{const}$=1. CPI MAPE stays within 2.87-3.24% (100%) and "
             "8.43-8.80% (5%) across all settings.", ha="center", fontsize=9, color=MUTED)
    plt.tight_layout()
    if os.path.exists(OUTPUT):
        raise SystemExit(f"{OUTPUT} exists, refusing to overwrite")
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", dpi=600)
    print(f"Saved {OUTPUT} (+ .pdf)")


if __name__ == "__main__":
    main()
