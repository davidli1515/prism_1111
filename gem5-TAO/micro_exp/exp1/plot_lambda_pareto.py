#!/usr/bin/env python3
"""
Trade-off between bottleneck-decomposition accuracy and cross-stage CPI consistency
as the loss weights vary (lam_const = 1).

  x: decomposition error = mean MAE of the FP/BP/SP proportions (all stages)
  y: cross-stage inconsistency = mean relative spread of the 4 predicted Base values
  lower-left is better; dashed gray = Pareto front over all points in the panel.
  one line per lam_mono (ordinal blue ramp + marker shape), points ordered by lam_task.

Panel (a): 100% labels, exp1/results/lambda_sweep_1004 (mean of 3 seeds)
Panel (b): 5% labels, V3, exp4/results_lambda_lowlabel_1004 (1 seed)

Usage: python plot_lambda_pareto.py
"""
import os, json, glob, argparse
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
OUTPUTS = {"spread": f"{MICRO}/exp1/results/lambda_pareto_1004.png",
           "viol":   f"{MICRO}/exp1/results/lambda_pareto_viol_1004.png"}
YKEY = {"spread": lambda s: s["base_spread_rel"] * 100, "viol": lambda s: s["mono_viol_gt0.01"]}
YLABEL = {"spread": "Cross-stage inconsistency (Base spread, %)",
          "viol": "Monotonicity violations (% samples, >0.01)"}
Y = "spread"

# lam_mono is ordinal -> one-hue ramp (validated: ordinal, light mode), plus marker shape
MONO_STYLE = {0: ("#86b6ef", "o"), 0.01: ("#3987e5", "s"), 0.1: ("#1c5cab", "^"), 1: ("#0d366b", "D")}
INK, MUTED = "#222222", "#707070"


def decomp_err(s):
    return np.mean([s["mae"][k] for k in ("FP", "BP", "SP")])


def load_exp1():
    acc = defaultdict(list)
    for f in glob.glob(f"{EXP1_DIR}/*.json"):
        j = json.load(open(f))
        if j["lam_const"] != 1:
            continue
        acc[(j["lam_task"], j["lam_mono"])].append(j["selected"])
    return {k: (np.mean([decomp_err(s) for s in v]),
                np.mean([YKEY[Y](s) for s in v]),
                np.mean([s["mape_mean4"] for s in v])) for k, v in acc.items()}


def load_lowlabel(ratio=5, variant="v3"):
    out = {}
    for f in glob.glob(f"{LOWL_DIR}/{variant}_r{ratio}_*.json"):
        j = json.load(open(f)); s = j["selected"]
        out[(j["lam_task"], j["lam_mono"])] = (decomp_err(s), YKEY[Y](s), s["mape_mean4"])
    return out


def pareto(points):
    pts = sorted(points)
    front, best_y = [], np.inf
    for x, y in pts:
        if y < best_y:
            front.append((x, y)); best_y = y
    return np.array(front)


def panel(ax, data, title):
    allpts = [(v[0], v[1]) for v in data.values()]
    fr = pareto(allpts)
    ax.step(fr[:, 0], fr[:, 1], where="post", color=MUTED, lw=1.0, ls="--", zorder=1)
    for m, (c, mk) in MONO_STYLE.items():
        keys = sorted(k for k in data if k[1] == m)
        if not keys:
            continue
        xs = [data[k][0] for k in keys]; ys = [data[k][1] for k in keys]
        ax.plot(xs, ys, color=c, lw=2, zorder=2)
        ax.scatter(xs, ys, s=46, color=c, marker=mk, edgecolor="white", linewidth=1.2,
                   zorder=3, label=f"$\\lambda_{{mono}}$={m:g}")
        if m == 0:          # label lam_task once, on the lam_mono = 0 line
            for k, x, y in zip(keys, xs, ys):
                ax.annotate(f"$\\lambda_{{task}}$={k[0]:g}, CPI {data[k][2]:.2f}%", (x, y),
                            textcoords="offset points", fontsize=8, color=INK,
                            xytext=(-4, 8) if x == max(xs) else (4, -16),
                            ha="right" if x == max(xs) else "left",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                            zorder=4)
    ax.set_xlabel("Decomposition error (MAE of FP/BP/SP)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=8)
    ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
    ax.xaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(1.0)


def main():
    global Y
    ap = argparse.ArgumentParser(); ap.add_argument("--y", choices=list(YKEY), default="spread")
    Y = ap.parse_args().y
    OUTPUT = OUTPUTS[Y]
    d100, d5 = load_exp1(), load_lowlabel(5, "v3")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panel(axes[0], d100, "(a) 100% labels (3 seeds)")
    panel(axes[1], d5, "(b) 5% labels, semi-supervised (1 seed)")
    axes[0].set_ylabel(YLABEL[Y], fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=MUTED, ls="--", lw=1.0)); l.append("Pareto front")
    axes[0].legend(h, l, loc="upper right", frameon=True, edgecolor="black", fancybox=False,
                   framealpha=1.0).get_frame().set_linewidth(1.0)
    fig.text(0.5, -0.02, "Annotations on the $\\lambda_{mono}$=0 line: $\\lambda_{task}$ and CPI MAPE. "
             "Lower-left is better; $\\lambda_{const}$=1.", ha="center", fontsize=9, color=MUTED)
    plt.tight_layout()
    if os.path.exists(OUTPUT):
        raise SystemExit(f"{OUTPUT} exists, refusing to overwrite")
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", dpi=600)
    print(f"Saved {OUTPUT} (+ .pdf)")

    for name, d in [("100%", d100), ("5% V3", d5)]:
        print(f"\n{name}: lam_task lam_mono | decomp_err  y  CPI%  | on front")
        fr = {tuple(p) for p in pareto([(v[0], v[1]) for v in d.values()])}
        for k in sorted(d):
            v = d[k]
            print(f"  {k[0]:>5g} {k[1]:>5g} | {v[0]:.4f}  {v[1]:.3f}  {v[2]:.2f} | {'*' if (v[0], v[1]) in fr else ''}")


if __name__ == "__main__":
    main()
