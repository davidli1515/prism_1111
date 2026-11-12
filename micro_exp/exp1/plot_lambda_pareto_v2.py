#!/usr/bin/env python3
"""
Loss-weight trade-off and the role of L_mono (100% labels, mean of 3 seeds, lam_const = 1).

(a) Pareto view: x = decomposition error (mean MAE of FP/BP/SP), y = cross-stage
    inconsistency (relative spread of the 4 predicted Base values). One line per
    lam_mono in {0, 0.01}; points ordered by lam_task in {0.1, 0.3, 1, 3, 10, 30}.
    Labels give lam_task and, at the low end, the share of predictions that violate
    monotonicity by > 0.01 (where L_mono matters in-distribution).
(b) Where L_mono matters out-of-distribution: CPI MAPE on unseen SimPoint phases
    (7 SPEC benchmarks) vs lam_task; markers = seed mean, bars = seed std.

Data: exp1/results/lambda_sweep (lam_task <= 10), exp1/results/lambda_probe
(lam_task = 30); unseen-phase errors from window_ids/unseen_simpoint/eval_lambda_1005.json
and eval_lambda_probe_seeds_1005.json.
Usage: python plot_lambda_pareto_v2.py
"""
import os, re, json, glob
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

MICRO = "/path/to/gem5/micro_exp"
SWEEP = f"{MICRO}/exp1/results/lambda_sweep"
PROBE = f"{MICRO}/exp1/results/lambda_probe"
UNSEEN = [f"{MICRO}/window_ids/unseen_simpoint/eval_lambda_1005.json",
          f"{MICRO}/window_ids/unseen_simpoint/eval_lambda_probe_seeds_1005.json"]
OUTPUT = f"{MICRO}/exp1/results/lambda_pareto_v2_1008.png"
B7 = ["508.namd_r", "638.imagick_s", "607.cactuBSSN_s", "526.blender_r",
      "623.xalancbmk_s", "631.deepsjeng_s", "641.leela_s"]
TASKS = [0.1, 0.3, 1, 3, 10, 30]
MONOS = [0, 0.01]
# lam_mono: without (light) vs with (dark) - same hue family as the earlier figure
STYLE = {0: dict(c="#86b6ef", m="o", ls="--", mfc="white", lab="without $L_{mono}$ ($\\lambda_{mono}$=0)"),
         0.01: dict(c="#1c5cab", m="s", ls="-", mfc="#1c5cab", lab="with $L_{mono}$ ($\\lambda_{mono}$=0.01)")}
INK, MUTED = "#222222", "#707070"


def tag_vals(path):
    t, m = re.search(r"t([\dp]+)_c1_m([\dp]+)_s\d+", path).groups()
    return float(t.replace("p", ".")), float(m.replace("p", "."))


def load_indist():
    acc = defaultdict(list)
    for f in glob.glob(f"{SWEEP}/t*_c1_m*_s*.json") + glob.glob(f"{PROBE}/t*_c1_m*_s*.json"):
        t, m = tag_vals(f)
        if t in TASKS and m in MONOS:
            acc[(t, m)].append(json.load(open(f))["selected"])
    out = {}
    for k, v in acc.items():
        out[k] = dict(n=len(v),
                      decomp=np.mean([np.mean([s["mae"][c] for c in ("FP", "BP", "SP")]) for s in v]),
                      spread=np.mean([s["base_spread_rel"] for s in v]) * 100,
                      viol=np.mean([s["mono_viol_gt0.01"] for s in v]),
                      cpi=np.mean([s["mape_mean4"] for s in v]))
    return out


def load_unseen():
    models = {}
    for p in UNSEEN:
        models.update(json.load(open(p))["models"])
    acc = defaultdict(list)
    for k, d in models.items():
        if not re.search(r"(lambda_sweep|lambda_probe)/t[\dp]+_c1_m[\dp]+_s\d+\.pt$", k):
            continue
        t, m = tag_vals(k)
        if t in TASKS and m in MONOS:
            acc[(t, m)].append(np.mean([d["per_benchmark_unseen"][b] for b in B7]))
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in acc.items()}


def style_ax(ax):
    ax.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(1.0)


def main():
    ind, uns = load_indist(), load_unseen()
    for k in sorted(ind):
        assert ind[k]["n"] == 3, f"in-dist {k}: {ind[k]['n']} seeds"
        assert uns[k][2] == 3, f"unseen {k}: {uns[k][2]} seeds"

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) Pareto view
    for m in MONOS:
        st = STYLE[m]
        xs = [ind[(t, m)]["decomp"] for t in TASKS]
        ys = [ind[(t, m)]["spread"] for t in TASKS]
        a.plot(xs, ys, color=st["c"], ls=st["ls"], lw=2, marker=st["m"], ms=7,
               mfc=st["mfc"], mec=st["c"], mew=1.5, label=st["lab"], zorder=3)
    for t in TASKS:                                   # lam_task labels (on the with-L_mono line)
        x, y = ind[(t, 0.01)]["decomp"], ind[(t, 0.01)]["spread"]
        a.annotate(f"$\\lambda_{{task}}$={t:g}", (x, y), textcoords="offset points",
                   xytext=(7, 4), fontsize=8.5, color=INK, zorder=4)
    for t in (0.1, 0.3):                              # violation rates where L_mono matters
        v0, v1 = ind[(t, 0)]["viol"], ind[(t, 0.01)]["viol"]
        x = max(ind[(t, 0)]["decomp"], ind[(t, 0.01)]["decomp"])
        y = (ind[(t, 0)]["spread"] + ind[(t, 0.01)]["spread"]) / 2
        a.annotate(f"violations\n{v0:.0f}% → {v1:.0f}%", (x, y), textcoords="offset points",
                   xytext=(10, -26 if t == 0.1 else -30), fontsize=8.5, color=INK, ha="left",
                   bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#B0B0B0", lw=0.6), zorder=5)
    a.set_xlabel("Decomposition error (MAE of FP/BP/SP)", fontweight="bold")
    a.set_ylabel("Cross-stage inconsistency (%)", fontweight="bold")
    a.set_title("(a) Task weight trades decomposition for consistency", fontweight="bold", fontsize=11, pad=8)
    a.set_xlim(0, max(ind[(0.1, 0)]["decomp"], ind[(0.1, 0.01)]["decomp"]) * 1.45)
    style_ax(a)
    a.legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False,
             framealpha=1.0).get_frame().set_linewidth(1.0)

    # (b) unseen SimPoint phases
    for i, m in enumerate(MONOS):
        st = STYLE[m]
        mu = [uns[(t, m)][0] for t in TASKS]
        sd = [uns[(t, m)][1] for t in TASKS]
        xs = [t * (0.93 if i == 0 else 1.07) for t in TASKS]   # small dodge
        b.errorbar(xs, mu, yerr=sd, color=st["c"], ls=st["ls"], lw=2, marker=st["m"], ms=7,
                   mfc=st["mfc"], mec=st["c"], mew=1.5, capsize=3, elinewidth=1.2,
                   label=st["lab"], zorder=3)
    b.set_xscale("log")
    b.set_xticks(TASKS); b.set_xticklabels([f"{t:g}" for t in TASKS])
    b.set_xlabel("$\\lambda_{task}$ (log scale)", fontweight="bold")
    b.set_ylabel("CPI MAPE on unseen SimPoint phases (%)", fontweight="bold")
    b.set_title("(b) $L_{mono}$ stabilizes extrapolation", fontweight="bold", fontsize=11, pad=8)
    style_ax(b)
    b.legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False,
             framealpha=1.0).get_frame().set_linewidth(1.0)

    cpi = [ind[k]["cpi"] for k in ind]
    fig.text(0.5, -0.03, f"100% labels, mean of 3 seeds, $\\lambda_{{const}}$=1. In-distribution CPI MAPE "
             f"stays within {min(cpi):.2f}-{max(cpi):.2f}% for all points. Bars in (b): std over seeds.",
             ha="center", fontsize=8.5, color=MUTED)
    plt.tight_layout()
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {OUTPUT} (+ .pdf)")
    for t in TASKS:
        r = [f"m={m:g}: decomp {ind[(t,m)]['decomp']:.3f} spread {ind[(t,m)]['spread']:.2f}% "
             f"viol {ind[(t,m)]['viol']:.1f}% unseen {uns[(t,m)][0]:.2f}±{uns[(t,m)][1]:.2f}" for m in MONOS]
        print(f"  t={t:<4g} | " + " | ".join(r))


if __name__ == "__main__":
    main()
