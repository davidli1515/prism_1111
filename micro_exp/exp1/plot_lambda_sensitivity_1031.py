#!/usr/bin/env python3
"""
Hyperparameter sensitivity of lam_task (V2, 100% labels, 3873 Concorde features, 3 seeds).

x: lam_task (log scale), lam_const = 1; one line per lam_mono in {0, 0.01}; bars = seed std.
Half-column figure, 2 x 2 panels:
  (a) CPI MAPE (%), mean of the 4 per-stage CPIs; y axis 2.5-3.5, does not start at 0
  (b) decomposition error: mean MAE of the FP/BP/SP proportions over all stages
  (c) cross-stage Base inconsistency (%): (max - min)/mean of the 4 predicted Base values,
      averaged over samples (0 for ground truth)
  (d) monotonicity violation rate (%): predictions where FP or SP increases, or BP decreases,
      from Decode to Commit by more than 0.01
The chosen setting (lam_task, lam_const, lam_mono) = (10, 1, 0) is ringed and labeled.

Data: results/lambda_sensitivity_nod3_1031.json (from collect_lambda_sensitivity_1031.py)
Usage: python plot_lambda_sensitivity_1031.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, FormatStrFormatter, NullLocator

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 8,
    'figure.dpi': 600, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 3, 'ytick.major.size': 3,
})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "results", "lambda_sensitivity_nod3_1031.json")
OUTPUT = os.path.join(HERE, "results", "lambda_sensitivity_nod3_1031_v3.png")
CHOSEN_T, CHOSEN_M = 10, 0

MONO_STYLE = {0: dict(color="#FF6F00", marker="o", ls="-", lw=1.2, ms=3.5, zorder=4),
              0.01: dict(color="#3987e5", marker="s", ls="--", lw=1.1, ms=3.0, zorder=3)}
PANELS = [("(a) CPI MAPE (%)", "cpi_mape", "{:.1f}", (2.5, 3.5), [2.5, 3.0, 3.5]),
          ("(b) Decomp. MAE", "component_mae", "{:.3f}", (0, None), None),
          ("(c) Base incons. (%)", "base_incons", "{:.1f}", (0, None), None),
          ("(d) Violation rate (%)", "mono_viol", "{:.2f}", (0, None), None)]


def main():
    rows =json.load(open(DATA))["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.45), sharex=True)
    axes = axes.flatten()
    for ax, (title, key, vfmt, ylim, yticks) in zip(axes, PANELS):
        for m, st in MONO_STYLE.items():
            r = sorted((x for x in rows if x["lam_const"] == 1 and x["lam_mono"] == m), key=lambda x: x["lam_task"])
            ts = [x["lam_task"] for x in r]
            ys = np.array([x[key]["mean"] for x in r]); es = np.array([x[key]["std"] for x in r])
            ax.errorbar(ts, ys, yerr=es, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"],
                        ms=st["ms"], mec="white", mew=0.5, capsize=1.5, elinewidth=0.7,
                        zorder=st["zorder"], clip_on=False, label=f"$\\lambda_{{mono}}$={m:g}")
            if m == CHOSEN_M:
                y0 = ys[ts.index(CHOSEN_T)]
                ax.scatter([CHOSEN_T], [y0], s=32, facecolor="none", edgecolor="black", lw=0.8, zorder=6, clip_on=False)
                chosen = (y0, vfmt.format(y0))
        ax.set_xscale("log")
        ticks = [0.1, 1, 10, 100]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0.07, 150)
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        if yticks:
            ax.set_yticks(yticks)
        if key in ("cpi_mape", "base_incons"):
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_title(title, fontweight="bold", pad=2)
        ax.yaxis.grid(True, linestyle=":", color="#C0C0C0", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        y0, txt = chosen
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.annotate(txt, (CHOSEN_T, y0), xytext=(CHOSEN_T, y0 + 0.13 * span), ha="center", va="bottom",
                    fontsize=7.5, color="black",
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.5, shrinkA=0, shrinkB=3))
    for ax in axes[2:]:
        ax.set_xlabel("$\\lambda_{task}$", labelpad=0.5)
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec="black", ms=5.5, mew=0.8))
    l.append("chosen ($\\lambda_{task}$=10)")
    plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0.25, rect=(0, 0, 1, 0.93))
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), borderaxespad=0,
               borderpad=0, columnspacing=1.2, handletextpad=0.4, handlelength=1.8)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUTPUT.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"Saved {OUTPUT} (+ .pdf)")


if __name__ == "__main__":
    main()
