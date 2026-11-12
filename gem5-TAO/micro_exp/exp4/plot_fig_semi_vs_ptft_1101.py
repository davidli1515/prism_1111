#!/usr/bin/env python3
"""
Semi-supervised vs. PT+FT, original data, a small figure for the trend. Same style as
exp1/results/lambda_sensitivity_nod3_1031_v3.png, drawn at its printed size (place at 0.49\\linewidth).
The y-axis stops at Y_MAX: PT(L)+FT(U) at 5% labels is above it, its value is written next to the clipped line.

Data: figures_final/fig_semi_vs_ptft_1101.json (from collect_fig_semi_vs_ptft_1101.py)
Usage: python plot_fig_semi_vs_ptft_1101.py   (rewrites the figure)
"""
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'figure.dpi': 600, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 3, 'ytick.major.size': 3,
})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "figures_final", "fig_semi_vs_ptft_1101.json")
OUTPUT = os.path.join(HERE, "figures_final", "fig_semi_vs_ptft_1101.png")
STYLES = {"V3": dict(color="#1565C0", marker="o", lw=1.5, ms=2.8, zorder=5),
          "Type1": dict(color="#C2185B", marker="v", lw=1.0, ms=2.6, zorder=4),
          "Type2": dict(color="#0097A7", marker=">", lw=1.0, ms=2.6, zorder=4)}
LABELS = {"V3": "Semi", "Type1": "PT(L)+FT(U)", "Type2": "PT(U)+FT(L)"}
Y_MAX = 20


def main():
    curves = json.load(open(DATA))["curves"]
    fig, ax = plt.subplots(figsize=(1.67, 1.45))
    for key, st in STYLES.items():
        c = curves[key]
        ax.plot(c["pct"], c["overall_mape"], label=LABELS[key], markeredgewidth=0, **st)
    t1 = curves["Type1"]
    ax.text(14.5, 18.9, f"↑{t1['overall_mape'][t1['pct'].index(5)]:.1f}", fontsize=6, fontweight="bold",
            color=STYLES["Type1"]["color"], ha="left", va="center", zorder=6)
    ax.set_xticks([5, 20, 40, 60, 80, 100])
    ax.set_xlim(1, 104)
    ax.set_ylim(0, Y_MAX)
    ax.set_yticks(range(0, Y_MAX + 1, 5))
    ax.set_title("CPI MAPE (%)", fontweight="bold", pad=2)
    ax.set_xlabel("Labeled data (%)", labelpad=0.5)
    ax.yaxis.grid(True, linestyle=":", color="#C0C0C0", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, borderaxespad=0.2, borderpad=0, labelspacing=0.2, handlelength=1.3,
              handletextpad=0.4)
    plt.tight_layout(pad=0.1)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUTPUT.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"Saved {OUTPUT} (+ .pdf)")


if __name__ == "__main__":
    main()
