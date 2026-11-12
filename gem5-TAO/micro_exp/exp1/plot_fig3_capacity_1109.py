#!/usr/bin/env python3
"""
Fig. 3: model capacity ablation, original data, same style as results/loss_ablation_nod3_1101.png (same colors
for the two hidden sizes). Drawn at its printed size, half of a 3.5 in column (place at 0.49\\linewidth).

Data: results/fig3_capacity_plotdata_1109.json (plot values, one decimal; V2 [512,256] = 3.0, aligned with
the chosen point of the sensitivity figure, lam (10, 1, 0) runs in results/lambda_sweep_nod3_1026, mean 2.951)
Usage: python plot_fig3_capacity_1109.py   (rewrites the figure)
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
DATA = os.path.join(HERE, "results", "fig3_capacity_plotdata_1109.json")
OUTPUT = os.path.join(HERE, "results", "fig3_capacity_1109.png")
SIZES = [("s256", "[256,128]", "#FF6F00"), ("s512", "[512,256]", "#3987e5")]
Y_MAX = 5.0
W = 0.4


def main():
    d = json.load(open(DATA))
    rows = [{"label": lab, "s256": {"value": a}, "s512": {"value": b}}
            for lab, a, b in zip(d["x"], d["[256,128]"], d["[512,256]"])]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(1.67, 1.45))
    for j, (key, name, color) in enumerate(SIZES):
        xs = x + (j - 0.5) * W
        vals = [r[key]["value"] for r in rows]
        ax.bar(xs, vals, W, color=color, edgecolor="black", linewidth=0.4, zorder=3, label=name)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.08, f"{v:.1f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_title("CPI MAPE (%)", fontweight="bold", pad=2)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=30, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_ylim(0, Y_MAX * 1.18)
    ax.set_yticks(range(int(Y_MAX) + 1))
    ax.yaxis.grid(True, linestyle=":", color="#C0C0C0", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    plt.tight_layout(pad=0.1, rect=(0, 0, 1, 0.89))
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), borderaxespad=0,
               borderpad=0, columnspacing=0.8, handletextpad=0.4, handlelength=1.2)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUTPUT.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"Saved {OUTPUT} (+ .pdf)")


if __name__ == "__main__":
    main()
