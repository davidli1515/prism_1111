#!/usr/bin/env python3
"""
Fig. 6: cos(grad L_task, grad L_mono), (1, 1, 0.01) vs (1, 0, 0.01), which differ only in L_const
(V2, 100% labels, 3873 Concorde features, 3 seeds; band = seed std; first 1000 of the 1521 epochs).
Same style as results/lambda_sensitivity_nod3_1031_v3.png, drawn at its printed size (half of a 3.4 in
column), so fonts match when placed at 0.49\\linewidth next to the loss ablation figure.

Data: results/grad_cosine_nod3_1031.json (from collect_grad_cosine_nod3_1031.py)
Usage: python plot_grad_cosine_single_1101.py   (rewrites the figure)
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

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
DATA = os.path.join(HERE, "results", "grad_cosine_nod3_1031.json")
OUTPUT = os.path.join(HERE, "results", "grad_cosine_nod3_1031_ep1000_single.png")
MAX_EPOCH = 1000
LINES = [((1, 1, 0.01), "with $L_{const}$", dict(color="#FF6F00", ls="-", lw=1.2, zorder=4)),
         ((1, 0, 0.01), "without", dict(color="#3987e5", ls="--", lw=1.1, zorder=3))]


def main():
    rows = {(r["lam_task"], r["lam_const"], r["lam_mono"]): r for r in json.load(open(DATA))["rows"]}
    fig, ax = plt.subplots(figsize=(1.67, 1.45))
    ax.axhline(0, color="black", lw=0.7, zorder=2)
    for cfg, label, st in LINES:
        r = rows[cfg]
        ep = np.array(r["epochs"])
        keep = (ep >= 1) & (ep <= MAX_EPOCH)                   # epoch 0 (initialization) has no log x
        m, s = np.array(r["cos_task_mono"]["mean"])[keep], np.array(r["cos_task_mono"]["std"])[keep]
        ax.fill_between(ep[keep], m - s, m + s, color=st["color"], alpha=0.2, lw=0, zorder=st["zorder"] - 1)
        ax.plot(ep[keep], m, label=label, **st)
    ax.set_xscale("log")
    ticks = [1, 10, 100, MAX_EPOCH]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(1, MAX_EPOCH)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1, 0, 1])
    ax.set_title("$\\cos(\\nabla L_{task}, \\nabla L_{mono})$", fontweight="bold", pad=2)
    ax.set_xlabel("Epoch", labelpad=0.5)
    ax.yaxis.grid(True, linestyle=":", color="#C0C0C0", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    plt.tight_layout(pad=0.1, rect=(0, 0, 1, 0.89))
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), borderaxespad=0,
               borderpad=0, columnspacing=0.8, handletextpad=0.4, handlelength=1.5)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUTPUT.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"Saved {OUTPUT} (+ .pdf)")


if __name__ == "__main__":
    main()
