#!/usr/bin/env python3
"""
Fig. 2: per-benchmark CPI error (average + P90), original data, same style as
results/lambda_sensitivity_nod3_1031_v3.png. Drawn at its printed size, full text width (7.16 in, figure*).
Benchmarks sorted by Concorde average error, GeoMean (geometric mean over benchmarks) first, as in the original.

Data: results/fig2_per_benchmark_1102.json (from collect_fig2_per_benchmark_1102.py)
Usage: python plot_fig2_per_benchmark_1102.py   (rewrites the figure)
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'figure.dpi': 600, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 3, 'ytick.major.size': 3,
})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "results", "fig2_per_benchmark_1102.json")
OUTPUT = os.path.join(HERE, "results", "fig2_per_benchmark_1102.png")
# (key, legend name, avg color, P90 color); colors follow the entity in the label-efficiency figure
SERIES = [("Concorde", "Concorde", "#2E7D32", "#A5D6A7"),
          ("PRISM", "PRISM", "#FF6F00", "#FFCC80"),
          ("V1", "V1", "#AB47BC", "#E1BEE7")]
W = 0.8 / len(SERIES)


def geomean(v):
    return float(np.exp(np.mean(np.log(np.array(v) + 1e-8))))


def short(bm):
    return bm.replace("_r", "").replace("_s", "").split(".")[-1]


def main():
    V = json.load(open(DATA))["variants"]
    bms = sorted(V["Concorde"]["avg"], key=lambda b: V["Concorde"]["avg"][b])
    labels = ["GeoMean"] + [short(b) for b in bms]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.16, 1.9))
    for i, (key, name, c_avg, c_p90) in enumerate(SERIES):
        avg = [V[key]["avg"][b] for b in bms]
        p90 = [V[key]["p90"][b] for b in bms]
        avg, p90 = [geomean(avg)] + avg, [geomean(p90)] + p90
        print(f"  {name:<9} GeoMean avg={avg[0]:.2f}%  P90={p90[0]:.2f}%")
        xi = x + (i - (len(SERIES) - 1) / 2) * W
        ax.bar(xi, p90, W, color=c_p90, edgecolor="black", linewidth=0.3, zorder=2, label=f"{name} P90")
        ax.bar(xi, avg, W, color=c_avg, edgecolor="black", linewidth=0.3, zorder=3, label=f"{name} avg")
    ax.axvline(0.5, color="black", lw=0.7, ls="--", alpha=0.5, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(0, 16.5)
    ax.set_yticks(range(0, 17, 4))
    ax.set_title("CPI MAPE (%)", fontweight="bold", pad=2)
    ax.yaxis.grid(True, linestyle=":", color="#C0C0C0", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    order = [1, 0, 3, 2, 5, 4]                                 # avg before P90 for each model
    ax.legend([h[k] for k in order], [l[k] for k in order], loc="upper left", ncol=3, frameon=False, fontsize=9,
              borderaxespad=0.3, borderpad=0, columnspacing=1.2, labelspacing=0.25, handlelength=1.5,
              handleheight=0.9, handletextpad=0.4)
    plt.tight_layout(pad=0.1)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUTPUT.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"Saved {OUTPUT} (+ .pdf)")


if __name__ == "__main__":
    main()
