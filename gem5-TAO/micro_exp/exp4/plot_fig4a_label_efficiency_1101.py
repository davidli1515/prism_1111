#!/usr/bin/env python3
"""
Fig. 4(a): label efficiency across variants, original data, same style as
exp1/results/lambda_sensitivity_nod3_1031_v3.png. Drawn at its printed size, one 3.5 in column.

Data: figures_final/fig4a_label_efficiency_1101.json (from collect_fig4a_label_efficiency_1101.py)
Usage: python plot_fig4a_label_efficiency_1101.py   (rewrites the figure)
"""
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

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
DATA = os.path.join(HERE, "figures_final", "fig4a_label_efficiency_1101.json")
OUTPUT = os.path.join(HERE, "figures_final", "fig4a_label_efficiency_1101.png")
STYLES = {"Concorde": dict(color="#2E7D32", marker="s", lw=1.6, ms=3.2, zorder=4),
          "V1": dict(color="#AB47BC", marker="^", lw=0.8, ms=2.6, zorder=3),
          "V2": dict(color="#00ACC1", marker="D", lw=0.8, ms=2.2, zorder=3),
          "V3": dict(color="#1565C0", marker="o", lw=0.8, ms=2.6, zorder=3),
          "PRISM": dict(color="#FF6F00", marker="p", lw=1.6, ms=3.8, zorder=5)}
LABELS = {"Concorde": "Concorde", "V1": "V1: +MSCS", "V2": "V2: +Physics", "V3": "V3: +Semi-Supervised",
          "PRISM": "PRISM"}
REF = {"PRISM": dict(color="#FF6F00", ls="--"), "Concorde": dict(color="#2E7D32", ls=":")}
# value labels, (dx, dy, ha, va) in points from the point; the other curves dictate most of these
UP, DOWN = (0, 3, "center", "bottom"), (0, -3, "center", "top")
LEFT = (-3, 0, "right", "center")
LABEL_POS = {"Concorde": {5: LEFT, 15: (7, 4, "center", "bottom"), 20: (5, 2, "center", "bottom"), 30: DOWN,
                          40: (-4, -3, "center", "top"), 50: (6, 3, "center", "bottom"),
                          60: (0, 11, "center", "bottom"), 70: (0, 5, "center", "bottom"),
                          80: (0, 6, "center", "bottom"), 90: (0, 11, "center", "bottom"),
                          100: (0, 7, "center", "bottom")},
             "PRISM": {5: LEFT, 15: (-6, -3, "center", "top"), 20: (-2, -3, "center", "top"), 30: DOWN, 40: DOWN}}
PRISM_ROW_Y = 2.62                                             # from 50% up, in one row under the 100% lines
THRESHOLDS = (3, 4, 5)


def main():
    curves = json.load(open(DATA))["curves"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for t in THRESHOLDS:
        ax.axhline(t, color="#BDBDBD", ls="--", lw=0.8, zorder=1)
        ax.text(-6, t + 0.06, f"{t}%", fontsize=7.5, color="#616161", ha="left", va="bottom", zorder=6)
    halo = [pe.withStroke(linewidth=1.5, foreground="white")]
    for key, st in STYLES.items():
        c = curves[key]
        ax.plot(c["pct"], c["overall_mape"], label=LABELS[key], markeredgewidth=0, **st)
        if key in LABEL_POS:
            for x, y in zip(c["pct"], c["overall_mape"]):
                dx, dy, ha, va = LABEL_POS[key].get(x, (0, 0, "center", "top"))
                anchor = (x, y) if x in LABEL_POS[key] else (x, PRISM_ROW_Y)
                ax.annotate(f"{y:.1f}", anchor, textcoords="offset points", xytext=(dx, dy), ha=ha, va=va, fontsize=5,
                            fontweight="bold", color=st["color"], zorder=6, path_effects=halo)
    for key, st in REF.items():
        c = curves[key]
        ax.axhline(c["overall_mape"][c["pct"].index(100)], lw=0.9, alpha=0.8, zorder=2, label=f"{key} 100%", **st)
    xs = sorted({p for c in curves.values() for p in c["pct"]})
    ax.set_xticks(xs)
    ax.set_xlim(-7, 104)
    ax.set_ylim(2, 12)
    ax.set_yticks(range(2, 13))
    ax.set_title("CPI MAPE (%)", fontweight="bold", pad=2)
    ax.set_xlabel("Labeled data (%)", labelpad=0.5)
    ax.yaxis.grid(True, linestyle=":", color="#C0C0C0", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", facecolor="white", edgecolor="none", framealpha=1, fancybox=False, borderaxespad=0.3,
              borderpad=0.1, labelspacing=0.25, handlelength=1.8, handletextpad=0.5)
    plt.tight_layout(pad=0.1)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUTPUT.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"Saved {OUTPUT} (+ .pdf)")


if __name__ == "__main__":
    main()
