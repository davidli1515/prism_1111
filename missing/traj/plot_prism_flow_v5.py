#!/usr/bin/env python3
"""Half-column overview diagram of Section V (v5 = v3 with ACM section numbers 5.2, 5.3, 5.4): Characterize -> Analyze -> Guide along an arch
opening downwards, PRISM at the centre, section numbers under each box. Style of plot_states_column.py (Arial, black edges, no colour coding).
Writes writing/fig_prism_flow_v5.{png,pdf}; refuses to overwrite."""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'figure.dpi': 600})
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "writing")
W, H = 1.7, 1.5                                                   # inches; data units = inches
BOXES = {"Characterize": ((0.48, 0.36), 0.90, "§5.2"),
         "Analyze": ((W / 2, 1.27), 0.66, "§5.3"),
         "Guide": ((1.38, 0.36), 0.58, "§5.4")}
BH = 0.24


def edge(name, side):
    (x, y), w, _ = BOXES[name]
    return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + BH / 2), "b": (x, y - BH / 2)}[side]


def main():
    outs = [os.path.join(OUT, f"fig_prism_flow_v5.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    for name, ((x, y), w, sec) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x - w / 2, y - BH / 2), w, BH, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc="white", ec="black", lw=0.9, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=4)
        ax.text(x, y - BH / 2 - 0.035, sec, ha="center", va="top", fontsize=6.5, color="#606060", zorder=4)
    ax.add_patch(Circle((W / 2, 0.80), 0.19, fc="#E6E6E6", ec="none", zorder=1))
    ax.text(W / 2, 0.80, "PRISM", ha="center", va="center", fontsize=9, fontweight="bold", zorder=2)
    arrows = [((0.30, 0.36 + BH / 2), (edge("Analyze", "l")[0], 1.27), -0.38),
              ((edge("Analyze", "r")[0], 1.27), (1.50, 0.36 + BH / 2), -0.38)]
    for a, b, rad in arrows:
        ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=7,
                                     lw=0.9, color="black", shrinkA=1, shrinkB=1, zorder=2))
    (gx, gy), gw, _ = BOXES["Guide"]                         # architect icon: head + shoulders, on the box corner
    px, py, r = gx - gw / 2 + 0.01, gy + BH / 2 - 0.004, 0.045
    ax.add_patch(Wedge((px, py), 1.9 * r, 0, 180, fc="black", ec="white", lw=0.8, zorder=6))
    ax.add_patch(Circle((px, py + 2.45 * r), 0.95 * r, fc="black", ec="white", lw=0.8, zorder=6))
    os.makedirs(OUT, exist_ok=True)
    for p in outs:
        fig.savefig(p, dpi=600, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
