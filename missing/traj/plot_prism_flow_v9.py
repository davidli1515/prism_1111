#!/usr/bin/env python3
"""Half-column overview diagram of Section V (v9 = copy of v8): Characterize (two tools) -> Analyze & Guide (Detect -> Trace -> When ->
Where) -> Resolve along an arch, 'Diagnosis' on the hand-off arrow
opening downwards, PRISM at the centre, section numbers under each box. Style of plot_states_column.py (Arial, black edges, no colour coding).
Writes writing/fig_prism_flow_v9.{png,pdf}; refuses to overwrite."""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'figure.dpi': 600, 'mathtext.fontset': 'dejavusans'})
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "writing")
W, H = 1.7, 1.5                                                   # inches; data units = inches
BOXES = {"Characterize": ((0.46, 0.37), 0.88, "§5.2"),
         "Analyze & Guide": ((W / 2, 1.22), 1.42, "§5.3"),
         "Resolve": ((1.38, 0.37), 0.60, "§5.4")}
BH = 0.31
YA = BOXES["Analyze & Guide"][0][1]


def edge(name, side):
    (x, y), w, _ = BOXES[name]
    return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + BH / 2), "b": (x, y - BH / 2)}[side]


def main():
    outs = [os.path.join(OUT, f"fig_prism_flow_v9.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    for name, ((x, y), w, sec) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x - w / 2, y - BH / 2), w, BH, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc="white", ec="black", lw=0.9, zorder=3))
        sub = {"Characterize": "Two tools", "Analyze & Guide": "Detect\u2192Trace\u2192When\u2192Where"}.get(name)
        if sub:                                             # second line, regular weight: tools of 5.2 / steps of 5.3
            ax.text(x, y + 0.055, name, ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=4)
            ax.text(x, y - 0.068, sub, ha="center", va="center", fontsize=6.4, zorder=4)
        else:
            ax.text(x, y, name, ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=4)
        ax.text(x, y - BH / 2 - 0.03, sec, ha="center", va="top", fontsize=7.5, color="#505050", zorder=4)
    ax.add_patch(Circle((W / 2, 0.76), 0.15, fc="#E6E6E6", ec="none", zorder=1))
    ax.text(W / 2, 0.76, "PRISM", ha="center", va="center", fontsize=8, fontweight="bold", zorder=2)
    ax.text(1.615, 0.80, "Diagnosis", ha="center", va="center", fontsize=7, rotation=270, zorder=4)
    yb, yt = YA - BH / 2, 0.37 + BH / 2                     # bottom of Analyze & Guide, top of the lower boxes
    arrows = [((0.33, yt), (0.27, yb), -0.3), ((1.43, yb), (1.47, yt), -0.3)]
    for a, b, rad in arrows:
        ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=7,
                                     lw=0.9, color="black", shrinkA=1, shrinkB=1, zorder=2))
    (gx, gy), gw, _ = BOXES["Resolve"]                         # architect icon: head + shoulders, on the box corner
    px, py, r = gx - gw / 2 + 0.01, gy + BH / 2 - 0.004, 0.045
    ax.add_patch(Wedge((px, py), 1.9 * r, 0, 180, fc="black", ec="white", lw=0.8, zorder=6))
    ax.add_patch(Circle((px, py + 2.45 * r), 0.95 * r, fc="black", ec="white", lw=0.8, zorder=6))
    (cx, cy), cw, _ = BOXES["Characterize"]                  # toolbox icon: body + lid line + latch + handle
    tw, th, tx, ty = 0.17, 0.095, cx - cw / 2 + 0.17 / 2 + 0.012, cy + BH / 2 - 0.004
    ax.add_patch(FancyBboxPatch((tx - tw / 2, ty), tw, th, boxstyle="round,pad=0,rounding_size=0.012",
                                fc="black", ec="white", lw=0.8, zorder=6))
    ax.plot([tx - tw / 2 + 0.006, tx + tw / 2 - 0.006], [ty + 0.62 * th] * 2, color="white", lw=0.8,
            solid_capstyle="butt", zorder=7)
    ax.add_patch(FancyBboxPatch((tx - 0.016, ty + 0.62 * th - 0.016), 0.032, 0.032,
                                boxstyle="round,pad=0,rounding_size=0.005", fc="white", ec="none", zorder=7))
    hw, hh = 0.075, 0.04
    ax.add_patch(FancyBboxPatch((tx - hw / 2, ty + th - 0.01), hw, hh + 0.01, boxstyle="round,pad=0,rounding_size=0.014",
                                fc="black", ec="white", lw=0.8, zorder=5))
    ax.add_patch(FancyBboxPatch((tx - hw / 2 + 0.017, ty + th - 0.01), hw - 0.034, hh + 0.01 - 0.017,
                                boxstyle="round,pad=0,rounding_size=0.006", fc="white", ec="none", zorder=5.5))
    os.makedirs(OUT, exist_ok=True)
    for p in outs:
        fig.savefig(p, dpi=600, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
