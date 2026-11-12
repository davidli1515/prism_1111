#!/usr/bin/env python3
"""Quarter-column figure: time to evaluate one design, gem5 vs PRISM (feature construction + inference).
  gem5  : mean wall time per run of gem5_val4/runs/*/*/result.json ("elapsed"; 205 small-start validation runs).
  PRISM : gpu_timing/644.nab_s_n2000.txt  (feat (CPU) + gpu_batched, ms/design).
The gem5 bar is broken with a wavy cut. Style of plot_states_column.py.
Writes gpu_timing/fig_runtime_bar.{png,pdf}; refuses to overwrite."""
import os, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 2.5, 'ytick.major.size': 0,
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
})
C_GEM5, C_FEAT, C_INF = "#7F8C8D", "#2E86C1", "#E67E22"


def timing(path):
    d = {}
    for l in open(path):
        p = l.split()
        if l.startswith("feat (CPU)"):
            d["feat"] = float(p[3])
        elif p and p[0] == "gpu_batched":
            d["inf"] = float(p[2])
    return d["feat"], d["inf"]


def main():
    outs = [os.path.join(HERE, "gpu_timing", f"fig_runtime_bar.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    el = [json.load(open(f))["elapsed"] for f in glob.glob(os.path.join(HERE, "gem5_val4", "runs", "*", "*", "result.json"))]
    g5 = float(np.mean(el))                                   # s
    feat, inf = timing(os.path.join(HERE, "gpu_timing", "644.nab_s_n2000.txt"))   # ms
    tot = feat + inf
    print(f"gem5 {g5:.1f} s over {len(el)} runs; PRISM feat {feat:.3f} + inf {inf:.3f} = {tot:.2f} ms; x{1000*g5/tot:,.0f}")

    XMAX, CUT, H = 14.5, 11.0, 0.62
    fig, ax = plt.subplots(figsize=(1.75, 1.05))
    y_g, y_p = 1, 0
    ax.barh(y_g, XMAX, height=H, color=C_GEM5, edgecolor="black", lw=0.7, zorder=3, clip_on=False)
    # wavy cut through the gem5 bar
    yy = np.linspace(y_g - H / 2 - 0.08, y_g + H / 2 + 0.08, 60)
    xs = 0.25 * np.sin((yy - yy[0]) / (yy[-1] - yy[0]) * 2 * np.pi * 1.5)
    gap = 0.45
    ax.add_patch(Polygon(np.r_[np.c_[CUT + xs, yy], np.c_[CUT + gap + xs, yy][::-1]], closed=True,
                         facecolor="white", edgecolor="none", zorder=4))
    for o in (0, gap):
        ax.plot(CUT + o + xs, yy, color="black", lw=0.7, zorder=5)
    ax.text(CUT / 2, y_g, f"{g5:.0f} s", ha="center", va="center", color="white", fontweight="bold", fontsize=8, zorder=6)
    # PRISM: feature construction + inference
    ax.barh(y_p, feat, height=H, color=C_FEAT, edgecolor="none", zorder=3)
    ax.barh(y_p, inf, left=feat, height=H, color=C_INF, edgecolor="none", zorder=3)
    ax.barh(y_p, tot, height=H, color="none", edgecolor="black", lw=0.7, zorder=4)
    ax.text(feat / 2, y_p, f"Feature\n{feat:.2f} ms", ha="center", va="center", color="white", fontweight="bold",
            fontsize=6.5, linespacing=1.0, zorder=6)
    ax.annotate(f"Inference\n{inf:.2f} ms", xy=(tot + 0.05, y_p), xytext=(tot + 0.8, y_p), ha="left", va="center",
                fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.7, color="black", shrinkA=1, shrinkB=0), zorder=6)

    ax.set_xlim(0, XMAX); ax.set_ylim(-0.52, 1.52)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([y_g, y_p]); ax.set_yticklabels(["gem5", "PRISM"], fontweight="bold")
    ax.set_xlabel("Time per design (ms)", fontweight="bold", labelpad=2)
    ax.grid(True, axis="x", linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("black"); s.set_linewidth(0.8)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
