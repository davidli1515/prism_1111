#!/usr/bin/env python3
"""Quarter-column figure: total time to evaluate the 337k designs of small-start greedy search
(rows of results_small/<bench>/part_*.npz, 17 benchmarks) = count x measured per-design time.
Offline precomputation is excluded, as in Concorde.
  gem5  : mean wall time of gem5_val4/runs/*/*/result.json ("elapsed", 205 runs), single core.
  PRISM : no d3, gpu_timing/644.nab_s_nod3_n2000.txt -- Feature = per-design gather/normalise (CPU),
          Inference = gpu_batched (single A100).
Speedup printed after the (broken) gem5 bar. Style of plot_states_column.py.
Writes gpu_timing/fig_runtime_bar_online.{png,pdf}; refuses to overwrite."""
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


def main():
    outs = [os.path.join(HERE, "gpu_timing", f"fig_runtime_bar_online.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    g5 = float(np.mean([json.load(open(f))["elapsed"]
                        for f in glob.glob(os.path.join(HERE, "gem5_val4", "runs", "*", "*", "result.json"))]))
    d = {}
    for l in open(os.path.join(HERE, "gpu_timing", "644.nab_s_nod3_n2000.txt")):
        if l.startswith("feat gather+norm"):
            d["feat"] = float(l.split()[-1])
        elif l.startswith("model gpu_batched"):
            d["inf"] = float(l.split()[-1])
    n = sum(len(np.load(f)["cpi"]) for f in glob.glob(os.path.join(HERE, "results_small", "[0-9]*", "part_*.npz")))
    g5m = n * g5 / 60                                         # min
    feat, inf = n * d["feat"] / 6e4, n * d["inf"] / 6e4       # min
    tot = feat + inf; sp = g5m / tot
    print(f"{n} designs: gem5 {g5m/60:,.0f} h; PRISM feature {feat:.2f} + inference {inf:.2f} = {tot:.2f} min; x{sp:,.0f}")

    XMAX, GEND, CUT, H = 16.0, 10.3, 8.6, 0.62
    fig, ax = plt.subplots(figsize=(1.75, 1.05))
    y_g, y_p = 1, 0
    ax.barh(y_g, GEND, height=H, color=C_GEM5, edgecolor="black", lw=0.7, zorder=3)
    yy = np.linspace(y_g - H / 2 - 0.08, y_g + H / 2 + 0.08, 60)
    xs = 0.2 * np.sin((yy - yy[0]) / (yy[-1] - yy[0]) * 2 * np.pi * 1.5); gap = 0.4
    ax.add_patch(Polygon(np.r_[np.c_[CUT + xs, yy], np.c_[CUT + gap + xs, yy][::-1]], closed=True,
                         facecolor="white", edgecolor="none", zorder=4))
    for o in (0, gap):
        ax.plot(CUT + o + xs, yy, color="black", lw=0.7, zorder=5)
    ax.text(CUT / 2, y_g, f"{g5m/60/1000:.1f}k h", ha="center", va="center", color="white", fontweight="bold",
            fontsize=8, zorder=6)
    ax.text(GEND + 0.4, y_g, f"{sp/1000:,.0f}k$\\times$", ha="left", va="center", fontweight="bold", fontsize=7.5)

    ax.barh(y_p, feat, height=H, color=C_FEAT, edgecolor="none", zorder=3)
    ax.barh(y_p, inf, left=feat, height=H, color=C_INF, edgecolor="none", zorder=3)
    ax.barh(y_p, tot, height=H, color="none", edgecolor="black", lw=0.7, zorder=4)
    ax.text(feat / 2, y_p, f"Feature\n{feat:.1f} min", ha="center", va="center", color="white", fontweight="bold",
            fontsize=6, linespacing=1.0, zorder=6)
    ax.annotate(f"Inference\n{inf:.1f} min", xy=(tot + 0.05, y_p), xytext=(tot + 0.7, y_p), ha="left", va="center",
                fontsize=6, linespacing=1.0, arrowprops=dict(arrowstyle="-", lw=0.7, color="black", shrinkA=1, shrinkB=0),
                zorder=6)

    ax.set_xlim(0, XMAX); ax.set_ylim(-0.52, 1.52)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([y_g, y_p]); ax.set_yticklabels(["gem5", "PRISM"], fontweight="bold")
    ax.text(-0.04, (y_p + 0.52 - 0.33) / 2.04, "single A100", transform=ax.transAxes, ha="right", va="top", fontsize=6)
    ax.set_xlabel("Total time (min)", fontweight="bold", labelpad=2)
    ax.grid(True, axis="x", linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("black"); s.set_linewidth(0.8)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
