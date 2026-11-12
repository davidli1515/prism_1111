#!/usr/bin/env python3
"""
Fig. 6 re-measurement (V2, 100% labels, 3873 Concorde features, 3 seeds; band = seed std).
x: epoch (log scale), first 1000 of the 1521 training epochs. Settings are (lam_task, lam_const, lam_mono).
  (a) cos(grad L_task, grad L_mono), (1, 1, 0.01) vs (1, 0, 0.01): differ only in L_const
  (b) cos(grad L_task, grad L_mono), (10, 1, 0) vs (0.1, 1, 0): L_mono not trained, only measured
  (c) cos(grad L_task, grad L_const), same two settings as (b)

Data: results/grad_cosine_nod3_1031.json (from collect_grad_cosine_nod3_1031.py)
Usage: python plot_grad_cosine_nod3_1031.py [--panels a --output results/grad_cosine_nod3_1031_ep1000_a.png]
"""
import os, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

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
MAX_EPOCH = 1000
SOLID =dict(color="#FF6F00", ls="-", lw=1.4, zorder=4)
DASHED = dict(color="#3987e5", ls="--", lw=1.2, zorder=3)
PANELS = [("(a) $\\cos(\\nabla L_{task}, \\nabla L_{mono})$", "cos_task_mono",
           [((1, 1, 0.01), "with $L_{const}$", SOLID), ((1, 0, 0.01), "without $L_{const}$", DASHED)]),
          ("(b) $\\cos(\\nabla L_{task}, \\nabla L_{mono})$", "cos_task_mono",
           [((10, 1, 0), "$\\lambda_{task}$=10", SOLID), ((0.1, 1, 0), "$\\lambda_{task}$=0.1", DASHED)]),
          ("(c) $\\cos(\\nabla L_{task}, \\nabla L_{const})$", "cos_task_const",
           [((10, 1, 0), "$\\lambda_{task}$=10", SOLID), ((0.1, 1, 0), "$\\lambda_{task}$=0.1", DASHED)])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "results", "grad_cosine_nod3_1031.json"))
    ap.add_argument("--output", default=os.path.join(HERE, "results", "grad_cosine_nod3_1031_ep1000.png"))
    ap.add_argument("--panels", default="abc", help="subset of panels to draw, e.g. a")
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"{args.output} exists, refusing to overwrite")
    rows = {(r["lam_task"], r["lam_const"], r["lam_mono"]): r for r in json.load(open(args.data))["rows"]}

    panels = [PANELS["abc".index(p)] for p in args.panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16 * len(panels) / 3 + 0.3 * (len(panels) < 3), 1.9),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (title, key, lines) in zip(axes, panels):
        ax.axhline(0, color="black", lw=0.7, zorder=2)
        for cfg, label, st in lines:
            r = rows[cfg]
            ep = np.array(r["epochs"])
            keep = (ep >= 1) & (ep <= MAX_EPOCH)               # epoch 0 (initialization) has no log x
            ep = ep[keep]
            m, s = np.array(r[key]["mean"])[keep], np.array(r[key]["std"])[keep]
            ax.fill_between(ep, m - s, m + s, color=st["color"], alpha=0.2, lw=0, zorder=st["zorder"] - 1)
            ax.plot(ep, m, label=label, **st)
        ax.set_xscale("log")
        ticks = [1, 10, 100, MAX_EPOCH]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(1, MAX_EPOCH)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_title(title.split(" ", 1)[1] if len(panels) == 1 else title, fontweight="bold", pad=4)
        ax.set_xlabel("Epoch")
        ax.yaxis.grid(True, linestyle=":", color="#C0C0C0", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False,
                  handlelength=1.8, columnspacing=1.2, borderpad=0)
    axes[0].set_ylabel("Gradient cosine")
    plt.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(args.output, dpi=600, bbox_inches="tight")
    fig.savefig(args.output.replace(".png", ".pdf"), format="pdf", bbox_inches="tight")
    print(f"Saved {args.output} (+ .pdf)")


if __name__ == "__main__":
    main()
