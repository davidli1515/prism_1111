#!/usr/bin/env python3
"""
Unseen-workload (leave-one-benchmark-out, window-isolated) results, one panel per
source-label ratio: CPI error vs. number of labeled target samples N for
Concorde, PRISM-V2 (supervised) and PRISM. Mean over 23 held-out benchmarks, seed 42.

Data: results_ood_lowlabel_grid/exp2_ood_all_1011.json (built by make_ood_all_json.py;
      results[source_ratio][method][N] = overall CPI MAPE over the 23 held-out benchmarks).
Output: figures/ood_lowlabel_panels.{png,pdf}
Usage: python3 plot_ood_lowlabel_panels.py [--data results_ood_lowlabel_grid/exp2_ood_all_1011.json]
                                           [--width 20] [--height 4.6] [--ncols 5]
                                           [--out figures/ood_lowlabel_panels.png]
"""
import os, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "ood_lowlabel_panels.png")

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 18, 'axes.labelsize': 21, 'axes.titlesize': 22,
    'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 20,
    'figure.dpi': 600, 'axes.linewidth': 1.2,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 5, 'ytick.major.size': 5,
})

DATA = os.path.join(HERE, "results_ood_lowlabel_grid", "exp2_ood_all_1011.json")
RATIOS = [("src5", "5%"), ("src20", "20%"), ("src50", "50%"), ("src80", "80%"), ("src100", "100%")]
NS = [256, 1024, 2048]
XS = list(range(len(NS)))              # evenly spaced categorical x positions
XLABELS = ["256", "1K", "2K"]
# (method key in the json, legend label, color, linestyle, marker, open marker)
VARIANTS = [("Concorde", "Concorde", '#7F8C8D', '--', 's', True),
            ("PRISM-V2", "PRISM-V2 (labeled only)", '#2E86C1', '-.', '^', True),
            ("PRISM", "PRISM", '#C0392B', '-', 'o', False)]
# value labels: (source ratio, method, N, text offset in points from the marker, leader arrow)
ANNOTATE = [("src20", "PRISM", 2048, (-38, -26), True),
            ("src50", "PRISM", 2048, (-38, -26), True),
            ("src50", "Concorde", 2048, (-24, 44), True),
            ("src80", "Concorde", 2048, (-38, -26), True)]
YPAD_BOTTOM = 0.26   # extra room below the lowest point (fraction of y range) in annotated panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--width", type=float, default=20)
    ap.add_argument("--height", type=float, default=4.6)
    ap.add_argument("--ncols", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = json.load(open(args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)))['results']

    nrows = int(np.ceil(len(RATIOS) / args.ncols))
    fig, axes = plt.subplots(nrows, args.ncols, figsize=(args.width, args.height * nrows), squeeze=False)
    axes = axes.ravel()
    print(f"{'source':6s} {'N':>5s} " + " ".join(f"{lab.split(' ')[0]:>9s}" for _, lab, *_ in VARIANTS))
    for i, (r, title) in enumerate(RATIOS):
        ax = axes[i]
        ys = {}
        for v, lab, c, ls, mk, hollow in VARIANTS:
            ys[v] = [res[r][v][str(n)] for n in NS]
            ax.plot(XS, ys[v], color=c, ls=ls, marker=mk, ms=9, lw=2.6,
                    mfc='white' if hollow else c, mew=2.2, zorder=4 if v == 'PRISM' else 3)
        color = {v: c for v, _, c, *_ in VARIANTS}
        notes = [a for a in ANNOTATE if a[0] == r]
        if notes:   # keep the labels inside the axes, clear of the x tick labels
            lo, hi = min(min(y) for y in ys.values()), max(max(y) for y in ys.values())
            ax.set_ylim(lo - YPAD_BOTTOM * (hi - lo), hi + 0.05 * (hi - lo))
        for _, v, n, (dx, dy), arrow in notes:
            ax.annotate(f"{res[r][v][str(n)]:.1f}", (XS[NS.index(n)], res[r][v][str(n)]),
                        xytext=(dx, dy), textcoords='offset points', ha='center', va='center',
                        fontsize=16, fontweight='bold', color=color[v], zorder=6,
                        bbox=dict(boxstyle='round,pad=0.22', fc=color[v], alpha=0.18, ec='none'),
                        arrowprops=dict(arrowstyle='-', color=color[v], lw=1.4, shrinkA=0, shrinkB=6)
                        if arrow else None)
        for k, n in enumerate(NS):
            print(f"{title:6s} {n:5d} " + " ".join(f"{ys[v][k]:9.2f}" for v, *_ in VARIANTS))
        ax.set_xticks(XS); ax.set_xticklabels(XLABELS)
        ax.set_xlim(-0.3, len(NS) - 0.7)
        ax.set_title(f"Source labels {title}", fontweight='bold')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
        if i % args.ncols == 0:
            ax.set_ylabel("Overall MAPE (%)", fontweight='bold')
        for s in ax.spines.values():
            s.set_visible(True); s.set_color('black'); s.set_linewidth(1.2)
    for ax in axes[len(RATIOS):]:
        ax.axis('off')
    last = range((nrows - 1) * args.ncols, len(RATIOS))
    mid = axes[last[len(last) // 2]]
    mid.set_xlabel("Labeled samples of the held-out benchmark (N)", fontweight='bold', labelpad=6)
    if len(last) % 2 == 0:   # even number of panels in the last row: center between the two middle ones
        mid.xaxis.set_label_coords(0.0 - plt.rcParams['figure.subplot.wspace'] / 2, -0.14)

    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=9, lw=2.6,
                      mfc='white' if hollow else c, mew=2.2, label=lab)
               for _, lab, c, ls, mk, hollow in VARIANTS]
    fig.legend(handles=handles, loc='upper center', ncol=len(VARIANTS), frameon=True,
               edgecolor='black', fancybox=False, bbox_to_anchor=(0.5, 1.0 + 0.02 / nrows))
    plt.tight_layout(rect=(0, 0, 1, 0.90 if nrows == 1 else 0.95), w_pad=1.0)
    fig.savefig(out, dpi=600, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), format='pdf', bbox_inches='tight', dpi=600)
    print(f"Saved {out} and .pdf")


if __name__ == "__main__":
    main()
