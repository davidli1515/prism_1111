#!/usr/bin/env python3
"""
Half-column version of the unseen-workload figure (no-D3 results: identical input features
for all methods): CPI error vs. labeled target samples N for Concorde, PRISM-V2 and PRISM.
Two rows of three cells: five source-label ratios r (value in each panel's upper-right
corner) and the legend in the sixth cell. Mean over 23 held-out benchmarks, seed 42.

Data: results_ood_lowlabel_grid_nod3_1025/exp2_ood_nod3_1031.json (make_ood_nod3_json.py)
Output: figures/ood_nod3_halfcol.{png,pdf}
Usage: python3 plot_ood_nod3_halfcol.py [--data <json>] [--width 7.0] [--height 4.7]
                                        [--out figures/ood_nod3_halfcol.png]
"""
import os, json, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "ood_nod3_halfcol.png")
DATA = os.path.join(HERE, "results_ood_lowlabel_grid_nod3_1025", "exp2_ood_nod3_1031.json")

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 15, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 15, 'figure.dpi': 600, 'axes.linewidth': 1.1,
    'xtick.major.width': 1.1, 'ytick.major.width': 1.1,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 4, 'ytick.major.size': 4,
})

RATIOS = [("src5", "5%"), ("src20", "20%"), ("src50", "50%"), ("src80", "80%"), ("src100", "100%")]
NS = [32, 256, 1024, 2048]
XS = list(range(len(NS)))
XLABELS = ["32", "256", "1K", "2K"]
# (method key in the json, legend label, color, linestyle, marker, open marker)
VARIANTS = [("Concorde", "Concorde", '#7F8C8D', '--', 's', True),
            ("PRISM-V2", "PRISM-V2\n(labeled only)", '#2E86C1', '-.', '^', True),
            ("PRISM", "PRISM", '#C0392B', '-', 'o', False)]
# value labels: (source ratio, method, N, text offset in points from the marker)
ANNOTATE = [("src20", "PRISM", 2048, (-30, -15)),
            ("src50", "PRISM", 2048, (-30, -15)),
            ("src50", "Concorde", 2048, (-14, 30)),
            ("src80", "Concorde", 2048, (-30, -15))]
YPAD_BOTTOM = 0.32   # extra room below the lowest point (fraction of y range) in annotated panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=4.7)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = json.load(open(args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)))['results']
    color = {v: c for v, _, c, *_ in VARIANTS}

    fig, axes = plt.subplots(2, 3, figsize=(args.width, args.height))
    axes = axes.ravel()
    for i, (r, title) in enumerate(RATIOS):
        ax = axes[i]
        ys = {v: [res[r][v][str(n)] for n in NS] for v, *_ in VARIANTS}
        for v, lab, c, ls, mk, hollow in VARIANTS:
            ax.plot(XS, ys[v], color=c, ls=ls, marker=mk, ms=6.5, lw=2.0,
                    mfc='white' if hollow else c, mew=1.7, zorder=4 if v == 'PRISM' else 3)
        notes = [a for a in ANNOTATE if a[0] == r]
        lo, hi = min(min(y) for y in ys.values()), max(max(y) for y in ys.values())
        ax.set_ylim(lo - (YPAD_BOTTOM if notes else 0.08) * (hi - lo), hi + 0.08 * (hi - lo))
        for _, v, n, (dx, dy) in notes:
            ax.annotate(f"{res[r][v][str(n)]:.1f}", (XS[NS.index(n)], res[r][v][str(n)]),
                        xytext=(dx, dy), textcoords='offset points', ha='center', va='center',
                        fontsize=13, fontweight='bold', color=color[v], zorder=6,
                        bbox=dict(boxstyle='round,pad=0.18', fc=color[v], alpha=0.18, ec='none'),
                        arrowprops=dict(arrowstyle='-', color=color[v], lw=1.1, shrinkA=0, shrinkB=4))
        ax.text(0.96, 0.93, f"r={title}", transform=ax.transAxes, ha='right', va='top',
                fontsize=15, fontweight='bold')
        ax.set_xticks(XS); ax.set_xticklabels(XLABELS)
        ax.set_xlim(-0.3, len(NS) - 0.7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
        ax.tick_params(pad=2)
        for s in ax.spines.values():
            s.set_color('black')

    axes[5].axis('off')
    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=6.5, lw=2.0,
                      mfc='white' if hollow else c, mew=1.7, label=lab)
               for _, lab, c, ls, mk, hollow in VARIANTS]
    axes[5].legend(handles=handles, loc='center', frameon=True, edgecolor='black', fancybox=False,
                   handlelength=1.9, labelspacing=0.5, borderpad=0.4, handletextpad=0.5)
    fig.supxlabel("Labeled samples of the held-out benchmark (N)", fontweight='bold',
                  fontsize=plt.rcParams['axes.labelsize'], y=0.0, va='bottom')
    fig.supylabel("Overall MAPE (%)", fontweight='bold',
                  fontsize=plt.rcParams['axes.labelsize'], x=0.0, ha='left')
    fig.subplots_adjust(left=0.098, right=0.995, top=0.995, bottom=0.135, wspace=0.27, hspace=0.15)
    fig.savefig(out, dpi=600, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), format='pdf', bbox_inches='tight', dpi=600)
    print(f"Saved {out} and .pdf")


if __name__ == "__main__":
    main()
