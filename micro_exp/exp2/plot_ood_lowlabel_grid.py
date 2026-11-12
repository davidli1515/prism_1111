#!/usr/bin/env python3
"""
Unseen-workload (leave-one-benchmark-out, window-isolated) results:
CPI error vs. number of labeled target samples N, for Concorde and PRISM,
one color per source-label ratio. Mean over 23 held-out benchmarks, seed 42.

Data: results_ood_lowlabel_grid (+ Concorde src5/src20 N<=1024 reused from
results_ood_lowlabel / results_ood_lowlabel23, identical split).
Output: figures/ood_lowlabel_grid.{png,pdf}
Usage: python3 plot_ood_lowlabel_grid.py [--width 9] [--height 6.2] [--out figures/ood_lowlabel_grid.png]
"""
import os, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "ood_lowlabel_grid.png")

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16, 'axes.labelsize': 20, 'axes.titlesize': 20,
    'xtick.labelsize': 17, 'ytick.labelsize': 17, 'legend.fontsize': 15,
    'figure.dpi': 600, 'axes.linewidth': 1.2,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 5, 'ytick.major.size': 5,
})

ALL = sorted(f[:-4] for f in os.listdir('/data/micro_traindata_packed/gem') if f.endswith('.npy'))
OLD = {"602.gcc_s", "605.mcf_s", "607.cactuBSSN_s", "638.imagick_s", "508.namd_r", "snappy", "re2", "657.xz_s"}
RATIOS = [("src5", "5%"), ("src20", "20%"), ("src50", "50%"), ("src80", "80%"), ("src100", "100%")]
COLORS = ['#C0392B', '#E67E22', '#27AE60', '#2E86C1', '#7D3C98']
NS = [256, 1024, 2048]


def path(r, v, t, n):
    if v == 'concorde' and r in ('src5', 'src20') and n in (256, 1024):
        d = 'results_ood_lowlabel' if t in OLD else 'results_ood_lowlabel23'
    else:
        d = 'results_ood_lowlabel_grid'
    return os.path.join(HERE, d, r, v, t, f"n{n}_s42.json")


def mean_mape(r, v, n):
    vals = [json.load(open(path(r, v, t, n)))['target_mape'] for t in ALL]
    assert len(vals) == len(ALL), (r, v, n)
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=9)
    ap.add_argument("--height", type=float, default=6.2)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    print(f"{'source':6s} " + " ".join(f"{'N='+str(n):>16s}" for n in NS) + "   (Concorde / PRISM)")
    for (r, lab), c in zip(RATIOS, COLORS):
        con = [mean_mape(r, 'concorde', n) for n in NS]
        pri = [mean_mape(r, 'prism', n) for n in NS]
        ax.plot(NS, con, color=c, ls='--', marker='s', ms=8, lw=2.2, mfc='white', mew=2, zorder=3)
        ax.plot(NS, pri, color=c, ls='-', marker='o', ms=8, lw=2.6, zorder=4)
        print(f"{lab:6s} " + " ".join(f"{a:7.2f} / {b:6.2f}" for a, b in zip(con, pri)))

    ax.set_xscale('log', base=2)
    ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
    ax.minorticks_off()
    ax.set_xlabel("Labeled samples of the held-out benchmark (N)", fontweight='bold')
    ax.set_ylabel("CPI Error (%)", fontweight='bold')
    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color('black'); s.set_linewidth(1.2)

    color_handles = [Line2D([], [], color=c, lw=3, label=lab) for (_, lab), c in zip(RATIOS, COLORS)]
    style_handles = [Line2D([], [], color='black', ls='-', marker='o', ms=8, lw=2.6, label='PRISM'),
                     Line2D([], [], color='black', ls='--', marker='s', ms=8, lw=2.2, mfc='white',
                            mew=2, label='Concorde')]
    kw = dict(frameon=True, edgecolor='black', facecolor='white', fancybox=False, framealpha=1.0)
    leg1 = ax.legend(handles=color_handles, title="Source labels", loc='upper right',
                     title_fontsize=15, **kw)
    leg1.get_frame().set_linewidth(1.0)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=style_handles, loc='upper center', **kw)
    leg2.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), format='pdf', bbox_inches='tight', dpi=600)
    print(f"Saved {out} and .pdf")


if __name__ == "__main__":
    main()
