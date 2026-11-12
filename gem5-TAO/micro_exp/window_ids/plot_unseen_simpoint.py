#!/usr/bin/env python3
"""
Per-benchmark CPI error on the original test split vs. on unseen simpoints
(simpoints absent from 18k, trace signature verified not to match any 18k window).
Style follows exp1/plot_error_breakdown.py.
Input:  unseen_simpoint/eval_unseen_simpoint.json (from eval_unseen_simpoint.py)
Usage:  python3 plot_unseen_simpoint.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 600, 'axes.linewidth': 1.0,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 4, 'ytick.major.size': 4,
})

COLOR_PRISM = '#E67E22';       COLOR_PRISM_LIGHT = '#F5CBA7'
COLOR_CONCORDE = '#2E86C1';    COLOR_CONCORDE_LIGHT = '#AED6F1'

# ============================================================
# Configure here
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_JSON = os.path.join(HERE, "unseen_simpoint/eval_unseen_simpoint.json")
CONCORDE_KEY = "exp4/results_77/exp4_18k_concorde/checkpoints/concorde_100pct.pt"
PRISM_KEY    = "exp1/results/exp1_v3_76/exp1_512_256_183k/gem_full.pt"
PRISM_LABEL  = "PRISM"
# Benchmarks whose extra 40k simpoints are genuinely new (0% trace-signature match
# with 18k); cactuBSSN: 75% new rows (marked *).
BENCHMARKS = ["508.namd_r", "638.imagick_s", "607.cactuBSSN_s", "526.blender_r",
              "623.xalancbmk_s", "631.deepsjeng_s", "641.leela_s"]
SHORT = {"607.cactuBSSN_s": "cactuBSSN*"}
OUTPUT = os.path.join(HERE, "unseen_simpoint/fig_unseen_simpoint.png")
# ============================================================


def _style_ax(ax):
    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
    ax.xaxis.grid(False)
    for s in ax.spines.values(): s.set_visible(True); s.set_color('black'); s.set_linewidth(1.0)

def _save(fig, path):
    fig.savefig(path, dpi=600, bbox_inches='tight')
    pdf = path.replace(".png", ".pdf")
    fig.savefig(pdf, format='pdf', bbox_inches='tight', dpi=600)
    print(f"  Saved: {path}\n  Saved: {pdf}")


def main():
    models = json.load(open(EVAL_JSON))["models"]
    con, pri = models[CONCORDE_KEY], models[PRISM_KEY]
    get = lambda m, k: [m[k][b] for b in BENCHMARKS]
    con_in, con_un = get(con, "per_benchmark_in_dist"), get(con, "per_benchmark_unseen")
    pri_in, pri_un = get(pri, "per_benchmark_in_dist"), get(pri, "per_benchmark_unseen")

    labels = ["Avg"] + [SHORT.get(b, b.split(".")[1].split("_")[0]) for b in BENCHMARKS]
    series = [[np.mean(s)] + s for s in (con_in, con_un, pri_in, pri_un)]
    n = len(labels); x = np.arange(n); w = 0.8 / 4

    fig, ax = plt.subplots(figsize=(max(8, n * 1.0), 4.0))
    specs = [(-1.5, series[0], COLOR_CONCORDE_LIGHT), (-0.5, series[1], COLOR_CONCORDE),
             (0.5, series[2], COLOR_PRISM_LIGHT),     (1.5, series[3], COLOR_PRISM)]
    for off, vals, c in specs:
        bars = ax.bar(x + off * w, vals, w, color=c, edgecolor='black', linewidth=0.3, zorder=3)
    for off, vals in ((-0.5, series[1]), (1.5, series[3])):        # annotate unseen bars
        for xi, v in zip(x + off * w, vals):
            ax.text(xi, v + 0.6, f"{v:.1f}", ha='center', va='bottom', fontsize=7, rotation=90, zorder=5)
    ax.axvline(x=0.5, color='black', linewidth=0.8, linestyle='--', zorder=4, alpha=0.5)
    ax.set_ylabel("Average CPI Error (%)", fontweight='bold')
    ax.set_title("Seen vs. Unseen Simpoints", fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11, fontweight='bold')
    ax.get_xticklabels()[0].set_fontsize(13); ax.get_xticklabels()[0].set_color('red')
    ax.set_ylim(0, max(max(s) for s in series) * 1.18)
    _style_ax(ax)
    handles = [Patch(facecolor=COLOR_CONCORDE_LIGHT, edgecolor='black', label='Concorde (seen simpoint)'),
               Patch(facecolor=COLOR_CONCORDE, edgecolor='black', label='Concorde (unseen simpoint)'),
               Patch(facecolor=COLOR_PRISM_LIGHT, edgecolor='black', label=f'{PRISM_LABEL} (seen simpoint)'),
               Patch(facecolor=COLOR_PRISM, edgecolor='black', label=f'{PRISM_LABEL} (unseen simpoint)')]
    ax.legend(handles=handles, loc='upper right', frameon=True, edgecolor='black', facecolor='white',
              fancybox=False, framealpha=1.0, fontsize=10, borderpad=0.8,
              handletextpad=0.5).get_frame().set_linewidth(1.0)
    ax.set_xlim(-0.5, n - 0.5)
    plt.tight_layout(); _save(fig, OUTPUT)

    print(f"\n  {'Benchmark':<18} {'Con seen':>9} {'Con unseen':>11} {'PRISM seen':>11} {'PRISM unseen':>13}")
    for i, l in enumerate(labels):
        print(f"  {l:<18} {series[0][i]:>9.2f} {series[1][i]:>11.2f} {series[2][i]:>11.2f} {series[3][i]:>13.2f}")


if __name__ == "__main__":
    main()
