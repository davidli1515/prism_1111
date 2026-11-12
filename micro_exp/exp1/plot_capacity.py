#!/usr/bin/env python3
"""
Plot model capacity comparison ([256,128] vs [512,256]).
Edit paths below, then run: python plot_capacity.py
"""

import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# EDIT PATHS HERE — set to None to skip a variant
# ============================================================
OUTPUT = "results/capacity_183k.png"

CONCORDE_256 = "results/exp1_v2_run2/exp1_256_128_183k/concorde.json"
CONCORDE_512 = "results/exp1_v2_run2/exp1_512_256_183k/concorde.json"

FULL_256     = "results/exp1_v3_run2/exp1_256_128_183k/gem_full.json"
FULL_512     = "results/exp1_v3_run2/exp1_512_256_183k/gem_full.json"

CONST_256    = "results/exp1_v3_run2/exp1_256_128_183k/gem_const.json"
CONST_512    = "results/exp1_v3_run2/exp1_512_256_183k/gem_const.json"

VANILLA_256  = "results/exp1_v2_run2/exp1_256_128_183k/gem_vanilla.json"
VANILLA_512  = "results/exp1_v2_run2/exp1_512_256_183k/gem_vanilla.json"

NOD3_256     = None
NOD3_512     = None
# ============================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
})


def _style_ax(ax):
    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)


def _save(fig, path):
    fig.savefig(path, dpi=600, bbox_inches='tight')
    pdf_path = path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=600)
    print(f"  Saved: {path}")
    print(f"  Saved: {pdf_path}")


def load_mape(path):
    if path is None:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if "overall_mape" in data:
            return data["overall_mape"]
        elif "overall" in data:
            for k in data["overall"]:
                return data["overall"][k]
        return None
    except:
        print(f"  WARNING: cannot load {path}")
        return None


def main():
    VARIANTS = [
        ("Concorde",        CONCORDE_256, CONCORDE_512),
        ("PRISM Full",      FULL_256,     FULL_512),
        ("PRISM (L_const)", CONST_256,    CONST_512),
        ("PRISM-noD3",      NOD3_256,     NOD3_512),
        ("PRISM (vanilla)", VANILLA_256,  VANILLA_512),
    ]

    active = []
    for label, path_256, path_512 in VARIANTS:
        val_256 = load_mape(path_256)
        val_512 = load_mape(path_512)
        if val_256 is not None or val_512 is not None:
            active.append((label, val_256, val_512))
            s256 = f"{val_256:.2f}%" if val_256 else "N/A"
            s512 = f"{val_512:.2f}%" if val_512 else "N/A"
            print(f"  {label:<20} [256,128]={s256:>8}  [512,256]={s512:>8}")

    if not active:
        print("ERROR: No data."); sys.exit(1)

    n = len(active)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 2.0), 4.0))

    vals_256 = [a[1] if a[1] else 0 for a in active]
    vals_512 = [a[2] if a[2] else 0 for a in active]
    mask_256 = [a[1] is not None for a in active]
    mask_512 = [a[2] is not None for a in active]

    bars1 = ax.bar(x - w/2, vals_256, w, label="[256,128]",
                   color="#F39C12", alpha=0.85, edgecolor='black',
                   linewidth=0.4, zorder=3)
    bars2 = ax.bar(x + w/2, vals_512, w, label="[512,256]",
                   color="#2ECC71", alpha=0.85, edgecolor='black',
                   linewidth=0.4, zorder=3)

    for bars, vals, mask in [(bars1, vals_256, mask_256),
                              (bars2, vals_512, mask_512)]:
        for bar, val, m in zip(bars, vals, mask):
            if m and val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.05,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("MAPE (%)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([a[0] for a in active],
                       rotation=20, ha="right", fontsize=11, fontweight='bold')
    _style_ax(ax)

    legend = ax.legend(frameon=True, edgecolor='black', facecolor='white',
                       fancybox=False, framealpha=1.0, fontsize=15,
                       borderpad=0.8, handletextpad=0.5)
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    _save(fig, OUTPUT)
    plt.close("all")
    print("Done!")


if __name__ == "__main__":
    main()
