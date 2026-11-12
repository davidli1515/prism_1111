#!/usr/bin/env python3
"""
Plot gradient cosine similarity comparison:
  - Full scenario (L_task + L_const + L_mono)
  - Mono-only scenario (L_task + L_mono, no L_const)

Highlights how cos(task, mono) flips sign depending on L_const presence.
Usage: python3 plot_grad_cosine.py
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.dpi': 600, 'axes.linewidth': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
})

# ============================================================
# Config
# ============================================================
FULL_JSON = "/scratch/gem5-TAO/micro_exp/exp1/grad_cosine.json"
MONO_JSON = "/scratch/gem5-TAO/micro_exp/exp1/grad_cosine_mono.json"
OUTPUT = "/scratch/gem5-TAO/micro_exp/exp1/grad_cosine_compare.png"
# ============================================================

def load(path):
    with open(path) as f:
        return json.load(f)

def main():
    full = load(FULL_JSON)
    mono = load(MONO_JSON)

    ep_f = [h["epoch"] for h in full]
    ep_m = [h["epoch"] for h in mono]

    # cos(task, mono) in both scenarios
    ctm_full = [h["cos_task_mono"] for h in full]
    ctm_mono = [h["cos_task_mono"] for h in mono]

    # ============================================================
    # Plot: cos(task, mono) — full vs mono-only
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    # Shade conflict region first (background)
    ax.axhspan(-1, 0, alpha=0.07, color='red', zorder=0)
    ax.axhline(y=0, color='black', linewidth=1.0, linestyle='-', alpha=0.6, zorder=1)

    ax.plot(ep_f, ctm_full, '-', color='#2E86C1', linewidth=2.2, zorder=3,
            label=r'with $\mathcal{L}_{const}$ (cooperative)')
    ax.plot(ep_m, ctm_mono, '-', color='#CD6155', linewidth=2.2, zorder=3,
            label=r'without $\mathcal{L}_{const}$ (conflict)')

    ax.set_xlabel("Epoch", fontweight='bold')
    ax.set_ylabel(r"$\cos(\nabla \mathcal{L}_{task},\ \nabla \mathcal{L}_{mono})$", fontweight='bold')
    ax.set_title("L_const Determines Whether L_mono Cooperates or Conflicts",
                 fontweight='bold', pad=12)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(1, min(len(ep_f), len(ep_m)))

    # Conflict region label (right side, away from the lines)
    ax.text(0.235, 0.32, 'conflict region\n(cos < 0)', transform=ax.transAxes,
            ha='right', va='center', fontsize=10, color='#CD6155', style='italic')

    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color('black'); s.set_linewidth(1.0)

    # Legend at TOP (outside line area) to avoid occlusion
    legend = ax.legend(frameon=True, edgecolor='black', facecolor='white',
                       fancybox=False, framealpha=1.0, fontsize=12,
                       loc='upper center', bbox_to_anchor=(0.5, 0.99),
                       ncol=2, columnspacing=1.5, borderpad=0.7)
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    fig.savefig(OUTPUT, dpi=600, bbox_inches='tight')
    fig.savefig(OUTPUT.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT}")
    print(f"Saved: {OUTPUT.replace('.png', '.pdf')}")

    # Summary
    print(f"\n  with L_const:    mean cos = {np.mean(ctm_full):+.3f}  (conflict epochs: {sum(1 for c in ctm_full if c < -0.05)}/{len(ctm_full)})")
    print(f"  without L_const: mean cos = {np.mean(ctm_mono):+.3f}  (conflict epochs: {sum(1 for c in ctm_mono if c < -0.05)}/{len(ctm_mono)})")

    plt.close("all")
    print("\nDone!")

if __name__ == "__main__":
    main()