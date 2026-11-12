#!/usr/bin/env python3
"""
Plot model capacity scaling: [256,128] vs [512,256].
  Fig 1 (main): Concorde → V1 → V2 → PRISM (progression)
  Fig 2 (deep dive): V2 vs V2(L_const) vs V2(L_mono) (ablation)
Usage: python3 plot_exp1_scale.py
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 600, 'axes.linewidth': 1.0,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 4, 'ytick.major.size': 4,
})

# ============================================================
# Configure paths here
# Each entry: JSON path (str), direct number (float), or None
# ============================================================
BASE = "/scratch/gem5-TAO/micro_exp/exp1/results"
EXP4 = "/scratch/gem5-TAO/micro_exp/exp4/results_77_p90/exp4_18k_extra_unlabeled"

# --- Fig 1: Main progression ---
MAIN_VARIANTS = [
    {
        "label": "Concorde",
        "color": "#2E86C1",
        "s256": f"{BASE}/exp1_v2_74/exp1_256_128_183k/concorde.json",
        "s512": f"{BASE}/exp1_v2_74/exp1_512_256_183k/concorde.json",
    },
    {
        "label": "V1 (+MSCS)",
        "color": "#95A5A6",
        "s256": f"{BASE}/exp1_v2_74/exp1_256_128_183k/gem_vanilla.json",
        "s512": f"{BASE}/exp1_v2_74/exp1_512_256_183k/gem_vanilla.json",
    },
    {
        "label": "V2 (+Physics)",
        "color": "#F39C12",
        "s256": f"{BASE}/exp1_v3_76/exp1_256_128_183k/gem_full.json",
        "s512": f"{BASE}/exp1_v3_76/exp1_512_256_183k/gem_full.json",
    },
    {
        "label": "PRISM",
        "color": "#E67E22",
        "s256": None,  # no [256,128] PRISM result file exists
        "s512": f"{EXP4}/exp4_100pct.json",
    },
]

# --- Fig 2: Deep dive ablation ---
ABLATION_VARIANTS = [
    {
        "label": "V2 (full)",
        "color": "#F39C12",
        "s256": f"{BASE}/exp1_v3_76/exp1_256_128_183k/gem_full.json",
        "s512": f"{BASE}/exp1_v3_76/exp1_512_256_183k/gem_full.json",
    },
    {
        "label": "V2 (L_const)",
        "color": "#2ECC71",
        "s256": f"{BASE}/exp1_v3_76/exp1_256_128_183k/gem_const.json",
        "s512": f"{BASE}/exp1_v3_76/exp1_512_256_183k/gem_const.json",
    },
    {
        "label": "V2 (L_mono)",
        "color": "#CD6155",
        "s256": f"{BASE}/exp1_v3_76/exp1_256_128_183k/gem_mono.json",
        "s512": f"{BASE}/exp1_v3_76/exp1_512_256_183k/gem_mono.json",
    },
]

OUTPUT_MAIN = f"{BASE}/exp1_scale_main.png"
OUTPUT_ABLATION = f"{BASE}/exp1_scale_ablation.png"

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

def resolve_mape(val):
    if val is None: return None
    if isinstance(val, (int, float)): return float(val)
    if not os.path.exists(val): return None
    with open(val) as f: data = json.load(f)
    if "overall_mape" in data: return data["overall_mape"]
    if "overall" in data:
        for k in data["overall"]: return data["overall"][k]
    return None

def plot_capacity(variants, output, title, y_max=None):
    """Plot [256,128] vs [512,256] grouped bars for a list of variants."""
    vals_256, vals_512 = [], []
    for v in variants:
        m256 = resolve_mape(v["s256"])
        m512 = resolve_mape(v["s512"])
        print(f"  {v['label']:<16} [256,128]={f'{m256:.2f}%' if m256 else 'N/A':>8}  [512,256]={f'{m512:.2f}%' if m512 else 'N/A':>8}")
        vals_256.append(m256)
        vals_512.append(m512)

    n = len(variants)
    x = np.arange(n)
    w = 0.35

    # Auto y_max if not set
    all_vals = [v for v in vals_256 + vals_512 if v is not None]
    if y_max is None:
        y_max = max(all_vals) * 1.25 if all_vals else 10

    fig, ax = plt.subplots(figsize=(max(6, n * 2.0), 4.0))

    for i in range(n):
        v256 = vals_256[i] if vals_256[i] is not None else 0
        v512 = vals_512[i] if vals_512[i] is not None else 0
        m256 = vals_256[i] is not None
        m512 = vals_512[i] is not None

        # Clip at y_max
        v256_clip = min(v256, y_max) if m256 else 0
        v512_clip = min(v512, y_max) if m512 else 0

        ax.bar(x[i] - w/2, v256_clip, w,
               color="#F39C12", alpha=0.85 if m256 else 0,
               edgecolor='black', linewidth=0.4 if m256 else 0, zorder=3,
               label='[256,128]' if i == 0 else "")
        ax.bar(x[i] + w/2, v512_clip, w,
               color="#2ECC71", alpha=0.85 if m512 else 0,
               edgecolor='black', linewidth=0.4 if m512 else 0, zorder=3,
               label='[512,256]' if i == 0 else "")

        # Labels
        if m256 and v256 > 0:
            if v256 > y_max:
                wave_x = np.linspace(x[i] - w/2 - w*0.4, x[i] - w/2 + w*0.4, 20)
                wave_y = y_max - 0.08 + 0.06 * np.sin(wave_x * 40)
                ax.plot(wave_x, wave_y, color='white', linewidth=2.5, zorder=5)
                ax.plot(wave_x, wave_y, color='black', linewidth=1.0, zorder=6)
                ax.text(x[i] - w/2, y_max + 0.08, f"{v256:.1f}", ha="center", va="bottom",
                        fontsize=10, fontweight='bold', color='red')
            else:
                ax.text(x[i] - w/2, v256 + 0.08, f"{v256:.1f}", ha="center", va="bottom", fontsize=10)

        if m512 and v512 > 0:
            if v512 > y_max:
                wave_x = np.linspace(x[i] + w/2 - w*0.4, x[i] + w/2 + w*0.4, 20)
                wave_y = y_max - 0.08 + 0.06 * np.sin(wave_x * 40)
                ax.plot(wave_x, wave_y, color='white', linewidth=2.5, zorder=5)
                ax.plot(wave_x, wave_y, color='black', linewidth=1.0, zorder=6)
                ax.text(x[i] + w/2, y_max + 0.08, f"{v512:.1f}", ha="center", va="bottom",
                        fontsize=10, fontweight='bold', color='red')
            else:
                ax.text(x[i] + w/2, v512 + 0.08, f"{v512:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("MAPE (%)", fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([v["label"] for v in variants],
                       rotation=20, ha="right", fontsize=12, fontweight='bold')
    _style_ax(ax)

    legend = ax.legend(frameon=True, edgecolor='black', facecolor='white',
                       fancybox=False, framealpha=1.0, fontsize=11,
                       borderpad=0.8, handletextpad=0.5)
    legend.get_frame().set_linewidth(1.0)

    ax.set_ylim(0, y_max * 1.1)
    ax.set_xlim(-0.5, n - 0.5)
    plt.tight_layout()
    _save(fig, output)

    # Table
    print(f"\n  {'Variant':<16} {'[256,128]':>10} {'[512,256]':>10} {'Δ':>8}")
    print("  " + "-" * 44)
    for i, v in enumerate(variants):
        v256, v512 = vals_256[i], vals_512[i]
        s256 = f"{v256:.2f}%" if v256 else "N/A"
        s512 = f"{v512:.2f}%" if v512 else "N/A"
        delta = f"{v512-v256:+.2f}%" if (v256 and v512) else "—"
        print(f"  {v['label']:<16} {s256:>9} {s512:>9} {delta:>8}")

    plt.close("all")

def main():
    print("=== Fig 1: Main Progression ===")
    plot_capacity(MAIN_VARIANTS, OUTPUT_MAIN,
                  "Model Capacity: Accuracy Progression", y_max=5.0)

    print("\n=== Fig 2: Deep Dive Ablation ===")
    plot_capacity(ABLATION_VARIANTS, OUTPUT_ABLATION,
                  "Model Capacity: Loss Ablation", y_max=5.0)

    print("\nDone!")

if __name__ == "__main__":
    main()