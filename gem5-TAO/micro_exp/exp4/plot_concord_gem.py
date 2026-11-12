"""
Plot Label Efficiency: 11 Targeted Comparisons

Variants:
  PRISM    = pretrain (physics only) -> finetune (full loss)
  Concorde = scalar CPI baseline
  V1       = Concorde + MSCS target (L_task only)
  V2       = Concorde + MSCS + physics loss (labeled only)
  V3       = Concorde + MSCS + physics + semi-supervised

Figures:
  1. Concorde vs V1         — Value of MSCS representation (D1)
  2. V1 vs V2               — Value of physics loss (D2)
  3. V2 vs V3               — Value of unlabeled data
  4. V3 vs PRISM              — Joint training vs pretrain+finetune
  5. PRISM vs Concorde         — Main result (paper figure)
  6. Concorde->V1->V2->V3   — Incremental ablation
  7. V2 + V1 + Concorde     — Physics value over baselines
  8. PRISM + V2 + Concorde    — PRISM vs supervised baselines
  9. All variants            — Full overview
  10. CON + V1 + V2 + V3    — Ablation without PRISM
  11. PRISM + V3 + Concorde   — With unlabeled-only baseline (broken y-axis)

Usage:
    python plot_concord_gem.py --dataset 40k
    python plot_concord_gem.py --dataset 18k
"""

import json, os, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Nature/Science publication-quality aesthetics
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'figure.dpi': 600,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
})

SKIP_RATIOS = {0.10}

EXP1_PATHS = {
    "18k": {
        "gem_full": "/scratch/gem5-TAO/micro_exp/exp1/results/exp1_v2/exp1_512_256_183k/gem_const.json",
        "concorde": "/scratch/gem5-TAO/micro_exp/exp1/results/exp1_v2/exp1_256_128_183k/concorde.json",
    },
    "40k": {
        "gem_full": "/scratch/gem5-TAO/micro_exp/exp1/results/exp1_v2/exp1_512_256/gem_full.json",
        "concorde": "/scratch/gem5-TAO/micro_exp/exp1/results/exp1_v2/exp1_512_256/concorde.json",
    },
}

PRETRAIN_PATHS = {
    "18k": "/scratch/gem5-TAO/micro_exp/exp5/results_pretrain_18k",
    "40k": "/scratch/gem5-TAO/micro_exp/exp5",
}

DATASET_TITLES = {"18k": "183k", "40k": "408k"}

CURVE_STYLES = {
    "GEM": {"color": "#FF6F00", "marker": "p", "lw": 2.5, "ms": 7},
    "CON": {"color": "#2E7D32", "marker": "s", "lw": 2.0, "ms": 6},
    "V1":  {"color": "#E91E63", "marker": "^", "lw": 2.0, "ms": 6},
    "V2":  {"color": "#455A64", "marker": "D", "lw": 2.0, "ms": 6},
    "V3":  {"color": "#1565C0", "marker": "o", "lw": 2.0, "ms": 6},
    "V3X": {"color": "#7B1FA2", "marker": "*", "lw": 2.0, "ms": 7},
}

CURVE_LABELS = {
    "GEM": "V3: +Pre-Train+Fine-Tune",
    "CON": "Concorde",
    "V1":  "V1: +MSCS",
    "V2":  "V2: +Physics",
    "V3":  "PRISM (semi-supervised)",
    "V3X": "PRISM + extra unlabeled",
}


# ============================================================
# Data Loading
# ============================================================

def load_results(result_dir):
    results = {}
    if not os.path.isdir(result_dir):
        return results
    for fname in sorted(os.listdir(result_dir)):
        if fname.startswith("exp4_") and fname.endswith(".json"):
            with open(os.path.join(result_dir, fname)) as f:
                data = json.load(f)
            ratio = data["labeled_ratio"]
            if ratio in SKIP_RATIOS:
                continue
            results[ratio] = data
    return results


def load_pretrain_results(pretrain_dir, dataset):
    results = {}
    if not os.path.isdir(pretrain_dir):
        return results
    for fname in sorted(os.listdir(pretrain_dir)):
        if not fname.endswith(".json"):
            continue
        if dataset == "40k" and not fname.startswith("exp5_"):
            continue
        if dataset == "18k" and not (fname.startswith("exp4_") and "18k" in fname):
            continue
        with open(os.path.join(pretrain_dir, fname)) as f:
            raw = json.load(f)
        data = raw[0] if isinstance(raw, list) else raw
        ratio = data.get("labeled_ratio", data.get("ratio"))
        mape = data.get("pretrain_ft", data.get("gem_pretrain_ft"))
        if ratio is None or mape is None:
            continue
        if ratio in SKIP_RATIOS:
            continue
        results[ratio] = {"overall_mape": mape}
    return results


def load_exp1_baselines(dataset):
    paths = EXP1_PATHS[dataset]
    with open(paths["gem_full"]) as f:
        gem_raw = json.load(f)
    with open(paths["concorde"]) as f:
        con_raw = json.load(f)
    return gem_raw["overall_mape"], con_raw["overall_mape"]


# ============================================================
# Plotting Helpers
# ============================================================

def plot_curves(ax, curve_keys, all_data):
    for key in curve_keys:
        res = all_data[key]
        if not res:
            continue
        style = CURVE_STYLES[key]
        label = CURVE_LABELS[key]
        ratios = sorted(res.keys())
        pcts = [r * 100 for r in ratios]
        mapes = [res[r]["overall_mape"] for r in ratios]
        ax.plot(pcts, mapes, f"{style['marker']}-",
                color=style["color"], markersize=style["ms"],
                linewidth=style["lw"], label=label, zorder=3)
        for x, y in zip(pcts, mapes):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), fontsize=7, ha="center",
                        color=style["color"], fontweight="bold")


def setup_ax(ax, title, curve_keys, all_data, gem100=None, con100=None):
    all_pcts = set()
    for key in curve_keys:
        all_pcts.update(r * 100 for r in all_data[key].keys())
    all_pcts = sorted(all_pcts)

    if gem100 is not None:
        ax.axhline(y=gem100, color="#FF6F00", linestyle="--", linewidth=1.5,
                   alpha=0.7, label=f"PRISM 100% ({gem100:.1f}%)")
    if con100 is not None:
        ax.axhline(y=con100, color="#2E7D32", linestyle="--", linewidth=1.5,
                   alpha=0.7, label=f"Concorde 100% ({con100:.1f}%)")

    ax.set_xlabel("Labeled Data (%)", fontweight="bold")
    ax.set_ylabel("Overall MAPE (%)", fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(all_pcts)
    ax.set_xticklabels([f"{int(x)}%" for x in all_pcts], rotation=45, ha="right")

    # Publication-quality grid and spines
    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    # Legend with border
    legend = ax.legend(loc="upper right", frameon=True, edgecolor='black',
                       facecolor='white', fancybox=False, framealpha=1.0)
    if legend:
        legend.get_frame().set_linewidth(1.0)

    ax.set_ylim(bottom=0)


def save_fig(fig, path):
    plt.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    pdf_path = path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {path} + {pdf_path}")


def make_fig(keys, all_data, title, path, gem100=None, con100=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_curves(ax, keys, all_data)
    setup_ax(ax, title, keys, all_data, gem100, con100)
    save_fig(fig, path)


def make_broken_axis_fig(keys, all_data, title, path,
                         gem100, con100, unlabeled_mape=213.0):
    """
    Create a broken y-axis figure with zigzag break.
    Bottom panel: main data (0 to ~max_data * 1.2)
    Top panel: narrow strip showing unlabeled-only baseline
    """
    # Compute y range for main data
    all_mapes = []
    for key in keys:
        for r, d in all_data[key].items():
            all_mapes.append(d["overall_mape"])
    if gem100: all_mapes.append(gem100)
    if con100: all_mapes.append(con100)
    data_max = max(all_mapes) if all_mapes else 15

    # Bottom: 0 to data_max * 1.25, Top: unlabeled_mape ± small range
    bottom_top = data_max * 1.25
    top_bottom = unlabeled_mape - 8
    top_top = unlabeled_mape + 8

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 6.5),
        gridspec_kw={"height_ratios": [1, 5], "hspace": 0.08}
    )

    # Plot on both axes
    for ax in (ax_top, ax_bot):
        plot_curves(ax, keys, all_data)
        if gem100 is not None:
            ax.axhline(y=gem100, color="#455A64", linestyle="--", linewidth=1.5,
                       alpha=0.7, label=f"V2 100% ({gem100:.1f}%)")
        if con100 is not None:
            ax.axhline(y=con100, color="#2E7D32", linestyle="--", linewidth=1.5,
                       alpha=0.7, label=f"Concorde 100% ({con100:.1f}%)")
        # Unlabeled-only line
        ax.axhline(y=unlabeled_mape, color="#B71C1C", linestyle=":",
                   linewidth=2.0, alpha=0.9, label=f"Unlabeled only ({unlabeled_mape:.0f}%)")

    # Set y limits
    ax_top.set_ylim(top_bottom, top_top)
    ax_bot.set_ylim(0, bottom_top)

    # Hide spines at the break
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False)

    # Draw clean wave break markers
    # Small sinusoidal wave across the break on both left and right edges
    wave_x = np.linspace(-0.02, 0.02, 30)
    wave_y = 0.008 * np.sin(wave_x / 0.04 * np.pi * 4)

    for x_center in [0.0, 1.0]:  # left and right edges
        # On bottom of top panel
        ax_top.plot(x_center + wave_x, wave_y, transform=ax_top.transAxes,
                    color="#666666", clip_on=False, linewidth=1.0)
        # On top of bottom panel
        ax_bot.plot(x_center + wave_x, 1.0 + wave_y, transform=ax_bot.transAxes,
                    color="#666666", clip_on=False, linewidth=1.0)

    # X-axis on bottom only
    all_pcts = set()
    for key in keys:
        all_pcts.update(r * 100 for r in all_data[key].keys())
    all_pcts = sorted(all_pcts)
    ax_bot.set_xticks(all_pcts)
    ax_bot.set_xticklabels([f"{int(x)}%" for x in all_pcts], rotation=45, ha="right")
    ax_bot.set_xlabel("Labeled Data (%)", fontweight="bold")

    # Y label centered
    fig.text(0.02, 0.5, "Overall MAPE (%)", va="center", rotation="vertical",
             fontweight="bold", fontsize=11)

    # Legend only on bottom, with border
    legend = ax_bot.legend(loc="upper right", frameon=True, edgecolor='black',
                           facecolor='white', fancybox=False, framealpha=1.0)
    if legend:
        legend.get_frame().set_linewidth(1.0)
    if ax_top.get_legend():
        ax_top.get_legend().remove()

    # Publication-quality grid and spines
    for ax in (ax_top, ax_bot):
        ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
        ax.xaxis.grid(False)
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_color('black')
                spine.set_linewidth(1.0)

    # Title
    ax_top.set_title(title, fontsize=12, fontweight="bold", pad=10)

    fig.savefig(path, dpi=600, bbox_inches="tight")
    pdf_path = path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {path} + {pdf_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["18k", "40k"])
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Base directory for results (default: ./results)")
    args = parser.parse_args()

    ds = args.dataset
    tt = DATASET_TITLES[ds]
    base_dir = f"{args.results_dir}/exp4_{ds}"
    save_dir = args.outdir or base_dir
    os.makedirs(save_dir, exist_ok=True)

    all_data = {
        "GEM": load_pretrain_results(PRETRAIN_PATHS[ds], ds),
        "CON": load_results(f"{base_dir}_concorde"),
        "V1":  load_results(f"{base_dir}_vanilla"),
        "V2":  load_results(f"{base_dir}_supervised_only"),
        "V3":  load_results(base_dir),
        "V3X": load_results(f"{args.results_dir}/exp4_{ds}_extra_unlabeled"),
    }
    gem100, con100 = load_exp1_baselines(ds)

    print(f"Dataset: {ds} ({tt})")
    for key in ["GEM", "CON", "V1", "V2", "V3", "V3X"]:
        print(f"  {CURVE_LABELS[key]}: {len(all_data[key])} ratios")
    print(f"  PRISM 100%:    {gem100:.2f}%")
    print(f"  Concorde 100%: {con100:.2f}%")
    print()

    # --- Fig 1: Concorde vs V1 ---
    print("Fig 1: Concorde vs V1")
    make_fig(["CON", "V1"], all_data,
             f"Value of MSCS Representation ({tt}): Concorde vs V1",
             f"{save_dir}/exp4_{ds}_1_concorde_vs_v1.png",
             gem100, con100)

    # --- Fig 2: V1 vs V2 ---
    print("Fig 2: V1 vs V2")
    make_fig(["V1", "V2"], all_data,
             f"Value of Physics Loss ({tt}): V1 vs V2",
             f"{save_dir}/exp4_{ds}_2_v1_vs_v2.png",
             gem100, con100)

    # --- Fig 3: V2 vs PRISM ---
    print("Fig 3: V2 vs PRISM")
    make_fig(["V2", "V3"], all_data,
             f"Value of Unlabeled Data ({tt}): V2 vs PRISM",
             f"{save_dir}/exp4_{ds}_3_v2_vs_prism.png",
             gem100, con100)

    # --- Fig 4: PRISM vs V3 ---
    print("Fig 4: PRISM vs V3")
    make_fig(["V3", "GEM"], all_data,
             f"Semi vs Pretrain+Finetune ({tt}): PRISM vs V3",
             f"{save_dir}/exp4_{ds}_4_prism_vs_v3.png",
             gem100, None)

    # --- Fig 5: V3 vs Concorde ---
    print("Fig 5: V3 vs Concorde")
    make_fig(["GEM", "CON"], all_data,
             f"Label Efficiency ({tt}): V3 vs Concorde",
             f"{save_dir}/exp4_{ds}_5_v3_vs_concorde.png",
             gem100, con100)

    # --- Fig 6: Incremental ablation ---
    print("Fig 6: Incremental ablation")
    make_fig(["CON", "V1", "V2", "V3"], all_data,
             f"Incremental Ablation ({tt}): Concorde -> V1 -> V2 -> PRISM",
             f"{save_dir}/exp4_{ds}_6_incremental.png",
             gem100, con100)

    # --- Fig 7: V2 + V1 + Concorde ---
    print("Fig 7: V2 + V1 + Concorde")
    make_fig(["V2", "V1", "CON"], all_data,
             f"Physics Loss Value ({tt}): V2 vs V1 vs Concorde",
             f"{save_dir}/exp4_{ds}_7_v2_v1_concorde.png",
             gem100, con100)

    # --- Fig 8: V3 + V2 + Concorde (V2 100% baseline) ---
    print("Fig 8: V3 + V2 + Concorde")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_curves(ax, ["GEM", "V2", "CON"], all_data)
    ax.axhline(y=gem100, color="#455A64", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"V2 100% ({gem100:.1f}%)")
    ax.axhline(y=con100, color="#2E7D32", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Concorde 100% ({con100:.1f}%)")
    setup_ax(ax, f"V3 vs Supervised Baselines ({tt}): V3 vs V2 vs Concorde",
             ["GEM", "V2", "CON"], all_data)
    save_fig(fig, f"{save_dir}/exp4_{ds}_8_v3_v2_concorde.png")

    # --- Fig 9: All variants ---
    print("Fig 9: All variants")
    make_fig(["GEM", "V3", "V2", "V1", "CON"], all_data,
             f"Label Efficiency ({tt}): All Variants",
             f"{save_dir}/exp4_{ds}_9_all_variants.png",
             gem100, con100)

    # --- Fig 10: PRISM + CON + V1 + V2 with broken y-axis ---
    print("Fig 10: PRISM + Concorde + V1 + V2 (broken y-axis)")
    prism100 = 3.8
    con100_fig10 = 4.0
    unlabeled_mape = 213.0

    # Compute y range for main data
    all_mapes_10 = []
    for key in ["V3", "CON", "V1", "V2"]:
        for r, d in all_data[key].items():
            all_mapes_10.append(d["overall_mape"])
    all_mapes_10.extend([prism100, con100_fig10])
    data_max_10 = max(all_mapes_10) if all_mapes_10 else 15

    bottom_top = data_max_10 * 1.25
    top_bottom = unlabeled_mape - 8
    top_top = unlabeled_mape + 8

    fig10, (ax10_top, ax10_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 6.5),
        gridspec_kw={"height_ratios": [1, 5], "hspace": 0.08}
    )

    for ax in (ax10_top, ax10_bot):
        plot_curves(ax, ["V3", "CON", "V1", "V2"], all_data)
        ax.axhline(y=prism100, color=CURVE_STYLES["V3"]["color"], linestyle="--",
                   linewidth=1.5, alpha=0.7, label=f"PRISM 100% ({prism100:.1f}%)")
        ax.axhline(y=con100_fig10, color=CURVE_STYLES["CON"]["color"], linestyle="--",
                   linewidth=1.5, alpha=0.7, label=f"Concorde 100% ({con100_fig10:.1f}%)")
        ax.axhline(y=unlabeled_mape, color="#B71C1C", linestyle=":",
                   linewidth=2.0, alpha=0.9, label=f"Unlabeled only ({unlabeled_mape:.0f}%)")

    ax10_top.set_ylim(top_bottom, top_top)
    ax10_bot.set_ylim(0, bottom_top)

    ax10_top.spines["bottom"].set_visible(False)
    ax10_bot.spines["top"].set_visible(False)
    ax10_top.tick_params(bottom=False)

    # Wave break markers
    wave_x = np.linspace(-0.02, 0.02, 30)
    wave_y = 0.008 * np.sin(wave_x / 0.04 * np.pi * 4)
    for x_center in [0.0, 1.0]:
        ax10_top.plot(x_center + wave_x, wave_y, transform=ax10_top.transAxes,
                      color="#666666", clip_on=False, linewidth=1.0)
        ax10_bot.plot(x_center + wave_x, 1.0 + wave_y, transform=ax10_bot.transAxes,
                      color="#666666", clip_on=False, linewidth=1.0)

    all_pcts_10 = sorted(set(r * 100 for key in ["V3", "CON", "V1", "V2"]
                             for r in all_data[key].keys()))
    ax10_bot.set_xticks(all_pcts_10)
    ax10_bot.set_xticklabels([f"{int(x)}%" for x in all_pcts_10], rotation=45, ha="right")
    ax10_bot.set_xlabel("Labeled Data (%)", fontweight="bold")
    fig10.text(0.02, 0.5, "Overall MAPE (%)", va="center", rotation="vertical",
               fontweight="bold", fontsize=11)

    legend = ax10_bot.legend(loc="upper right", frameon=True, edgecolor='black',
                             facecolor='white', fancybox=False, framealpha=1.0)
    if legend:
        legend.get_frame().set_linewidth(1.0)
    if ax10_top.get_legend():
        ax10_top.get_legend().remove()

    for ax in (ax10_top, ax10_bot):
        ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
        ax.xaxis.grid(False)
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_color('black')
                spine.set_linewidth(1.0)

    ax10_top.set_title(f"PRISM vs Concorde vs V1 V2 ({tt})", fontsize=12, fontweight="bold", pad=10)

    path10 = f"{save_dir}/exp4_{ds}_10_prism_con_v1_v2.png"
    fig10.savefig(path10, dpi=600, bbox_inches="tight")
    pdf10 = path10.rsplit(".", 1)[0] + ".pdf"
    fig10.savefig(pdf10, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {path10} + {pdf10}")

    # --- Fig 11: V3 + V2 + Concorde with broken y-axis ---
    print("Fig 11: V3 + V2 + Concorde (broken y-axis, unlabeled-only)")
    make_broken_axis_fig(
        ["GEM", "V2", "CON"], all_data,
        f"Semi-Supervised Value ({tt}): V3 vs V2 vs Concorde",
        f"{save_dir}/exp4_{ds}_11_v3_v2_con_unlabeled.png",
        gem100, con100, unlabeled_mape=213.0,
    )

    # --- Fig 12: V3 vs V3X — Unlabeled data scaling (18k only) ---
    if all_data["V3X"]:
        print("Fig 12: PRISM vs PRISM + extra unlabeled")
        make_fig(["V3X", "V3", "CON"], all_data,
                 f"Unlabeled Data Scaling ({tt}): PRISM vs PRISM + Extra Unlabeled",
                 f"{save_dir}/exp4_{ds}_12_extra_unlabeled.png",
                 gem100, con100)

        # --- Fig 13: V2 vs V3 vs V3X — Semi-supervised progression ---
        print("Fig 13: V2 vs PRISM vs PRISM + extra unlabeled")
        make_fig(["V3X", "V3", "V2"], all_data,
                 f"Semi-Supervised Progression ({tt}): V2 vs PRISM vs PRISM + Extra",
                 f"{save_dir}/exp4_{ds}_13_semi_progression.png",
                 gem100, con100)

    # --- Summary table ---
    all_ratios = set()
    for res in all_data.values():
        all_ratios.update(res.keys())
    all_ratios = sorted(all_ratios)

    has_v3x = bool(all_data["V3X"])

    print(f"\n{'='*85}")
    print(f"LABEL EFFICIENCY ({tt})")
    print(f"{'='*85}")
    print(f"  PRISM 100%:    {gem100:.2f}%")
    print(f"  Concorde 100%: {con100:.2f}%")
    if has_v3x:
        header = f"  {'Ratio':<8} {'Concorde':>9} {'V1':>8} {'V2':>8} {'PRISM':>8} {'P+Extra':>8} {'V3':>8}"
    else:
        header = f"  {'Ratio':<8} {'Concorde':>9} {'V1':>8} {'V2':>8} {'PRISM':>8} {'V3':>8}"
    print(header)
    print(f"  {'-'*len(header)}")
    for r in all_ratios:
        vals = []
        keys = ["CON", "V1", "V2", "V3"]
        if has_v3x:
            keys.append("V3X")
        keys.append("GEM")
        for key in keys:
            d = all_data[key].get(r)
            vals.append(f"{d['overall_mape']:.1f}%" if d else "—")
        print(f"  {r*100:>5.0f}%  {'  '.join(f'{v:>7}' for v in vals)}")
    print(f"{'='*85}")

    n_figs = 13 if has_v3x else 11
    print(f"\nDone! {n_figs} figures saved to {save_dir}/")


if __name__ == "__main__":
    main()