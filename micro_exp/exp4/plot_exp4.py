"""
Plot Experiment 4: Label Efficiency + Ablation

Usage:
    python plot_exp4.py --dataset 40k
    python plot_exp4.py --dataset 18k
"""

import json, os, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SKIP_RATIOS = {0.10}

# Dataset → exp1 baseline JSON paths
EXP1_PATHS = {
    "18k": {
        "gem_full": "/path/to/gem5/micro_exp/exp1/results/exp1_v2/exp1_512_256_183k/gem_const.json",
        "concorde": "/path/to/gem5/micro_exp/exp1/results/exp1_v2/exp1_256_128_183k/concorde.json",
    },
    "40k": {
        "gem_full": "/path/to/gem5/micro_exp/exp1/results/exp1_v2/exp1_512_256/gem_full.json",
        "concorde": "/path/to/gem5/micro_exp/exp1/results/exp1_v2/exp1_512_256/concorde.json",
    },
}


def load_ratio_results(result_dir):
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


def load_exp1_baselines(dataset):
    """Load gem_full and concorde 100% baselines from separate JSON files."""
    paths = EXP1_PATHS[dataset]
    with open(paths["gem_full"]) as f:
        gem_raw = json.load(f)
    with open(paths["concorde"]) as f:
        con_raw = json.load(f)
    return gem_raw["per_benchmark"], con_raw["per_benchmark"]


def plot_label_efficiency(variants, gem_full_bm, concorde_bm, save_path, dataset_tag):
    ref_results = None
    for name, (res, _, _, _) in variants.items():
        if res:
            ref_results = res
            break
    if ref_results is None:
        print("  No results found!"); return

    ref_ratios = sorted(ref_results.keys())
    ratio_pcts = [r * 100 for r in ref_ratios]
    benchmarks = sorted(ref_results[ref_ratios[0]]["per_benchmark"].keys())

    def get_per_bm(results, bm):
        r_sorted = sorted(results.keys())
        return ([r*100 for r in r_sorted],
                [results[r]["per_benchmark"].get(bm, np.nan) for r in r_sorted])

    def get_overall(results):
        r_sorted = sorted(results.keys())
        return ([r*100 for r in r_sorted],
                [results[r]["overall_mape"] for r in r_sorted])

    bm_in_both = [b for b in benchmarks if b in gem_full_bm and b in concorde_bm]
    gem_full_overall = np.exp(np.mean(np.log([gem_full_bm[b] for b in bm_in_both])))
    concorde_overall = np.exp(np.mean(np.log([concorde_bm[b] for b in bm_in_both])))

    n_panels = len(benchmarks) + 1
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.5, nrows*2.8))
    axes = axes.flatten()
    color_gem = "#E65100"; color_con = "#2E7D32"

    def setup_xaxis(ax):
        ax.set_xticks(ratio_pcts)
        ax.set_xticklabels([f"{int(x)}%" for x in ratio_pcts], fontsize=6, rotation=45, ha="right")
        ax.set_xlabel("Labeled Data", fontsize=7)

    for i, bm in enumerate(benchmarks):
        ax = axes[i]
        for name, (res, color, marker, label) in variants.items():
            if not res: continue
            xs, ys = get_per_bm(res, bm)
            ax.plot(xs, ys, f"{marker}-", color=color, markersize=3, linewidth=1.5, zorder=3, label=label)
        if bm in gem_full_bm:
            ax.axhline(y=gem_full_bm[bm], color=color_gem, linestyle="--", linewidth=1.5, label="GEM 100%")
        if bm in concorde_bm:
            ax.axhline(y=concorde_bm[bm], color=color_con, linestyle="--", linewidth=1.5, label="Concorde 100%")
        ax.set_title(bm.replace("_s","").replace("_r",""), fontsize=9, fontweight="bold", pad=3)
        ax.set_ylim(bottom=0); ax.grid(alpha=0.2); ax.tick_params(labelsize=7); setup_xaxis(ax)
        if i % ncols == 0: ax.set_ylabel("MAPE (%)", fontsize=8)

    ax = axes[len(benchmarks)]
    for name, (res, color, marker, label) in variants.items():
        if not res: continue
        xs, ys = get_overall(res)
        lw = 2.0 if name == "full" else 1.5; ms = 4 if name == "full" else 3
        ax.plot(xs, ys, f"{marker}-", color=color, markersize=ms, linewidth=lw, zorder=3, label=label)
    ax.axhline(y=gem_full_overall, color=color_gem, linestyle="--", linewidth=1.5, label="GEM 100%")
    ax.axhline(y=concorde_overall, color=color_con, linestyle="--", linewidth=1.5, label="Concorde 100%")
    ax.set_title("Overall", fontsize=10, fontweight="bold", pad=3)
    ax.set_ylim(bottom=0); ax.grid(alpha=0.2); ax.tick_params(labelsize=7); setup_xaxis(ax)

    for j in range(len(benchmarks)+1, len(axes)): axes[j].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(handles),7), fontsize=7, frameon=True, bbox_to_anchor=(0.5,-0.03))
    fig.suptitle(f"Exp 4 ({dataset_tag}): Label Efficiency — Semi-Supervised Ablation", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  Saved {save_path}")


def plot_summary_grouped(variants, gem_full_bm, concorde_bm, save_path, dataset_tag):
    bm_keys = [b for b in gem_full_bm if b in concorde_bm]
    gem_overall = np.exp(np.mean(np.log([gem_full_bm[b] for b in bm_keys])))
    con_overall = np.exp(np.mean(np.log([concorde_bm[b] for b in bm_keys])))

    variant_order = [
        ("full",       "(g) GEM: L_const+L_mono"),
        ("const_only", "(e) GEM: L_const only"),
        ("mono_only",  "(f) GEM: L_mono only"),
        ("sup_only",   "(h) GEM: no unlabeled"),
        ("concorde",   "(i) Concorde"),
        ("vanilla",    "(j) GEM: L_task only"),
    ]
    active = []
    for key, label in variant_order:
        if key in variants and variants[key][0]:
            res, color, _, _ = variants[key]
            active.append((key, res, color, label))

    n_variants = len(active)
    if n_variants == 0: return

    all_ratios = set()
    for key, res, _, _ in active: all_ratios.update(res.keys())
    all_ratios = sorted(all_ratios); n_ratios = len(all_ratios)

    bar_height = 0.13; group_gap = 0.3; group_height = n_variants * bar_height
    fig_height = max(4, n_ratios * (group_height + group_gap) * 0.85 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    group_centers = np.arange(n_ratios) * (group_height + group_gap)

    for v_idx, (key, res, color, label) in enumerate(active):
        y_positions = []; x_values = []
        for r_idx, ratio in enumerate(all_ratios):
            y = group_centers[r_idx] + (v_idx - (n_variants-1)/2) * bar_height
            y_positions.append(y)
            x_values.append(res[ratio]["overall_mape"] if ratio in res else 0)
        ax.barh(y_positions, x_values, height=bar_height*0.88, color=color, label=label, edgecolor="white", linewidth=0.5, zorder=3)
        for y, x in zip(y_positions, x_values):
            if x > 0: ax.text(x+0.12, y, f"{x:.1f}%", va="center", fontsize=7, fontweight="bold", color=color)

    ax.axvline(x=gem_overall, color="#E65100", linestyle="--", linewidth=1.8, label=f"GEM 100% ({gem_overall:.1f}%)", zorder=2)
    ax.axvline(x=con_overall, color="#2E7D32", linestyle="--", linewidth=1.8, label=f"Concorde 100% ({con_overall:.1f}%)", zorder=2)
    ax.set_yticks(group_centers)
    ax.set_yticklabels([f"{int(r*100)}% labeled" for r in all_ratios], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Overall MAPE (%)", fontsize=12)
    ax.set_title(f"Label Efficiency Summary ({dataset_tag})", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, zorder=0)
    max_val = max(max((res[r]["overall_mape"] for r in res), default=0) for _, res, _, _ in active)
    ax.set_xlim(right=max_val*1.18)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["18k", "40k"])
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Base directory for results (default: ./results)")
    args = parser.parse_args()

    ds = args.dataset
    base_dir = f"{args.results_dir}/exp4_{ds}"
    save_dir = args.outdir or base_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading results for dataset={ds}...")
    if SKIP_RATIOS: print(f"  Skipping: {[f'{r*100:.0f}%' for r in SKIP_RATIOS]}")

    full_results  = load_ratio_results(base_dir)
    const_results = load_ratio_results(f"{base_dir}_const_only")
    mono_results  = load_ratio_results(f"{base_dir}_mono_only")
    sup_results   = load_ratio_results(f"{base_dir}_supervised_only")
    con_results   = load_ratio_results(f"{base_dir}_concorde")
    van_results   = load_ratio_results(f"{base_dir}_vanilla")

    variants = {
        "full":       (full_results,  "#1565C0", "o", "(g) L_const+L_mono"),
        "const_only": (const_results, "#7B1FA2", "s", "(e) L_const only"),
        "mono_only":  (mono_results,  "#C62828", "^", "(f) L_mono only"),
        "sup_only":   (sup_results,   "#455A64", "D", "(h) no unlabeled"),
        "concorde":   (con_results,   "#2E7D32", "v", "(i) Concorde"),
        "vanilla":    (van_results,   "#E91E63", "P", "(j) L_task only"),
    }
    for name, (res, _, _, label) in variants.items():
        print(f"  {label}: {len(res)} ratios")

    print(f"Loading exp1 baselines ({ds})...")
    gem_full_bm, concorde_bm = load_exp1_baselines(ds)

    print("\nPlotting...")
    plot_label_efficiency(variants, gem_full_bm, concorde_bm,
                          f"{save_dir}/exp4_{ds}_label_efficiency.png", ds)
    plot_summary_grouped(variants, gem_full_bm, concorde_bm,
                         f"{save_dir}/exp4_{ds}_summary.png", ds)
    print("Done!")


if __name__ == "__main__":
    main()