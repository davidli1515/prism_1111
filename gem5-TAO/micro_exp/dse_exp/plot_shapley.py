"""
Plot Shapley Value Attribution — Concorde-style stacked bar chart.

Reads JSON results from shapley_pruning.py and generates:
1. Concorde-style stacked bar (one bar per method, stacked by parameter)
2. Horizontal bar comparison (three panels side by side)

Usage:
    python plot_shapley.py --json results/605.mcf_s/mcf_shapley_results.json
    python plot_shapley.py --json results/605.mcf_s/mcf_shapley_results.json --output figures/
"""

import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# Parameter metadata
# ============================================================

# Display names (matching DeepMind collaborator's convention)
PARAM_DISPLAY = {
    "fetchWidth":      "Fetch width",
    "decodeWidth":     "Decode width",
    "dispatchWidth":   "Dispatch width",
    "commitWidth":     "Commit width",
    "wbWidth":         "Writeback width",
    "numROBEntries":   "ROB",
    "numIQEntries":    "Issue queue",
    "LQEntries":       "Load queue",
    "SQEntries":       "Store queue",
    "fetchBufferSize": "Fetch buffer",
    "IntALU":          "Int ALU",
    "IntMultDiv":      "Int Mult/Div",
    "FP_ALU":          "FP ALU",
    "FP_MultDiv":      "FP Mult/Div",
    "RdWrPort":        "Read/Write ports",
    "l1_size":         "L1 cache",
    "l1d_assoc":       "L1D assoc",
    "l1i_assoc":       "L1I assoc",
}

# Colors by category — muted, professional palette
PARAM_COLORS = {
    # Pipeline widths — slate blues
    "fetchWidth":      "#4C72B0",
    "decodeWidth":     "#6C8EBF",
    "dispatchWidth":   "#8FADD0",
    "commitWidth":     "#2E4A7A",
    "wbWidth":         "#A8C0D8",
    # Queue/buffer sizes — warm earth tones
    "numROBEntries":   "#C44E52",
    "numIQEntries":    "#D4817A",
    "LQEntries":       "#DD8452",
    "SQEntries":       "#E8A87C",
    "fetchBufferSize": "#EDCBA4",
    # Functional units — muted greens
    "IntALU":          "#55A868",
    "IntMultDiv":      "#7DC08A",
    "FP_ALU":          "#4C9D6E",
    "FP_MultDiv":      "#9CD1A8",
    "RdWrPort":        "#B8DFC3",
    # Cache — teals
    "l1_size":         "#47998B",
    "l1d_assoc":       "#3A7D8C",
    "l1i_assoc":       "#76B7B2",
}

# Category grouping for legend
PARAM_CATEGORIES = {
    "Pipeline Width": ["fetchWidth", "decodeWidth", "dispatchWidth", "commitWidth", "wbWidth"],
    "Queue / Buffer": ["numROBEntries", "numIQEntries", "LQEntries", "SQEntries", "fetchBufferSize"],
    "Functional Unit": ["IntALU", "IntMultDiv", "FP_ALU", "FP_MultDiv", "RdWrPort"],
    "Cache":           ["l1_size", "l1d_assoc", "l1i_assoc"],
}

FALLBACK_COLOR = "#cccccc"


# ============================================================
# Plot 1: Concorde-style stacked bar chart (DeepMind/Nature style)
# ============================================================
def plot_concorde_style(sv_full, sv_pruned, sv_oaat, param_list, benchmark, output_path,
                        base_cpi=None):
    """
    Stacked bar chart matching Nature/Science publication aesthetics.
    Style from DeepMind collaborator (prism_figure_5.py).
    """
    import pandas as pd

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

    # Sort params by Full Shapley descending
    sorted_params = sorted(param_list, key=lambda p: sv_full.get(p, 0), reverse=True)

    # Build raw_data dict: display_name -> {method: abs(value)}
    raw_data = {}
    if base_cpi is not None:
        raw_data["Baseline"] = {
            "Full MC Shapley": base_cpi,
            "MSCS-Pruned Shapley": base_cpi,
            "One-at-a-time": base_cpi,
        }
    for p in sorted_params:
        display = PARAM_DISPLAY.get(p, p)
        raw_data[display] = {
            "Full MC Shapley":     abs(sv_full.get(p, 0)),
            "MSCS-Pruned Shapley": abs(sv_pruned.get(p, 0)),
            "One-at-a-time":       abs(sv_oaat.get(p, 0)),
        }

    # Convert to DataFrame (rows = methods, columns = params)
    df = pd.DataFrame(raw_data)
    df.rename(index={
        'MSCS-Pruned Shapley': 'MSCS-Pruned (ours)',
        'One-at-a-time': r'One-at-a-time ($\Delta$CPI)',
    }, inplace=True)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = list(plt.cm.tab20.colors)
    if base_cpi is not None:
        colors = ['#B0B0B0'] + colors

    df.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=colors[:len(df.columns)],
        width=0.25,
        edgecolor='black',
        linewidth=0.4,
        zorder=3,
    )

    ax.set_title(f'CPI Attribution — {benchmark}', pad=15, fontweight='bold')
    ax.set_ylabel('CPI', fontweight='bold')
    plt.xticks(rotation=0)

    # Subtle dotted grid
    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
    ax.xaxis.grid(False)

    # All 4 spines visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    # Legend at bottom, reversed order, black border
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles[::-1], labels[::-1],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=True,
        edgecolor='black',
        facecolor='white',
        fancybox=False,
        shadow=False,
        framealpha=1.0,
        columnspacing=1.2,
        handletextpad=0.5,
        borderpad=0.8,
    )
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    fig.savefig(output_path, format='pdf', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"  Saved: {output_path}")


# ============================================================
# Plot 2: Horizontal bar comparison (three panels)
# ============================================================
def plot_bar_comparison(sv_full, sv_pruned, sv_oaat, param_list, benchmark, output_path):
    """Three horizontal bar panels, sorted by Full Shapley."""
    sorted_params = sorted(param_list, key=lambda p: sv_full.get(p, 0), reverse=True)
    y = np.arange(len(sorted_params))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    datasets = [
        (sv_full,   'Full MC Shapley\n(ground truth)', '#3498db'),
        (sv_pruned, 'MSCS-Pruned Shapley\n(ours)',     '#2ecc71'),
        (sv_oaat,   'One-at-a-time\n(ΔCPI)',           '#e74c3c'),
    ]

    # Find global x range for consistent axes
    all_vals = ([sv_full.get(p, 0) for p in sorted_params] +
                [sv_pruned.get(p, 0) for p in sorted_params] +
                [sv_oaat.get(p, 0) for p in sorted_params])
    xmin = min(all_vals) * 1.15
    xmax = max(all_vals) * 1.15

    for ax, (sv, title, color) in zip(axes, datasets):
        vals = [sv.get(p, 0) for p in sorted_params]
        colors = [PARAM_COLORS.get(p, FALLBACK_COLOR) for p in sorted_params]
        ax.barh(y, vals, color=colors, edgecolor='white', linewidth=0.3, height=0.7)

        # Value labels
        for i, (v, p) in enumerate(zip(vals, sorted_params)):
            if abs(v) > 0.005:
                ha = 'left' if v >= 0 else 'right'
                offset = 0.01 if v >= 0 else -0.01
                ax.text(v + offset, i, f'{v:+.3f}', ha=ha, va='center', fontsize=6.5)

        ax.set_yticks(y)
        labels = [PARAM_DISPLAY.get(p, p) for p in sorted_params]
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel('Shapley Value (CPI improvement)', fontsize=9)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlim(xmin, xmax)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(f'Shapley Value Comparison — {benchmark}', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================
# Plot 3: Cosine similarity heatmap (from matrix in JSON)
# ============================================================
def plot_cosine_heatmap(sim_matrix, param_list, groups, benchmark, output_path):
    """Cosine similarity heatmap with group boundaries."""
    # Reorder by groups
    ordered_params = []
    group_boundaries = [0]
    for g in groups:
        for p in g:
            if p in param_list:
                ordered_params.append(p)
        group_boundaries.append(len(ordered_params))

    # Build reorder index
    idx_map = [param_list.index(p) for p in ordered_params if p in param_list]
    n = len(idx_map)
    reordered = np.array(sim_matrix)[np.ix_(idx_map, idx_map)]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(reordered, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')

    for b in group_boundaries[1:-1]:
        ax.axhline(b - 0.5, color='blue', linewidth=2)
        ax.axvline(b - 0.5, color='blue', linewidth=2)

    display_labels = [PARAM_DISPLAY.get(p, p) for p in ordered_params]
    ax.set_xticks(range(n))
    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(display_labels, fontsize=8)
    ax.set_title(f'MSCS Cosine Similarity — {benchmark}', fontsize=13)
    plt.colorbar(im, ax=ax, label='|cos(v_i, v_j)|', shrink=0.8)

    for gi, g in enumerate(groups):
        start = group_boundaries[gi]
        end = group_boundaries[gi + 1]
        mid = (start + end - 1) / 2
        ax.annotate(f'G{gi+1}', xy=(-0.08, mid), fontsize=9, color='blue',
                    fontweight='bold', ha='right', va='center',
                    xycoords=('axes fraction', 'data'),
                    annotation_clip=False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Plot Shapley attribution results")
    parser.add_argument("--json", required=True, help="Path to shapley_results.json")
    parser.add_argument("--output", default=None, help="Output directory (default: same as JSON)")
    parser.add_argument("--exclude", nargs="*", default=["dispatchWidth"],
                        help="Parameters to exclude from plots (default: dispatchWidth)")
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    benchmark = data.get("benchmark", "unknown")
    bname = benchmark.split(".")[-1]
    for suffix in ("_s", "_r"):
        bname = bname.replace(suffix, "")
    param_list = [p for p in data["params"] if p not in args.exclude]
    sv_full = data["shapley_full"]
    sv_pruned = data["shapley_pruned"]
    sv_oaat = data["shapley_oaat"]
    groups = [[p for p in g if p not in args.exclude] for g in data.get("groups", [param_list])]
    groups = [g for g in groups if g]  # remove empty groups
    sim_matrix = data.get("cosine_matrix", None)
    metrics = data.get("metrics", {})

    output_dir = args.output or os.path.dirname(args.json)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Plotting Shapley results — {bname}")
    print(f"{'='*60}")
    if args.exclude:
        print(f"  Excluded: {args.exclude}")
    print(f"  Parameters: {len(param_list)}")
    print(f"  Groups: {len(groups)} (sizes: {[len(g) for g in groups]})")
    if metrics:
        for method in ["full", "pruned", "oaat"]:
            m = metrics.get(method, {})
            print(f"  {method:<10}: ρ={m.get('rho','?')}, MSE={m.get('mse','?')}, "
                  f"evals={m.get('evals','?')}, time={m.get('time_s','?')}s")

    # Print attribution table
    print(f"\n  {'Param':<20} {'Full MC':>10} {'Pruned':>10} {'1-at-a':>10} {'Δ(P-F)':>10}")
    print(f"  {'─'*60}")
    sorted_p = sorted(param_list, key=lambda p: sv_full.get(p, 0), reverse=True)
    for p in sorted_p:
        f = sv_full.get(p, 0)
        pr = sv_pruned.get(p, 0)
        o = sv_oaat.get(p, 0)
        print(f"  {p:<20} {f:>+10.4f} {pr:>+10.4f} {o:>+10.4f} {pr-f:>+10.4f}")

    # Get base CPI (CPI at DEFAULT config)
    base_cpi = data.get("cpi_default", None)
    if base_cpi is None:
        # Try to compute from: CPI(all min) - sum(full shapley) = base
        cpi_min = data.get("cpi_all_min", None)
        if cpi_min is not None:
            base_cpi = cpi_min - sum(sv_full.get(p, 0) for p in data["params"])

    # Generate plots
    plot_concorde_style(sv_full, sv_pruned, sv_oaat, param_list, bname,
                        os.path.join(output_dir, f"{bname}_shapley_concorde.pdf"),
                        base_cpi=base_cpi)

    plot_bar_comparison(sv_full, sv_pruned, sv_oaat, param_list, bname,
                        os.path.join(output_dir, f"{bname}_shapley_bars.png"))

    if sim_matrix is not None:
        # sim_matrix is indexed by original param order
        orig_param_list = data["params"]
        plot_cosine_heatmap(sim_matrix, orig_param_list, groups, bname,
                            os.path.join(output_dir, f"{bname}_cosine_heatmap.png"))

    print(f"\n  Done!")


if __name__ == "__main__":
    main()