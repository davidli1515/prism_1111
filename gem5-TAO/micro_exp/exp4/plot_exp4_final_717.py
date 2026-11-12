"""
plot_exp4_final_717.py — Final paper figures for PRISM label efficiency.

Three figures:
  Fig 1: Main results (Concorde, V1, V2, V3, PRISM)
  Fig 2: Why semi-supervised (V3 vs Type1 vs Type2)
  Fig 3: Roles of L_const and L_mono (TBD)

Usage:
    python plot_exp4_final_717.py
"""

import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Publication-quality style
# ============================================================
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

# ============================================================
# DATA PATHS — Edit these to match your results
# ============================================================

# Fig 1: Main results
PATHS_FIG1 = {
    "V1":       "/scratch/gem5-TAO/micro_exp/exp4/results_77/exp4_18k_vanilla",
    "V2":       "/scratch/gem5-TAO/micro_exp/exp4/results_77/exp4_18k_supervised_only",
    "Concorde": "/scratch/gem5-TAO/micro_exp/exp4/results_77/exp4_18k_concorde",
    "V3":       "/scratch/gem5-TAO/micro_exp/exp4/results_77/exp4_18k",
    "PRISM":    "/scratch/gem5-TAO/micro_exp/exp4/results_77/exp4_18k_extra_unlabeled",
}

# Fig 2: Why semi-supervised
PATHS_FIG2 = {
    "V3":    "/scratch/gem5-TAO/micro_exp/exp4/results_77/exp4_18k",
    "Type1": "/scratch/gem5-TAO/micro_exp/exp4/results_77/exp4_18k_pretrain_ft_type1",
    "Type2": "/scratch/gem5-TAO/micro_exp/exp5/results_pt_ft_v2_18k",
}

# Fig 3: Roles of losses
PATHS_FIG3 = {
    "V3":              "/scratch/gem5-TAO/micro_exp/exp4/results_77/exp4_18k",
    "PRISM":           "/scratch/gem5-TAO/micro_exp/exp4/results_77/exp4_18k_extra_unlabeled",
    "constonly_L":     "/scratch/gem5-TAO/micro_exp/exp4/results_77_ablation/exp4_18k",
    "constonly_U":     "/scratch/gem5-TAO/micro_exp/exp4/results_77_ablation/exp4_18k_const_only",
    "constonly_LU":    "/scratch/gem5-TAO/micro_exp/exp4/results_77_ablation/exp4_18k_extra_unlabeled",
}

# Ratios to skip per figure
SKIP_RATIOS_FIG1 = {0.10, 0.25, 0.65, 0.68}
SKIP_RATIOS_FIG2 = {0.10, 0.25, 0.65, 0.68}
SKIP_RATIOS_FIG3 = {0.10, 0.25, 0.65, 0.68}

# Reference lines (horizontal dashed) — best results, hardcoded
REF_LINES = {
    "PRISM 100%":    {"y": 2.7, "color": "#FF6F00", "ls": "--"},
    "Concorde 100%": {"y": 2.7, "color": "#2E7D32", "ls": ":"},
}

# Output directory
SAVE_DIR = "/scratch/gem5-TAO/micro_exp/exp4/figures_final"

# ============================================================
# Curve styles
# ============================================================

STYLES = {
    "Concorde": {"color": "#2E7D32", "marker": "s", "lw": 2.0, "ms": 6},
    "V1":       {"color": "#E91E63", "marker": "^", "lw": 2.0, "ms": 6},
    "V2":       {"color": "#455A64", "marker": "D", "lw": 2.0, "ms": 6},
    "V3":       {"color": "#1565C0", "marker": "o", "lw": 2.0, "ms": 6},
    "PRISM":    {"color": "#FF6F00", "marker": "p", "lw": 2.5, "ms": 7},
    "Type1":    {"color": "#7B1FA2", "marker": "v", "lw": 2.0, "ms": 6},
    "Type2":    {"color": "#00838F", "marker": ">", "lw": 2.0, "ms": 6},
    "constonly_L":  {"color": "#C62828", "marker": "<", "lw": 2.0, "ms": 6},
    "constonly_U":  {"color": "#AD1457", "marker": "h", "lw": 2.0, "ms": 6},
    "constonly_LU": {"color": "#00695C", "marker": "*", "lw": 2.0, "ms": 7},
}

LABELS = {
    "Concorde": "Concorde",
    "V1":       "V1: +MSCS",
    "V2":       "V2: +Physics",
    "V3":       "V3: +Semi-Supervised",
    "PRISM":    "PRISM (+Extra Unlabeled)",
    "Type1":    "PT(L) + FT(U)",
    "Type2":    "PT(U) + FT(L)",
    "constonly_L":  r"V3: remove $\mathcal{L}_{mono}$ from (L)",
    "constonly_U":  r"V3: remove $\mathcal{L}_{mono}$ from (U)",
    "constonly_LU": r"PRISM: remove $\mathcal{L}_{mono}$ from both",
}

# ============================================================
# Data loading
# ============================================================

def load_results(directory, skip_ratios=None):
    """Load all exp4_{ratio}pct.json from a directory."""
    if skip_ratios is None:
        skip_ratios = set()
    results = {}
    if not os.path.isdir(directory):
        print(f"  WARNING: {directory} not found")
        return results
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        if not fname.startswith("exp4_"):
            continue
        with open(os.path.join(directory, fname)) as f:
            data = json.load(f)
        ratio = data.get("labeled_ratio", data.get("ratio"))
        mape = data.get("overall_mape")
        if ratio is None or mape is None:
            continue
        if ratio in skip_ratios:
            continue
        results[ratio] = {"overall_mape": mape}
    return results


def load_type2_results(directory, skip_ratios=None):
    """Load Type2 (pretrain+ft) results — reads 'pretrain_ft' key from any .json."""
    if skip_ratios is None:
        skip_ratios = set()
    results = {}
    if not os.path.isdir(directory):
        print(f"  WARNING: directory not found: {directory}")
        return results
    json_files = [f for f in sorted(os.listdir(directory)) if f.endswith(".json")]
    if not json_files:
        print(f"  WARNING: no .json files in {directory}")
        # Try looking for files directly (might be flat structure)
        return results
    for fname in json_files:
        fpath = os.path.join(directory, fname)
        with open(fpath) as f:
            raw = json.load(f)
        data = raw[0] if isinstance(raw, list) else raw
        ratio = data.get("labeled_ratio", data.get("ratio"))
        mape = data.get("pretrain_ft_v2", data.get("pretrain_ft", data.get("overall_mape")))
        if ratio is None or mape is None:
            print(f"    SKIP {fname}: ratio={ratio}, mape={mape}")
            continue
        if ratio in skip_ratios:
            continue
        results[ratio] = {"overall_mape": mape}
    return results


# ============================================================
# Plotting helpers
# ============================================================

def plot_curves(ax, keys, data_dict, label_keys=None, offset_overrides=None):
    """Plot multiple curves on the same axes.
    label_keys: set of key names to annotate with values. None = no labels.
    offset_overrides: dict of {key: direction} to override default offset.
    """
    # Predefined vertical offset direction per key to avoid overlap
    offset_map = {
        "Concorde": +1, "V1": +1, "V2": -1, "V3": +1, "PRISM": -1,
        "Type1": +1, "Type2": -1,
        "constonly_L": +1, "constonly_U": -1, "constonly_LU": +1,
    }
    if offset_overrides:
        offset_map.update(offset_overrides)
    if label_keys is None:
        label_keys = set()
    
    for key in keys:
        if key not in data_dict or not data_dict[key]:
            continue
        ratios = sorted(data_dict[key].keys())
        pcts = [r * 100 for r in ratios]
        mapes = [data_dict[key][r]["overall_mape"] for r in ratios]

        s = STYLES.get(key, {"color": "gray", "marker": "x", "lw": 1.5, "ms": 5})
        ax.plot(pcts, mapes,
                color=s["color"], marker=s["marker"],
                linewidth=s["lw"], markersize=s["ms"],
                label=LABELS.get(key, key),
                zorder=3)
        
        if key in label_keys:
            direction = offset_map.get(key, +1)
            # Custom offsets for specific points: (key, x%) → (dx, dy, va)
            custom = {
                ("Concorde", 30): (0, -10, "top"),
                ("Concorde", 40): (0, -10, "top"),
                ("Concorde", 60): (0, -6, "top"),
                ("Concorde", 90): (12, 8, "bottom"),
            }
            for x, y in zip(pcts, mapes):
                if (key, int(x)) in custom:
                    dx, dy, v = custom[(key, int(x))]
                else:
                    dx = 0
                    dy = 8 * direction
                    v = "bottom" if direction > 0 else "top"
                ax.annotate(f"{y:.1f}",
                           (x, y),
                           textcoords="offset points",
                           xytext=(dx, dy),
                           ha="center", va=v,
                           fontsize=7, color=s["color"],
                           fontweight="bold")


def setup_ax(ax, title, keys, data_dict, zoom_in=False):
    """Configure axes with publication styling."""
    # Collect all x values
    all_pcts = sorted(set(
        r * 100 for key in keys
        if key in data_dict
        for r in data_dict[key].keys()
    ))

    ax.set_xlabel("Labeled Data (%)", fontweight="bold")
    ax.set_ylabel("Overall MAPE (%)", fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(all_pcts)
    ax.set_xticklabels([f"{int(x)}%" for x in all_pcts], rotation=45, ha="right")

    # Reference lines (added to legend, no inline text)
    for ref_name, ref in REF_LINES.items():
        ax.axhline(y=ref["y"], color=ref["color"], linestyle=ref["ls"],
                   linewidth=1.5, alpha=0.7, zorder=1, label=ref_name)

    # Grid and spines
    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    # Y-axis limits
    if zoom_in:
        # Collect all mape values to determine tight range
        all_mapes = [data_dict[k][r]["overall_mape"]
                     for k in keys if k in data_dict
                     for r in data_dict[k]]
        y_min = max(0, min(all_mapes) - 0.3)
        y_max = max(all_mapes) + 0.3
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(bottom=0)

    # Legend
    legend = ax.legend(loc="upper right", frameon=True, edgecolor='black',
                       facecolor='white', fancybox=False, framealpha=1.0)
    if legend:
        legend.get_frame().set_linewidth(1.0)


def save_fig(fig, path):
    """Save as both PNG and PDF."""
    plt.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    pdf_path = path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {path} + {pdf_path}")


# ============================================================
# Figure 1: Main Results
# ============================================================

def plot_fig1(data, save_dir):
    """Concorde, V1, V2, V3, PRISM — the incremental ablation chain."""
    print("\n=== Figure 1: Main Results ===")
    keys = ["Concorde", "V1", "V2", "V3", "PRISM"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    # Threshold lines at 3%, 4%, 5% for label savings reference
    for thresh in [3.0, 4.0, 5.0]:
        ax.axhline(y=thresh, color="#BDBDBD", linestyle="--", linewidth=1.2,
                   alpha=0.6, zorder=0)
        ax.text(4, thresh + 0.1, f"{thresh:.0f}%", fontsize=8, color="#757575",
                va="bottom", ha="left")
    
    plot_curves(ax, keys, data, label_keys={"PRISM", "Concorde"})
    setup_ax(ax, "Label Efficiency: Incremental Ablation", keys, data)
    save_fig(fig, os.path.join(save_dir, "fig1_main_results.png"))


# ============================================================
# Figure 2: Why Semi-Supervised
# ============================================================

def plot_fig2(data, save_dir):
    """V3 (joint) vs Type1 (L→U) vs Type2 (U→L)."""
    print("\n=== Figure 2: Why Semi-Supervised ===")
    keys = ["V3", "Type1", "Type2"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_curves(ax, keys, data, label_keys={"V3", "Type1", "Type2"},
                offset_overrides={"V3": -1})
    setup_ax(ax, "Semi Vs. PT+FT", keys, data)
    save_fig(fig, os.path.join(save_dir, "fig2_why_semi.png"))


# ============================================================
# Figure 3: Roles of Losses (TBD)
# ============================================================

def plot_fig3(data, save_dir):
    """Roles of L_const and L_mono: V3, PRISM as baselines, 3 ablations.
    With zoom-in inset at 5% region where V3 variants overlap."""
    if not data:
        print("\n=== Figure 3: Roles of Losses === SKIPPED (no paths configured)")
        return
    print("\n=== Figure 3: Roles of Losses ===")
    keys = ["V3", "PRISM", "constonly_L", "constonly_U", "constonly_LU"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Phase dividers (vertical lines)
    ax.axvline(x=15, color='#9E9E9E', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)
    ax.axvline(x=60, color='#9E9E9E', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)

    plot_curves(ax, keys, data, label_keys={"V3", "PRISM"})
    setup_ax(ax, r"Roles of $\mathcal{L}_{const}$ and $\mathcal{L}_{mono}$",
             keys, data, zoom_in=True)

    # Phase labels (lower-middle of y range)
    y_lo, y_hi = ax.get_ylim()
    y_pos = y_lo + (y_hi - y_lo) * 0.35
    ax.text(9, y_pos, "Phase 1", fontsize=11, color='#757575', ha='center',
            fontstyle='italic', fontweight='bold', alpha=0.6)
    ax.text(37, y_pos, "Phase 2", fontsize=11, color='#757575', ha='center',
            fontstyle='italic', fontweight='bold', alpha=0.6)
    ax.text(80, y_pos, "Phase 3", fontsize=11, color='#757575', ha='center',
            fontstyle='italic', fontweight='bold', alpha=0.6)

    # --- Zoom-in inset at 5% region ---
    # Collect values at 5% for the three V3 variants to determine y range
    zoom_keys = ["V3", "constonly_L", "constonly_U"]
    vals_at_5 = []
    for k in zoom_keys:
        if k in data and 0.05 in data[k]:
            vals_at_5.append(data[k][0.05]["overall_mape"])

    if vals_at_5:
        y_center = np.mean(vals_at_5)
        y_margin = max(max(vals_at_5) - min(vals_at_5), 0.3) * 2.0
        zoom_y_lo = y_center - y_margin
        zoom_y_hi = y_center + y_margin
        zoom_x_lo = 2
        zoom_x_hi = 8

        # Create inset axes
        axins = ax.inset_axes([0.18, 0.55, 0.25, 0.35])

        # Plot 5% points with larger markers
        for key in zoom_keys:
            if key not in data or not data[key]:
                continue
            s = STYLES.get(key, {"color": "gray", "marker": "x", "lw": 1.5, "ms": 5})
            if 0.05 in data[key]:
                x, y = 5, data[key][0.05]["overall_mape"]
                axins.plot(x, y, color=s["color"], marker=s["marker"],
                          markersize=s["ms"] + 4, zorder=3)
                axins.annotate(f"{y:.2f}",
                             (x, y),
                             textcoords="offset points",
                             xytext=(10, 0),
                             ha="left", va="center",
                             fontsize=9, color=s["color"],
                             fontweight="bold")

        axins.set_xlim(zoom_x_lo, zoom_x_hi)
        axins.set_ylim(zoom_y_lo, zoom_y_hi)
        axins.set_xticks([5])
        axins.set_xticklabels(["5%"])
        axins.tick_params(labelsize=8)
        axins.yaxis.grid(True, linestyle=':', alpha=0.5)
        axins.set_title("5% labeled (zoom)", fontsize=9, fontstyle="italic")
        axins.patch.set_facecolor('white')
        axins.patch.set_alpha(0.95)
        for spine in axins.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1.2)

    save_fig(fig, os.path.join(save_dir, "fig3_roles_of_losses.png"))


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Output: {SAVE_DIR}/")

    # Load Fig 1 data
    print("\nLoading Fig 1 data...")
    data_fig1 = {}
    for key, path in PATHS_FIG1.items():
        data_fig1[key] = load_results(path, SKIP_RATIOS_FIG1)
        print(f"  {key}: {len(data_fig1[key])} ratios from {path}")   

    # Load Fig 2 data
    print("\nLoading Fig 2 data...")
    data_fig2 = {}
    data_fig2["V3"] = load_results(PATHS_FIG2["V3"], SKIP_RATIOS_FIG2)
    data_fig2["Type1"] = load_results(PATHS_FIG2["Type1"], SKIP_RATIOS_FIG2)
    data_fig2["Type2"] = load_type2_results(PATHS_FIG2["Type2"], SKIP_RATIOS_FIG2)
    for key in ["V3", "Type1", "Type2"]:
        print(f"  {key}: {len(data_fig2[key])} ratios from {PATHS_FIG2[key]}")

    # Load Fig 3 data
    print("\nLoading Fig 3 data...")
    data_fig3 = {}
    for key, path in PATHS_FIG3.items():
        data_fig3[key] = load_results(path, SKIP_RATIOS_FIG3)
        print(f"  {key}: {len(data_fig3[key])} ratios from {path}")

    # Plot
    plot_fig1(data_fig1, SAVE_DIR)
    plot_fig2(data_fig2, SAVE_DIR)
    plot_fig3(data_fig3, SAVE_DIR)

    print(f"\nDone! Figures saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()