"""
Plot DSE Results

Generates three figures from saved DSE results:
  1. Search Path (Fig 10 style): step-by-step MSCS + CPI + Area for best trajectory
  2. Pareto Frontier (Fig 13 style): all methods' Pareto fronts overlaid
  3. HV Curve (Fig 12 style): Pareto hypervolume vs number of evaluations

Usage:
    python plot_dse_results.py --benchmark 605.mcf_s
    python plot_dse_results.py --benchmark 623.xalancbmk_s
    python plot_dse_results.py --benchmark 605.mcf_s --results-dir /path/to/results
"""

import argparse
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Style: Nature/Science publication quality
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
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

DSE_DIR = "/scratch/gem5-TAO/micro_exp/dse_exp"
RESULTS_DIR = os.path.join(DSE_DIR, "results")

# Reference point: computed from ALL methods' data combined (fair + meaningful scale)
FIXED_REF_POINT = None  # Will be computed at plot time from all results


# ============================================================
# HV computation (standalone, no dse_engine import needed)
# ============================================================
def compute_pareto_front(points):
    """Return indices of Pareto-optimal points (minimize both objectives)."""
    pts = np.array(points)
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if pts[j][0] <= pts[i][0] and pts[j][1] <= pts[i][1] and \
               (pts[j][0] < pts[i][0] or pts[j][1] < pts[i][1]):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def compute_hypervolume_2d(pareto_points, ref_point):
    """Compute 2D hypervolume (area dominated by Pareto front below ref point)."""
    if not pareto_points:
        return 0.0
    pts = sorted(pareto_points, key=lambda p: p[0])
    hv = 0.0
    prev_area = ref_point[1]
    for cpi, area in pts:
        if cpi >= ref_point[0] or area >= ref_point[1]:
            continue
        hv += (ref_point[0] - cpi) * (prev_area - area)
        prev_area = area
    return hv


def get_common_ref(results):
    """Compute a common ref point from all methods' data combined."""
    max_cpi = max(max(r["cpis"]) for r in results.values() if r["cpis"])
    max_area = max(max(r["areas"]) for r in results.values() if r["areas"])
    return (max_cpi * 1.1, max_area * 1.1)


def recompute_hv_curve(cpis, areas, ref_point):
    """Recompute HV curve from raw cpis/areas. Subsampled for speed."""
    n = len(cpis)
    # Subsample: compute at ~50 points max
    if n <= 50:
        checkpoints = list(range(1, n + 1))
    else:
        step = n // 50
        checkpoints = list(range(step, n, step)) + [n]

    curve = []
    for i in checkpoints:
        pts = list(zip(cpis[:i], areas[:i]))
        pidx = compute_pareto_front(pts)
        pareto_pts = [pts[j] for j in pidx]
        hv = compute_hypervolume_2d(pareto_pts, ref_point)
        curve.append((i, hv))
    return curve

# Method display names and colors
METHOD_STYLE = {
    "gem_bo":               {"label": "BO",              "color": "#888888", "marker": "s", "ls": "--"},
    "gem_scalar_grad":      {"label": "Scalar ΔCPI",           "color": "#3498db", "marker": "v", "ls": "-"},
    "gem_scalar_noreclaim": {"label": "Scalar ΔCPI (no reclaim)", "color": "#e74c3c", "marker": "^", "ls": "-"},
    "gem_attribution":      {"label": "MSCS-guided",           "color": "#2ecc71", "marker": "D", "ls": "-"},
    "gem_coordinated":      {"label": "MSCS coordinated",      "color": "#16a085", "marker": "h", "ls": "-"},
    "gem_bo_scalar":        {"label": "BO + Scalar refine",    "color": "#9b59b6", "marker": "P", "ls": "--"},
    "gem_bo_mscs":          {"label": "BO + MSCS refine",      "color": "#e67e22", "marker": "*", "ls": "-"},
    "gem_bo_coordinated":   {"label": "BO + Coordinated",      "color": "#c0392b", "marker": "X", "ls": "-"},
}


def load_results(results_dir, label=""):
    """Load all *_results.json from flat directory."""
    pattern = os.path.join(results_dir, "*_results.json")
    files = sorted(glob.glob(pattern))
    print(f"  Searching: {pattern} → {len(files)} files")

    results = {}
    for path in files:
        fname = os.path.basename(path)
        if "comparison" in fname:
            continue
        with open(path) as f:
            data = json.load(f)
        method = data["method"]
        results[method] = data
        print(f"  Loaded: {method} ({len(data['cpis'])} evals)")
    return results


def _style_axes(ax):
    """Apply consistent Nature/Science styling to axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
    ax.xaxis.grid(False)


def _save_fig(fig, output_dir, name):
    """Save figure as both PNG (600dpi) and PDF."""
    png_path = os.path.join(output_dir, f"{name}.png")
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


# ============================================================
# Figure 1: Search Path
# ============================================================

def plot_search_path(results, benchmark, output_dir):
    """
    Plot the best trajectory from gem_full as a step-by-step search path.
    Shows: CPI, Area, and MSCS stacked bar at each step.
    """
    gem_attr = results.get("gem_attribution")
    if gem_attr is None or "trajectories" not in gem_attr:
        print("  WARNING: gem_attribution trajectories not found, skipping search path plot")
        return

    trajectories = gem_attr["trajectories"]

    # Find the trajectory with the best final CPI-Area trade-off
    best_idx = 0
    best_score = float('inf')
    for i, traj in enumerate(trajectories):
        if len(traj) < 3:
            continue
        final = traj[-1]
        score = final["cpi"] * final["area"]  # simple trade-off metric
        if score < best_score:
            best_score = score
            best_idx = i

    traj = trajectories[best_idx]
    n_steps = len(traj)

    if n_steps < 2:
        print("  WARNING: best trajectory too short, skipping")
        return

    # Check if MSCS data is available
    has_mscs = "mscs" in traj[0]

    if has_mscs:
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(3, 1, height_ratios=[1, 1, 2], hspace=0.3)
        ax_cpi = fig.add_subplot(gs[0])
        ax_area = fig.add_subplot(gs[1])
        ax_mscs = fig.add_subplot(gs[2])
    else:
        fig, (ax_cpi, ax_area) = plt.subplots(2, 1, figsize=(12, 5), height_ratios=[1, 1])

    steps = [s["step"] for s in traj]
    cpi_vals = [s["cpi"] for s in traj]
    area_vals = [s["area"] for s in traj]

    # CPI
    ax_cpi.plot(steps, cpi_vals, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    ax_cpi.set_ylabel("CPI")
    ax_cpi.set_title(f"GEM Attribution Search Path — {benchmark} (trajectory {best_idx})")
    ax_cpi.grid(True, alpha=0.3)

    # Area
    ax_area.plot(steps, area_vals, 's-', color='#3498db', linewidth=2, markersize=6)
    ax_area.set_ylabel("Area (mm²)")
    ax_area.grid(True, alpha=0.3)

    # Annotate key changes
    for s in traj[1:]:
        changes = s.get("changes", {})
        inc = [k for k, v in changes.items() if v.get("action", "").startswith("increase")]
        dec = [k for k, v in changes.items() if v.get("action") == "decrease"]
        hold = [k for k, v in changes.items() if v.get("action") == "hold"]
        step = s["step"]

        label_parts = []
        if inc:
            label_parts.append(f"↑{'，'.join(inc[:3])}")
        if dec:
            label_parts.append(f"↓{'，'.join(dec[:2])}")
        if hold:
            label_parts.append(f"={len(hold)}")

        if label_parts and step <= 10:
            label = '\n'.join(label_parts)
            ax_area.annotate(label, (step, s["area"]),
                             textcoords="offset points", xytext=(0, -30),
                             fontsize=6, ha='center', color='#555')

    # MSCS stacked bar
    if has_mscs:
        bar_width = 0.18
        stage_names = ["Decode", "Dispatch", "Issue", "Commit"]
        colors_bn = {"Base": "#cccccc", "FP": "#e74c3c", "BP": "#3498db", "SP": "#2ecc71"}

        for step_idx, s in enumerate(traj):
            mscs = np.array(s["mscs"])  # (16,)
            for stage_idx in range(4):
                base_idx = stage_idx * 4
                base_val = mscs[base_idx]
                fp_val = mscs[base_idx + 1]
                bp_val = mscs[base_idx + 2]
                sp_val = mscs[base_idx + 3]

                x = s["step"] + (stage_idx - 1.5) * bar_width
                bottom = 0
                for val, color, label in [(base_val, colors_bn["Base"], "Base"),
                                           (fp_val, colors_bn["FP"], "FP"),
                                           (bp_val, colors_bn["BP"], "BP"),
                                           (sp_val, colors_bn["SP"], "SP")]:
                    ax_mscs.bar(x, val, bar_width, bottom=bottom, color=color,
                                edgecolor='white', linewidth=0.3,
                                label=label if step_idx == 0 and stage_idx == 0 else "")
                    bottom += val

        ax_mscs.set_xlabel("Step")
        ax_mscs.set_ylabel("MSCS Proportion")
        ax_mscs.set_ylim(0, 1.05)
        ax_mscs.legend(loc='upper right', ncol=4)
        ax_mscs.grid(True, alpha=0.3, axis='y')

    if not has_mscs:
        ax_area.set_xlabel("Step")

    plt.tight_layout()
    _save_fig(fig, output_dir, f"{benchmark}_search_path")


# ============================================================
# Figure 2: Pareto Frontier (Fig 13 style)
# ============================================================

def plot_pareto_frontier(results, benchmark, output_dir):
    """Pareto frontiers of all methods overlaid on CPI vs Area plane."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for method_name, style in METHOD_STYLE.items():
        if method_name not in results:
            continue
        r = results[method_name]
        cpis = np.array(r["cpis"])
        areas = np.array(r["areas"])
        pidx = r["pareto_indices"]

        # All explored points (transparent)
        ax.scatter(areas, cpis, alpha=0.06, s=8, color=style["color"])

        # Pareto front
        p_areas = areas[pidx]
        p_cpis = cpis[pidx]
        sort_idx = np.argsort(p_areas)
        p_areas = p_areas[sort_idx]
        p_cpis = p_cpis[sort_idx]

        ax.plot(p_areas, p_cpis, style["ls"], color=style["color"],
                linewidth=2, alpha=0.9)
        ax.scatter(p_areas, p_cpis, marker=style["marker"], s=60,
                   color=style["color"], edgecolors='black', linewidth=0.6,
                   label=style["label"], zorder=5)

    ax.set_xlabel("Area (mm²)", fontweight='bold')
    ax.set_ylabel("Geometric Mean CPI", fontweight='bold')
    ax.set_title(f"Pareto Frontiers — {benchmark}", fontweight='bold', pad=10)

    _style_axes(ax)

    legend = ax.legend(loc='upper right', frameon=True, edgecolor='black',
                        facecolor='white', fancybox=False, framealpha=1.0,
                        fontsize=11, borderpad=1.0)
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    _save_fig(fig, output_dir, f"{benchmark}_pareto_frontier")


# ============================================================
# Figure 3: HV Curve (Fig 12 style)
# ============================================================

def plot_hv_curve(results, benchmark, output_dir):
    """Pareto hypervolume vs number of evaluations."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ref = get_common_ref(results)

    for method_name, style in METHOD_STYLE.items():
        if method_name not in results:
            continue
        r = results[method_name]

        cpis = r.get("cpis", [])
        areas = r.get("areas", [])
        if not cpis:
            continue

        hv_curve = recompute_hv_curve(cpis, areas, ref)

        n_evals = [h[0] for h in hv_curve]
        hvs = [h[1] for h in hv_curve]

        if len(n_evals) > 500:
            step = len(n_evals) // 200
            n_evals = n_evals[::step] + [n_evals[-1]]
            hvs = hvs[::step] + [hvs[-1]]

        ax.plot(n_evals, hvs, color=style["color"], linewidth=2,
                linestyle=style["ls"], label=style["label"])

    ax.set_xlabel("Number of Evaluations", fontweight='bold')
    ax.set_ylabel("Pareto Hypervolume", fontweight='bold')
    ax.set_title(f"Pareto HV Convergence — {benchmark}", fontweight='bold', pad=10)

    _style_axes(ax)

    legend = ax.legend(loc='lower right', frameon=True, edgecolor='black',
                        facecolor='white', fancybox=False, framealpha=1.0,
                        fontsize=11, borderpad=1.0)
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    _save_fig(fig, output_dir, f"{benchmark}_hv_curve")


# ============================================================
# Summary Table
# ============================================================

def print_summary(results):
    """Print comparison table with HV using common ref point."""
    ref = get_common_ref(results)
    print(f"\n  Ref point: ({ref[0]:.2f}, {ref[1]:.2f})")
    print(f"\n{'Method':<25s} {'Evals':>6s} {'Pareto':>7s} {'Final HV':>10s} {'Time':>8s}")
    print("-" * 60)

    for method_name in METHOD_STYLE:
        if method_name not in results:
            continue
        r = results[method_name]
        cpis = r.get("cpis", [])
        areas = r.get("areas", [])
        n_evals = len(cpis)
        wt = r.get("wall_time")
        wt_str = f"{wt/60:.1f}min" if wt else "?"

        # Recompute Pareto and HV with fixed ref point
        if cpis and areas:
            pts = list(zip(cpis, areas))
            pidx = compute_pareto_front(pts)
            pareto_pts = [pts[j] for j in pidx]
            hv = compute_hypervolume_2d(pareto_pts, ref)
            n_pareto = len(pidx)
        else:
            hv = 0.0
            n_pareto = 0

        print(f"  {METHOD_STYLE[method_name]['label']:<23s} {n_evals:>6d} {n_pareto:>7d} {hv:>10.4f} {wt_str:>8s}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Plot DSE results")
    parser.add_argument("--results-dir", required=True,
                        help="Flat directory containing *_results.json files")
    parser.add_argument("--label", default="joint",
                        help="Label for plot titles and filenames (default: joint)")
    args = parser.parse_args()

    label = args.label
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading results from {results_dir}...")
    results = load_results(results_dir, label)

    if not results:
        print("No results found!")
        sys.exit(1)

    print(f"\nGenerating plots...")

    plot_search_path(results, label, results_dir)
    plot_pareto_frontier(results, label, results_dir)
    plot_hv_curve(results, label, results_dir)
    print_summary(results)

    print(f"\nAll plots saved to {results_dir}")


if __name__ == "__main__":
    main()