"""
Plot case study trajectories from saved JSON.

Usage:
    python plot_case_study.py --json results/mcf_trajectory.json
    python plot_case_study.py --json results/mcf_trajectory.json results/perlbench_trajectory.json
    python plot_case_study.py --json results/*_trajectory.json
"""

import argparse, os, sys, json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 600,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def _style_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
    ax.xaxis.grid(False)


# Per-benchmark key-decision annotations (rendered in plot_trajectory).
KEY_DECISIONS = {
    "mcf": {
        # Main box: two-color trace comparison at the PROTECT decision (step 0-1)
        "protect_lines": [
            ("scalar reclaims LQEntries, IntALU", "#3498db"),
            ("MSCS protects them", "#2ecc71"),
        ],
        "protect_box_xy": (0.30, 0.72),      # axes fraction (empty middle-upper region)
        "protect_point_xy": (29.7, 2.422),   # step 1: lines still overlap
        # Second arrow: scalar re-adds what it discarded
        "readd_text": "scalar re-adds\nLQEntries, IntALU",
        "readd_box_xy": (0.56, 0.44),
        "readd_point_xy": (37.0, 2.220),     # scalar step 6-8 region
        "readd_color": "#3498db",
        "legend_loc": "upper right",
        "waste_label": True,
    },
    "xalancbmk": {
        # DECOUPLE story: scalar oscillates RdWrPort<->SQEntries, area blows up to 73;
        # MSCS decouples via LQEntries/numIQEntries, lower CPI at 35mm2.
        "protect_lines": [
            ("MSCS decouples via LQEntries,", "#2ecc71"),
            ("numIQEntries → lower CPI at 35mm²", "#2ecc71"),
        ],
        "protect_box_xy": (0.34, 0.40),
        "protect_point_xy": (35.48, 1.564),   # MSCS DECOUPLE region (step 13-14)
        "readd_text": "scalar oscillates\nRdWrPort↔SQEntries\n→ area blows up to 73",
        "readd_box_xy": (0.50, 0.66),
        "readd_point_xy": (73.15, 1.566),     # scalar oscillation endpoint
        "readd_color": "#3498db",
        "legend_loc": "upper right",
    },
    "perlbench": {
        # Step 8+: scalar stalls (ΔCPI≈0, oscillates); MSCS decouples and continues.
        "protect_lines": [
            ("MSCS decouples via", "#2ecc71"),
            ("IntALU, commitWidth", "#2ecc71"),
            ("→ breaks plateau", "#2ecc71"),
        ],
        "protect_box_xy": (0.46, 0.40),
        "protect_point_xy": (44.80, 0.3685),   # MSCS DECOUPLE region (step 13-15)
        "readd_text": "scalar stalls:\nΔCPI ≈ 0, oscillates",
        "readd_box_xy": (0.30, 0.28),
        "readd_point_xy": (37.62, 0.3741),     # scalar oscillation (step 8-15)
        "readd_color": "#3498db",
        "legend_loc": "upper right",
    },
}


SCALAR_COLOR = "#2471a3"
MSCS_COLOR = "#1e8449"


def _save_fig(fig, output_dir, name):
    pdf_path = os.path.join(output_dir, f"{name}_final.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved: {pdf_path}")


def plot_trajectory(data, output_dir):
    """CPI vs Area trajectory: Scalar vs MSCS."""
    bname = data["bname"]
    scalar = data["scalar"]
    mscs = data["mscs"]

    fig, ax = plt.subplots(figsize=(8, 6))

    s_a = [t["area"] for t in scalar if t["area"] is not None]
    s_c = [t["cpi"] for t in scalar if t["area"] is not None]
    m_a = [t["area"] for t in mscs if t["area"] is not None]
    m_c = [t["cpi"] for t in mscs if t["area"] is not None]

    ax.plot(s_a, s_c, 'o-', color='#3498db', linewidth=2.5, markersize=9,
            label='Scalar ΔCPI', zorder=4,
            markeredgecolor='white', markeredgewidth=1.0)
    ax.plot(m_a, m_c, 'D-', color='#2ecc71', linewidth=2.5, markersize=9,
            label='MSCS-guided', zorder=5,
            markeredgecolor='white', markeredgewidth=1.0)

    # Step numbers
    for i in range(len(s_a)):
        ax.annotate(str(i), (s_a[i], s_c[i]), fontsize=5.5, color='white',
                    ha='center', va='center', fontweight='bold', zorder=6)
    for i in range(len(m_a)):
        ax.annotate(str(i), (m_a[i], m_c[i]), fontsize=5.5, color='white',
                    ha='center', va='center', fontweight='bold', zorder=7)

    # Per-benchmark key-decision annotation
    ann = KEY_DECISIONS.get(bname)
    if ann:
        # Main box: PROTECT decision, with color-coded trace lines.
        bx, by = ann["protect_box_xy"]
        lines = ann["protect_lines"]
        # Use a padded version of the longest line to guarantee the box is wide enough
        longest = max((t for t, _ in lines), key=len)
        spacer = "\n".join(longest + "MMMM" for _ in lines)
        ax.annotate(
            spacer,
            xy=ann["protect_point_xy"], xycoords='data',
            xytext=(bx, by), textcoords='axes fraction',
            fontsize=10, ha='left', va='center', color='none',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFFDF5',
                      edgecolor='black', linewidth=1.2),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                            connectionstyle="arc3,rad=0.15"),
            zorder=10,
        )
        line_dy = 0.042
        y0 = by + line_dy * (len(lines) - 1) / 2
        for k, (txt, col) in enumerate(lines):
            ax.text(bx + 0.005, y0 - k * line_dy, txt,
                    transform=ax.transAxes, fontsize=10, color=col,
                    ha='left', va='center', fontweight='bold', zorder=11)

        # Second annotation: only if readd_text is defined for this benchmark
        if "readd_text" in ann:
            rbx, rby = ann["readd_box_xy"]
            ax.annotate(
                ann["readd_text"],
                xy=ann["readd_point_xy"], xycoords='data',
                xytext=(rbx, rby), textcoords='axes fraction',
                fontsize=9.5, ha='left', va='center', color=ann["readd_color"],
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFDF5',
                          edgecolor='black', linewidth=1.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                                connectionstyle="arc3,rad=-0.2"),
                zorder=10,
            )
            # Optional "✗ waste" label
            if ann.get("waste_label", False):
                ax.text(rbx - 0.02, rby, "✗ waste", transform=ax.transAxes,
                        fontsize=11, color='red', fontweight='bold',
                        ha='right', va='center', zorder=12)

    ax.set_xlabel('Area (mm²)', fontweight='bold')
    ax.set_ylabel('CPI', fontweight='bold')
    ax.set_title(f'CPI vs Area — {bname}', fontweight='bold', pad=12)

    _style_axes(ax)

    legend_loc = (ann.get("legend_loc", "upper right") if ann else "upper right")
    legend = ax.legend(loc=legend_loc, frameon=True, edgecolor='black',
                        facecolor='white', fancybox=False, framealpha=1.0,
                        fontsize=12, borderpad=1.0)
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    _save_fig(fig, output_dir, f"{bname}_trajectory_cpi_area")


def plot_summary_table(all_data, output_dir):
    """Print summary table across all benchmarks."""
    print(f"\n{'Benchmark':<20} {'Scalar CPI':>12} {'MSCS CPI':>12} {'Δ':>8} {'Winner':>8}")
    print("-" * 62)
    for data in all_data:
        bname = data["bname"]
        sc = data["scalar"][-1]["cpi"]
        mc = data["mscs"][-1]["cpi"]
        delta = mc - sc
        winner = "MSCS" if mc < sc else "Scalar" if sc < mc else "Tie"
        print(f"  {bname:<18} {sc:>12.4f} {mc:>12.4f} {delta:>+8.4f} {winner:>8}")


def main():
    parser = argparse.ArgumentParser(description="Plot case study trajectories")
    parser.add_argument("--json", nargs="+", required=True,
                        help="Path(s) to *_trajectory.json files")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as first JSON)")
    args = parser.parse_args()

    # Expand globs
    json_files = []
    for p in args.json:
        json_files.extend(glob.glob(p))
    json_files = sorted(set(json_files))

    if not json_files:
        print("No JSON files found!")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(json_files[0]) or "."
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    for jf in json_files:
        print(f"Loading: {jf}")
        with open(jf) as f:
            data = json.load(f)
        all_data.append(data)
        plot_trajectory(data, output_dir)

    if len(all_data) > 1:
        plot_summary_table(all_data, output_dir)


if __name__ == "__main__":
    main()