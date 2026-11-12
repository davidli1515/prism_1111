"""
Plot Exp2 OOD Ablation: Full vs L_const vs Concorde (skip L_mono alone — catastrophic)

Usage:
    python plot_exp2_ablation.py --dir /path/to/gem5/micro_exp/exp2/results_ood_ablation
"""

import os, json, glob, argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "ood_abl_*.json")))
    return [json.load(open(f)) for f in files]


def plot_bar_chart(results, save_dir):
    """4 bars per benchmark: Full, L_const only, L_mono only, Concorde. + GeoMean."""
    results = sorted([r for r in results if "concorde" in r], key=lambda x: x["target"])
    
    benchmarks = [r["target"] for r in results]
    full  = [r["full"] for r in results]
    const = [r["const_only"] for r in results]
    mono  = [min(r["mono_only"], 50) for r in results]  # clip for display
    con   = [r["concorde"] for r in results]

    # Geomean (use unclipped mono for actual value)
    gm = lambda v: np.exp(np.mean(np.log([max(x, 0.01) for x in v])))
    mono_raw = [r["mono_only"] for r in results]
    full.append(gm(full)); const.append(gm(const))
    mono.append(min(gm(mono_raw), 50)); con.append(gm(con))
    labels = benchmarks + ["GeoMean"]

    n = len(labels)
    x = np.arange(n)
    w = 0.2

    fig, ax = plt.subplots(figsize=(max(14, n * 0.9), 5.5))
    ax.bar(x - 1.5*w, full,  w, label="Full (L_task+L_const+L_mono)", color="#2196F3")
    ax.bar(x - 0.5*w, const, w, label="L_task+L_const", color="#1976D2", alpha=0.7)
    ax.bar(x + 0.5*w, mono,  w, label="L_task+L_mono", color="#FF9800")
    ax.bar(x + 1.5*w, con,   w, label="Concorde", color="#4CAF50")

    # Mark clipped bars with arrow
    for i, r in enumerate(results):
        if r["mono_only"] > 50:
            ax.annotate(f"{r['mono_only']:.0f}%", xy=(x[i]+0.5*w, 50),
                       fontsize=6, ha="center", va="bottom", color="#FF9800",
                       fontweight="bold")

    ax.axvline(x=n-1.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("MAPE (%)", fontsize=12)
    ax.set_ylim(0, 55)
    ax.set_xticks(x)
    tl = [b.replace("_", "\n") for b in benchmarks] + [r"$\bf{GeoMean}$"]
    ax.set_xticklabels(tl, fontsize=7, rotation=45, ha="right")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("OOD Generalization (5% target data): Loss Ablation", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "exp2_ood_ablation_bar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved to {path}")


def plot_full_vs_const(results, save_dir):
    """Scatter: Full vs Const-only per benchmark. Points below diagonal = L_mono helped."""
    results = sorted([r for r in results], key=lambda x: x["target"])

    full  = [r["full"] for r in results]
    const = [r["const_only"] for r in results]
    names = [r["target"] for r in results]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(const, full, s=50, c="#2196F3", zorder=3)

    # Diagonal
    lim = max(max(full), max(const)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="Equal")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    # Label points
    for i, name in enumerate(names):
        short = name.split(".")[0] if "." in name else name[:8]
        ax.annotate(short, (const[i], full[i]), fontsize=6, alpha=0.7,
                   xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("L_task + L_const only (MAPE %)", fontsize=11)
    ax.set_ylabel("Full: L_task + L_const + L_mono (MAPE %)", fontsize=11)
    ax.set_title("OOD: Does L_mono Help?\n(below diagonal = L_mono helped)", fontsize=12)
    ax.grid(alpha=0.3)

    below = sum(1 for f, c in zip(full, const) if f < c)
    ax.text(0.05, 0.95, f"L_mono helped: {below}/{len(full)} benchmarks",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(save_dir, "exp2_ood_full_vs_const.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved to {path}")


def plot_summary_hbar(results, save_dir):
    """Horizontal bar: overall average of 3 variants."""
    results_with_con = [r for r in results if "concorde" in r]

    mono_avg = np.median([r["mono_only"] for r in results])  # median to handle outliers
    items = [
        ("GEM Full\n(L_task+L_const+L_mono)", np.mean([r["full"] for r in results]), "#2196F3"),
        ("GEM\n(L_task+L_const)", np.mean([r["const_only"] for r in results]), "#1976D2"),
        ("GEM\n(L_task+L_mono)", mono_avg, "#FF9800"),
        ("Concorde\n(scalar CPI)", np.mean([r["concorde"] for r in results_with_con]), "#4CAF50"),
    ]

    labels = [it[0] for it in items]
    vals   = [it[1] for it in items]
    colors = [it[2] for it in items]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(items))
    bars = ax.barh(y, vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}%", va="center", fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Average MAPE (%)", fontsize=11)
    ax.set_title("OOD (5% target data): Loss Ablation Summary", fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "exp2_ood_ablation_summary.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Summary saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str,
                        default="/path/to/gem5/micro_exp/exp2/results_ood_ablation")
    args = parser.parse_args()

    results = load_results(args.dir)
    print(f"Loaded {len(results)} benchmarks")

    plot_bar_chart(results, args.dir)
    plot_full_vs_const(results, args.dir)
    plot_summary_hbar(results, args.dir)
    print("Done!")


if __name__ == "__main__":
    main()