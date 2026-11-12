"""
Plot Exp2 OOD Ablation: Full vs L_const vs Concorde (skip L_mono alone — catastrophic)

Usage:
    python plot_exp2_ablation.py --dir /scratch/gem5-TAO/micro_exp/exp2/results_ood_ablation
"""

import os, json, glob, argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "ood_abl_*.json")))
    return [json.load(open(f)) for f in files]


def plot_bar_chart(results, save_dir):
    """4 bars per benchmark: Full, L_const only, L_mono only, Concorde.
    Left: GEM wins. Right: GEM loses. Divider between."""
    results = [r for r in results if "concorde" in r]

    # Compute gem_best = min(full, const) for sorting
    for r in results:
        r["_gem_best"] = min(r["full"], r["const_only"])
        r["_margin"] = r["concorde"] - r["_gem_best"]  # positive = GEM wins

    # Sort: winners first (by margin descending), then losers (by margin ascending)
    winners = sorted([r for r in results if r["_margin"] > 0], key=lambda x: -x["_margin"])
    losers  = sorted([r for r in results if r["_margin"] <= 0], key=lambda x: x["_margin"])
    results = winners + losers
    n_winners = len(winners)

    benchmarks = [r["target"] for r in results]
    full  = [r["full"] for r in results]
    const = [r["const_only"] for r in results]
    mono  = [r["mono_only"] for r in results]
    con   = [r["concorde"] for r in results]

    # Geomean
    gm = lambda v: np.exp(np.mean(np.log([max(x, 0.01) for x in v])))
    full.append(gm(full)); const.append(gm(const))
    mono.append(gm(mono)); con.append(gm(con))
    labels = benchmarks + ["GeoMean"]

    n = len(labels)
    x = np.arange(n)
    w = 0.2

    fig, ax = plt.subplots(figsize=(max(14, n * 0.9), 5.5))
    full_clip  = [min(v, 30) for v in full]
    const_clip = [min(v, 30) for v in const]
    mono_clip  = [min(v, 30) for v in mono]
    con_clip   = [min(v, 30) for v in con]
    ax.bar(x - 1.5*w, full_clip,  w, label="Full (L_task+L_const+L_mono)", color="#2196F3")
    ax.bar(x - 0.5*w, const_clip, w, label="L_task+L_const", color="#1976D2", alpha=0.7)
    ax.bar(x + 0.5*w, mono_clip,  w, label="L_task+L_mono", color="#FF9800")
    ax.bar(x + 1.5*w, con_clip,   w, label="Concorde", color="#4CAF50")

    # Mark where Full or Const beats Concorde with a star
    for i in range(len(results)):
        if full[i] < con[i]:
            ax.text(x[i] - 1.5*w, min(full[i], 29) + 0.3, "★", ha="center",
                    fontsize=8, color="#2196F3")
        if const[i] < con[i]:
            ax.text(x[i] - 0.5*w, min(const[i], 29) + 0.3, "★", ha="center",
                    fontsize=8, color="#1976D2")

    # Annotate bars exceeding 30%
    for i in range(len(results)):
        for val, offset, color in [(full[i], -1.5*w, "#2196F3"),
                                    (const[i], -0.5*w, "#1976D2"),
                                    (mono[i], 0.5*w, "#FF9800"),
                                    (con[i], 1.5*w, "#4CAF50")]:
            if val > 30:
                ax.annotate(f"{val:.1f}%", xy=(x[i]+offset, 30),
                           fontsize=5.5, ha="center", va="bottom", color=color,
                           fontweight="bold")

    # Divider: winners | losers
    if 0 < n_winners < len(results):
        ax.axvline(x=n_winners - 0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.6)
        ax.text(n_winners * 0.5 - 0.5, 31, "GEM wins ←", ha="center", fontsize=8, color="green")
        ax.text(n_winners + (len(results) - n_winners) * 0.5 - 0.5, 31,
                "→ Concorde wins", ha="center", fontsize=8, color="red")

    # Separator before GeoMean
    ax.axvline(x=n-1.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Stats: top 4 max gains + summary
    gem_gains = sorted([(r["_margin"], r["target"]) for r in results if r["_margin"] > 0], reverse=True)
    gem_losses = sorted([(abs(r["_margin"]), r["target"]) for r in results if r["_margin"] <= 0], reverse=True)

    lines = [f"GEM wins {n_winners}/{len(results)} benchmarks"]
    lines.append(f"Top gains:")
    for margin, name in gem_gains[:4]:
        short = name.replace("_s", "").replace("_r", "")
        lines.append(f"  {short}: +{margin:.1f}%")
    if gem_losses:
        lines.append(f"Max loss: {gem_losses[0][1].replace('_s','').replace('_r','')}: -{gem_losses[0][0]:.1f}%")

    stats_text = "\n".join(lines)
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=7,
            va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_ylabel("MAPE (%)", fontsize=12)
    ax.set_ylim(0, 32)
    ax.set_xticks(x)
    tl = [b.replace("_", "\n") for b in benchmarks] + [r"$\bf{GeoMean}$"]
    ax.set_xticklabels(tl, fontsize=7, rotation=45, ha="right")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.set_title("OOD Generalization (5% target data): Loss Ablation\n★ = GEM beats Concorde",
                 fontsize=12)
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
                        default="/scratch/gem5-TAO/micro_exp/exp2/results_ood_ablation")
    args = parser.parse_args()

    results = load_results(args.dir)
    print(f"Loaded {len(results)} benchmarks")

    plot_bar_chart(results, args.dir)
    plot_full_vs_const(results, args.dir)
    plot_summary_hbar(results, args.dir)
    print("Done!")


if __name__ == "__main__":
    main()