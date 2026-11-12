"""
Plot Gradient Projection Decomposition Results

Compares novelty_ratio of L_const and L_mono under full vs semi supervision.

Usage:
    python plot_grad_projection.py
    python plot_grad_projection.py --full grad_projection_full_100pct.json --semi grad_projection_semi_5pct.json
"""
import json, os, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", default="grad_projection_full_100pct.json")
    parser.add_argument("--semi", default="grad_projection_semi_5pct.json")
    args = parser.parse_args()

    full = load(args.full)
    semi = load(args.semi)

    if not full and not semi:
        print("No data found. Provide --full and/or --semi JSON paths.")
        return

    # ============================================================
    # Figure 1: Novelty ratio over epochs (side by side)
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    if full:
        ax = axes[0]
        epochs = [d["epoch"] for d in full]
        ax.plot(epochs, [d.get("novelty_const", 0) for d in full], "s-",
                color="#1565C0", markersize=3, linewidth=1.5,
                label="L_const novelty")
        ax.plot(epochs, [d.get("novelty_mono", 0) for d in full], "o-",
                color="#C62828", markersize=3, linewidth=1.5,
                label="L_mono novelty")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Novelty Ratio  ||∇L_⊥|| / ||∇L||", fontsize=10)
        ax.set_title("Full Supervision (100% labeled)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    if semi:
        ax = axes[1]
        epochs = [d["epoch"] for d in semi]
        ax.plot(epochs, [d.get("novelty_const", 0) for d in semi], "s-",
                color="#1565C0", markersize=3, linewidth=1.5,
                label="L_const novelty")
        ax.plot(epochs, [d.get("novelty_mono", 0) for d in semi], "o-",
                color="#C62828", markersize=3, linewidth=1.5,
                label="L_mono novelty")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_title("Semi-Supervised (5% labeled)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    plt.suptitle("Gradient Projection: Novelty of Physics Constraints vs L_task",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("grad_projection_novelty.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved grad_projection_novelty.png")

    # ============================================================
    # Figure 2: Bar chart summary (early + late, full vs semi)
    # ============================================================

    def avg_metric(data, key, start, end):
        subset = [d for d in data if start <= d["epoch"] <= end]
        vals = [d.get(key, 0) for d in subset]
        return np.mean(vals) if vals else 0

    categories = []
    if full:
        max_ep = max(d["epoch"] for d in full)
        categories.append(("100% labeled\n(early)", full, 1, 10))
        categories.append(("100% labeled\n(late)", full, max_ep - 9, max_ep))
    if semi:
        max_ep = max(d["epoch"] for d in semi)
        categories.append(("5% labeled\n(early)", semi, 1, 10))
        categories.append(("5% labeled\n(late)", semi, max_ep - 9, max_ep))

    labels = [c[0] for c in categories]
    const_vals = [avg_metric(c[1], "novelty_const", c[2], c[3]) for c in categories]
    mono_vals  = [avg_metric(c[1], "novelty_mono",  c[2], c[3]) for c in categories]

    x = np.arange(len(labels))
    w = 0.3

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - w/2, const_vals, w, color="#1565C0", label="L_const novelty",
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, mono_vals,  w, color="#C62828", label="L_mono novelty",
                   edgecolor="white", linewidth=0.5)

    # Annotate
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=9,
                fontweight="bold", color="#1565C0")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=9,
                fontweight="bold", color="#C62828")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Novelty Ratio  ||∇L_⊥|| / ||∇L||", fontsize=11)
    ax.set_title("Gradient Novelty: Full vs Semi-Supervised",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig("grad_projection_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved grad_projection_summary.png")

    # ============================================================
    # Print summary table
    # ============================================================

    print(f"\n{'='*60}")
    print("GRADIENT PROJECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Setting':<22} {'L_const novelty':>16} {'L_mono novelty':>16}")
    print(f"  {'-'*55}")
    for label, data_src, s, e in categories:
        nc = avg_metric(data_src, "novelty_const", s, e)
        nm = avg_metric(data_src, "novelty_mono", s, e)
        tag = label.replace("\n", " ")
        print(f"  {tag:<22} {nc:>15.4f} {nm:>15.4f}")
    print(f"{'='*60}")
    print(f"\n  novelty → 1: new information beyond L_task")
    print(f"  novelty → 0: redundant with L_task")
    if full and semi:
        late_full_mono = avg_metric(full, "novelty_mono",
                                    max(d["epoch"] for d in full) - 9,
                                    max(d["epoch"] for d in full))
        late_semi_mono = avg_metric(semi, "novelty_mono",
                                    max(d["epoch"] for d in semi) - 9,
                                    max(d["epoch"] for d in semi))
        delta = late_semi_mono - late_full_mono
        print(f"\n  L_mono novelty increase (semi vs full, late): {delta:+.4f}")
        if delta > 0.05:
            print(f"  → Confirmed: L_mono provides MORE new information under semi-supervision")


if __name__ == "__main__":
    main()
