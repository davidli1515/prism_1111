"""
Plot gradient conflict analysis for semi-supervised setting.

Usage:
    python plot_grad_conflict_semi.py
    python plot_grad_conflict_semi.py --json grad_conflict_semi_5pct.json
    python plot_grad_conflict_semi.py --json grad_conflict_semi_5pct.json --fullsup grad_conflict_results.json
"""
import json, os, argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--json", default="grad_conflict_semi_5pct.json",
                    help="Semi-supervised gradient conflict JSON")
parser.add_argument("--fullsup", default=None,
                    help="Optional: full-supervision gradient conflict JSON for comparison")
args = parser.parse_args()

with open(args.json) as f:
    data = json.load(f)

epochs = [d["epoch"] for d in data]

has_fullsup = False

# ============================================================
# Figure 1: Cosine similarity — labeled gradients
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel 1: Labeled-side gradient alignment
ax = axes[0]
ax.plot(epochs, [d.get("cos_task_const_lab", 0) for d in data], "s-",
        color="#2196F3", markersize=3, linewidth=1.5,
        label="cos(∇L_task, ∇L_const)")
ax.plot(epochs, [d.get("cos_task_mono_lab", 0) for d in data], "o-",
        color="#F44336", markersize=3, linewidth=1.5,
        label="cos(∇L_task, ∇L_mono)")
ax.plot(epochs, [d.get("cos_const_mono_lab", 0) for d in data], "^-",
        color="#FF9800", markersize=3, linewidth=1.5,
        label="cos(∇L_const, ∇L_mono)")
ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Cosine Similarity", fontsize=11)
ax.set_title("Labeled: Task vs Physics", fontsize=12, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: Unlabeled-side gradient alignment
ax = axes[1]
ax.plot(epochs, [d.get("cos_cself_mono_unl", 0) for d in data], "^-",
        color="#FF9800", markersize=3, linewidth=1.5,
        label="cos(∇L_const_self, ∇L_mono)")
ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Cosine Similarity", fontsize=11)
ax.set_title("Unlabeled: L_const_self vs L_mono", fontsize=12, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle("Gradient Conflict Analysis — Semi-Supervised (5% labeled)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("grad_conflict_semi_cosine.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved grad_conflict_semi_cosine.png")


# ============================================================
# Figure 2: Gradient norms
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Labeled norms
ax1.plot(epochs, [d.get("norm_task", 0) for d in data], "o-",
         color="#2196F3", markersize=3, linewidth=1.5, label="||∇L_task||")
ax1.plot(epochs, [d.get("norm_const_lab", 0) for d in data], "s-",
         color="#1976D2", markersize=3, linewidth=1.5, label="||∇L_const||")
ax1.plot(epochs, [d.get("norm_mono_lab", 0) for d in data], "^-",
         color="#F44336", markersize=3, linewidth=1.5, label="||∇L_mono|| (labeled)")
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Gradient Norm", fontsize=11)
ax1.set_title("Labeled Gradient Magnitudes", fontsize=12, fontweight="bold")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax1.set_yscale("log")

# Unlabeled norms
ax2.plot(epochs, [d.get("norm_const_self", 0) for d in data], "s-",
         color="#7B1FA2", markersize=3, linewidth=1.5, label="||∇L_const_self||")
ax2.plot(epochs, [d.get("norm_mono_unl", 0) for d in data], "^-",
         color="#C62828", markersize=3, linewidth=1.5, label="||∇L_mono|| (unlabeled)")
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Gradient Norm", fontsize=11)
ax2.set_title("Unlabeled Gradient Magnitudes", fontsize=12, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_yscale("log")

plt.suptitle("Gradient Norms — Semi-Supervised (5% labeled)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("grad_conflict_semi_norms.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved grad_conflict_semi_norms.png")


# ============================================================
# Figure 3 (optional): Compare with full-supervision
# ============================================================

if args.fullsup and os.path.exists(args.fullsup):
    with open(args.fullsup) as f:
        fdata = json.load(f)

    fepochs = [d["epoch"] for d in fdata]
    # Align epoch range
    max_ep = min(max(epochs), max(fepochs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # cos(task, mono) comparison
    ax1.plot(epochs, [d.get("cos_task_mono_lab", 0) for d in data], "o-",
             color="#F44336", markersize=3, linewidth=1.5,
             label="5% labeled: cos(task, mono_lab)")
    ax1.plot(epochs, [d.get("cos_task_mono_unl", 0) for d in data], "^-",
             color="#C62828", markersize=3, linewidth=1.5,
             label="5% labeled: cos(task, mono_unl)")
    ax1.plot(fepochs, [d.get("cos_task_mono", 0) for d in fdata], "o--",
             color="#FF9800", markersize=3, linewidth=1.5, alpha=0.7,
             label="100% labeled: cos(task, mono)")
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cosine Similarity", fontsize=11)
    ax1.set_title("L_task vs L_mono: 5% vs 100%", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(right=max_ep)

    # cos(task, const) comparison
    ax2.plot(epochs, [d.get("cos_task_const_lab", 0) for d in data], "s-",
             color="#2196F3", markersize=3, linewidth=1.5,
             label="5% labeled: cos(task, const_lab)")
    ax2.plot(epochs, [d.get("cos_task_cself", 0) for d in data], "s-",
             color="#7B1FA2", markersize=3, linewidth=1.5,
             label="5% labeled: cos(task, const_self)")
    ax2.plot(fepochs, [d.get("cos_task_const", 0) for d in fdata], "s--",
             color="#FF9800", markersize=3, linewidth=1.5, alpha=0.7,
             label="100% labeled: cos(task, const)")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Cosine Similarity", fontsize=11)
    ax2.set_title("L_task vs L_const: 5% vs 100%", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(right=max_ep)

    plt.suptitle("Gradient Conflict: Semi-Supervised vs Full Supervision",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("grad_conflict_semi_vs_full.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved grad_conflict_semi_vs_full.png")

print("\nDone!")