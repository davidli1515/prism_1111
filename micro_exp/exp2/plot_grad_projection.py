"""
Plot gradient projection analysis: ID vs OOD.
Usage:
    python plot_grad_projection.py \
        --id results_grad_projection/proj_id.json \
        --ood results_grad_projection/proj_ood_avg.json
"""
import json, argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, default=None)
parser.add_argument("--ood", type=str, default=None)
parser.add_argument("--out", type=str, default="grad_projection_compare.png")
args = parser.parse_args()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Load data
datasets = []
if args.id:
    d = json.load(open(args.id))
    datasets.append(("In-Distribution", d["history"], "#F44336", "#2196F3"))
if args.ood:
    d = json.load(open(args.ood))
    n_bm = d.get("n_benchmarks", "")
    label = f"OOD 5% (avg {n_bm} benchmarks)" if n_bm else "OOD 5%"
    datasets.append((label, d["history"], "#FF9800", "#4CAF50"))

# Left: orthogonal ratio (unique contribution)
for label, data, color_mono, color_const in datasets:
    epochs = [d["epoch"] for d in data]
    ax1.plot(epochs, [d["ortho_ratio_mono"] for d in data], "o-", color=color_mono,
             label=f"L_mono ({label})", markersize=3, linewidth=1.5)
    ax1.plot(epochs, [d["ortho_ratio_const"] for d in data], "s--", color=color_const,
             label=f"L_const ({label})", markersize=3, linewidth=1.2, alpha=0.6)

ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Orthogonal Ratio  ||∇L_⊥|| / ||∇L||", fontsize=10)
ax1.set_title("Unique Signal (not covered by L_task)", fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
ax1.legend(fontsize=7)
ax1.grid(alpha=0.3)

# Right: projection coefficient (alignment direction)
for label, data, color_mono, color_const in datasets:
    epochs = [d["epoch"] for d in data]
    ax2.plot(epochs, [d["proj_coeff_mono"] for d in data], "o-", color=color_mono,
             label=f"L_mono ({label})", markersize=3, linewidth=1.5)

ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Projection Coefficient", fontsize=10)
ax2.set_title("L_mono → L_task: + aligned, − conflicting", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.suptitle("Gradient Projection Decomposition (PCGrad-style)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(args.out, dpi=200, bbox_inches="tight")
print(f"Saved to {args.out}")
