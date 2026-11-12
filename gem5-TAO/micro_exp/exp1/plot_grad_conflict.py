"""
Plot gradient conflict analysis results.
Usage: python plot_grad_conflict.py --json grad_conflict_results.json
"""
import json, argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--json", default="grad_conflict_results.json")
args = parser.parse_args()

with open(args.json) as f:
    data = json.load(f)

epochs = [d["epoch"] for d in data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: cosine similarity
ax1.plot(epochs, [d["cos_task_mono"] for d in data], "o-", color="#F44336",
         label="cos(∇L_task, ∇L_mono)", markersize=3, linewidth=1.5)
ax1.plot(epochs, [d["cos_task_const"] for d in data], "s-", color="#2196F3",
         label="cos(∇L_task, ∇L_const)", markersize=3, linewidth=1.5)
ax1.plot(epochs, [d["cos_const_mono"] for d in data], "^-", color="#FF9800",
         label="cos(∇L_const, ∇L_mono)", markersize=3, linewidth=1.5)
ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Cosine Similarity", fontsize=11)
ax1.set_title("Gradient Direction Alignment", fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Right: gradient norms
ax2.plot(epochs, [d["norm_task"] for d in data], "o-", color="#2196F3",
         label="||∇L_task||", markersize=3, linewidth=1.5)
ax2.plot(epochs, [d["norm_const"] for d in data], "s-", color="#1976D2",
         label="||∇L_const||", markersize=3, linewidth=1.5)
ax2.plot(epochs, [d["norm_mono"] for d in data], "^-", color="#F44336",
         label="||∇L_mono||", markersize=3, linewidth=1.5)
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Gradient Norm", fontsize=11)
ax2.set_title("Gradient Magnitude", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_yscale("log")

plt.suptitle("Gradient Conflict Analysis: L_task vs L_const vs L_mono", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("grad_conflict_plot.png", dpi=200, bbox_inches="tight")
print("Saved to grad_conflict_plot.png")
