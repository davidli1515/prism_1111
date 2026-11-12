"""
Plot gradient conflict for Exp2 OOD (averaged across benchmarks).
Usage: python plot_grad_conflict_exp2.py --json results_grad_conflict/grad_conflict_avg.json
"""
import json, argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--json", default="results_grad_conflict/grad_conflict_avg.json")
args = parser.parse_args()

with open(args.json) as f:
    raw = json.load(f)

data = raw["history"]
n_bm = raw.get("n_benchmarks", "?")
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

plt.suptitle(f"Gradient Conflict — OOD 5% (averaged over {n_bm} benchmarks)", fontsize=13, y=1.02)
plt.tight_layout()
out = args.json.replace(".json", ".png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved to {out}")