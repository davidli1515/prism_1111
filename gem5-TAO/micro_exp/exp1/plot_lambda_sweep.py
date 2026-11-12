"""
Plot λ_mono sweep: MAPE vs λ_mono value.
Usage: python plot_lambda_sweep.py --dir results/exp1/lambda_sweep
"""
import os, json, glob, argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="results/exp1/lambda_sweep")
args = parser.parse_args()

files = sorted(glob.glob(os.path.join(args.dir, "lambda_mono_*.json")))
points = []
for f in files:
    r = json.load(open(f))
    points.append((r["lambda_mono"], r["overall_mape"]))
points.sort()

lams = [p[0] for p in points]
mapes = [p[1] for p in points]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(len(lams)), mapes, "o-", color="#2196F3", markersize=8, linewidth=2)

# Annotate each point
for i, (l, m) in enumerate(zip(lams, mapes)):
    ax.annotate(f"{m:.2f}%", (i, m), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9)

ax.set_xticks(range(len(lams)))
ax.set_xticklabels([str(l) for l in lams], fontsize=10)
ax.set_xlabel("λ_mono", fontsize=12)
ax.set_ylabel("MAPE (%)", fontsize=12)
ax.set_title("In-Distribution Accuracy vs λ_mono\n(L_task + L_const fixed, λ_const=1.0)", fontsize=12)
ax.grid(alpha=0.3)

# Highlight λ=0 baseline
ax.axhline(y=mapes[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(len(lams)-1, mapes[0]+0.05, f"λ_mono=0 baseline: {mapes[0]:.2f}%",
        ha="right", fontsize=8, color="gray")

plt.tight_layout()
out = os.path.join(args.dir, "lambda_sweep.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved to {out}")
