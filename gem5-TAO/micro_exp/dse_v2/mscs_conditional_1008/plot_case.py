#!/usr/bin/env python3
"""A four-point counterfactual from the fixed surrogate, not a gem5 measurement."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
data = json.loads((HERE/"results/623.xalancbmk_s.json").read_text())
state = next(s for s in data["states"] if s["state"] == "legacy_1")
r = next(r for r in state["rows"] if r["j"] == "numIQEntries" and r["j_stride"] == 4
         and r["k"] == "decodeWidth")
v0, vj, vk, vjk = r["cpi"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
ax = axes[0]
ax.plot([90, 58], [v0, vj], "o-", label="Decode width = 8", color="#2563eb")
ax.plot([90, 58], [vk, vjk], "o-", label="Decode width = 12", color="#c2410c")
ax.set(xlabel="IQ entries", ylabel="Predicted CPI", xticks=[58, 90],
       title="Same reclaim, different future cost")
ax.annotate(f"+{100*r['current_cost']:.3f}%", (58, vj), xytext=(5, -16), textcoords="offset points", color="#2563eb")
ax.annotate(f"+{100*r['future_cost']:.3f}%", (58, vjk), xytext=(5, -16), textcoords="offset points", color="#c2410c")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=.2)
delta = 100*np.array(r["dqj"]).reshape(4, 3)
limit = np.abs(delta).max()
im = axes[1].imshow(delta, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
axes[1].set(xticks=range(3), xticklabels=["FP", "BP", "SP"], yticks=range(4),
            yticklabels=["Decode", "Dispatch", "Issue", "Commit"],
            title="MSCS change before decode grows")
for i in range(4):
    for j in range(3):
        axes[1].text(j, i, f"{delta[i,j]:+.2f}", ha="center", va="center", fontsize=10)
fig.colorbar(im, ax=axes[1], label="Conditional pressure share change (pp)")
fig.suptitle("xalancbmk: frozen-surrogate counterfactual (not gem5 validation)", fontsize=12)
for suffix in ("png", "pdf"):
    target = HERE/f"conditional_case_v2.{suffix}"
    if target.exists():
        raise FileExistsError(target)
    fig.savefig(target, dpi=180)
