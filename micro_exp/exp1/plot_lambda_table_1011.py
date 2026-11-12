#!/usr/bin/env python3
"""
White-background table of the loss-weight sweep (100% labels, mean of 3 seeds, lam_const = 1
unless noted): CPI MAPE, MSCS component error (MAE of FP/BP/SP), cross-stage Base
inconsistency (% spread of the 4 predicted Base values), monotonicity violation rate
(% predictions where FP or SP increases, or BP decreases, from Decode to Commit by > 0.01).
Data: results/lambda_sweep (lam_task <= 10) and results/lambda_probe (lam_task = 30).
Usage: python plot_lambda_table_1011.py
"""
import glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/path/to/gem5/micro_exp/exp1/results"
OUT = f"{R}/lambda_table_1011.png"
f = lambda x: f"{x:g}".replace(".", "p")


def load(t, c, m):
    ps = glob.glob(f"{R}/lambda_sweep/t{f(t)}_c{f(c)}_m{f(m)}_s*.json") + \
         glob.glob(f"{R}/lambda_probe/t{f(t)}_c{f(c)}_m{f(m)}_s*.json")
    S = [json.load(open(p))["selected"] for p in ps]
    if len(S) != 3:
        return None
    return (np.mean([s["mape_mean4"] for s in S]),
            np.mean([np.mean([s["mae"][k] for k in ("FP", "BP", "SP")]) for s in S]),
            np.mean([s["base_spread_rel"] for s in S]) * 100,
            np.mean([s["mono_viol_gt0.01"] for s in S]))


rows, chosen = [], None
for t in [0.1, 0.3, 1, 3, 10, 30]:
    for m in [0, 0.01, 0.1]:
        v = load(t, 1, m)
        if v is None:
            continue
        if (t, m) == (10, 0):
            chosen = len(rows)
        rows.append([f"{t:g}", "1", f"{m:g}", f"{v[0]:.2f}", f"{v[1]:.3f}", f"{v[2]:.2f}", f"{v[3]:.1f}"])
v = load(1, 0, 0.01)
rows.append(["1", "0", "0.01", f"{v[0]:.2f}", f"{v[1]:.3f}", f"{v[2]:.2f}", f"{v[3]:.1f}"])

cols = ["λ_task", "λ_const", "λ_mono", "CPI\nMAPE (%)", "Component\nerror (MAE)",
        "Base\ninconsistency (%)", "Monotonicity\nviolations (%)"]
WIDTHS = [0.09, 0.09, 0.09, 0.13, 0.16, 0.19, 0.19]
fig, ax = plt.subplots(figsize=(9.2, 0.33 * len(rows) + 1.0), facecolor="white")
ax.axis("off")
tb = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center", colWidths=WIDTHS)
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.35)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#BBBBBB"); cell.set_linewidth(0.6); cell.set_facecolor("white")
    if r == 0:
        cell.set_text_props(weight="bold"); cell.set_height(cell.get_height() * 1.7)
    elif r - 1 == chosen:
        cell.set_text_props(weight="bold"); cell.set_facecolor("#EEF3FB")
    elif r == len(rows):
        cell.set_text_props(style="italic")
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
print(f"Saved {OUT} (+ .pdf)")
for r in rows:
    print(" | ".join(r))
