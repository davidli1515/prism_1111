#!/usr/bin/env python3
"""Case-study figure (single column, style of plot_balance_gem5_column.py) for cactuBSSN, start 840.
  (a) CPI along the greedy walk from step 9 (L1 = 256 kB): model (line) and gem5 (dashed + markers), the lowest CPI reached by
      tuning (band: model over 18k random designs to gem5 over all runs), and the end design with 16 L1D MSHRs. Four-stage state label
      of each gem5 point on top (12-dim centroids of results_small/state_clusters_12d.npz); floor band from results_rw18k.
  (b) gem5 four-stage stacks of the end design with 4 and with 16 MSHRs; Jain index above Dispatch and Commit.
Data: expert/607.cactuBSSN_s_s840_at10_port2.json (model walk), gem5_val*/runs, feature/gem5/runs, results_rw18k.
Copy of plot_case_cactuBSSN.py; only the state labels and the floor source differ.
Writes feature/figs/fig_case_cactuBSSN_12d.{png,pdf}; refuses to overwrite."""
import os, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_states_12d as Q

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'figure.dpi': 600, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'hatch.linewidth': 0.4, 'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
})
HERE = os.path.dirname(os.path.abspath(__file__)); B = "607.cactuBSSN_s"
MODEL, GEM5, FLOOR = "#2E86C1", "#E67E22", "#7F7F7F"
COMP = [("Base", "#D9D9D9"), ("FP", "#2E86C1"), ("BP", "#E67E22"), ("SP", "#27AE60")]   # FP/BP/SP as in plot_states_column.py
STAGES = ["Dec", "Disp", "Iss", "Com"]; XF = 25.6
C = np.load(Q.OUT + ".npz")["centroids"]


def state(cpi, m):
    f = Q.feat12(np.array([cpi], float), np.array([m], float))[0]; return int(np.argmin(((C - f) ** 2).sum(1)))


def jain(c):
    c = np.maximum(c, 0); return float(c.sum() ** 2 / max(3 * (c ** 2).sum(), 1e-12))


def main():
    fd = os.path.join(HERE, "feature", "figs"); os.makedirs(fd, exist_ok=True)
    outs = [os.path.join(fd, f"fig_case_cactuBSSN_12d.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    W = json.load(open(os.path.join(HERE, "expert", "607.cactuBSSN_s_s840_at10_port2.json")))["original"]
    G = {}
    for f in glob.glob(os.path.join(HERE, "gem5_val4", "runs", B, "p840_s*", "result.json")):
        G[int(os.path.basename(os.path.dirname(f)).split("_s")[1])] = json.load(open(f))
    F = json.load(open(os.path.join(HERE, "feature", "gem5", "runs", B, "p840_s23_mshr16", "result.json")))
    fs = sorted(glob.glob(os.path.join(HERE, "results_rw18k", B, "part_*.npz")))
    mmin = float(min(np.load(f)["cpi"].min() for f in fs))
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "gem5_val*", "runs", B, "*", "result.json"))]
    gmin = min(r["cpi"] for r in runs if r.get("cpi"))
    print(f"floor: model {mmin:.3f} over 18k, gem5 {gmin:.3f} over {sum(1 for r in runs if r.get('cpi'))} runs; +16 MSHRs {F['cpi']:.3f} ({100 * (F['cpi'] / gmin - 1):+.1f}%)")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.4, 1.8), gridspec_kw={"width_ratios": [1.25, 1]})
    k0 = 9; xm = np.arange(k0, len(W)); gk = sorted(k for k in G if k >= k0)
    ax.axhspan(mmin, gmin, color=FLOOR, alpha=0.55, lw=0, zorder=1)
    ax.plot(xm, [W[k]["cpi"] for k in xm], color=MODEL, lw=1.3, zorder=3)
    ax.plot(gk, [G[k]["cpi"] for k in gk], color=GEM5, lw=1.1, ls="--", marker="o", ms=2.6, mfc=GEM5, mec="black", mew=0.4, zorder=4)
    ax.plot([XF], [F["cpi"]], ls="none", marker="o", ms=3.6, mfc=GEM5, mec="black", mew=0.5, zorder=5)
    ax.axvline(24.3, color="black", lw=0.7, ls=":", zorder=1)
    ax.set_xlim(7.9, 26.9); ax.set_ylim(0.5, 1.72)
    ax.set_xticks([9, 13, 17, 21, XF]); ax.set_xticklabels(["9", "13", "17", "21", "+16\nMSHRs"])
    ax.set_xlabel("Search step", fontweight="bold", labelpad=-5, x=0.4); ax.set_ylabel("CPI", fontweight="bold", labelpad=2)
    ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
    seq = [(k, state(G[k]["cpi"], G[k]["mscs16"])) for k in gk] + [(XF, state(F["cpi"], F["mscs16"]))]
    print("gem5 states:", " ".join(f"{k}:S{s}" for k, s in seq))
    i = 0
    while i < len(seq):                                     # one label per run of equal states
        j = i
        while j + 1 < len(seq) and seq[j + 1][1] == seq[i][1] and seq[j + 1][0] <= 23:
            j += 1
        xl = {9: 9.0, 10: 11.0}.get(seq[i][0], (seq[i][0] + seq[j][0]) / 2)     # steps 9 and 10 are adjacent: move the second label aside
        ax.text(xl, 1.66, f"S{seq[i][1]}", ha="center", va="center", fontsize=6, fontweight="bold")
        i = j + 1
    ax.text(15.8, 1.3, "window grows", ha="center", va="bottom", fontsize=6)
    ax.text(22.2, 1.14, "ports", ha="center", va="bottom", fontsize=6)
    ax.text(8.8, 0.96, "tuning floor", ha="left", va="top", fontsize=6, color="#404040")
    ax.text(XF - 0.1, F["cpi"] + 0.06, f"\u2212{100 * (1 - F['cpi'] / gmin):.0f}%", ha="center", va="bottom", fontsize=6, fontweight="bold")
    ax.text(0.03, 0.04, "(a)", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")

    w = 0.38; A = np.array(G[23]["mscs16"]) * G[23]["cpi"]; D = np.array(F["mscs16"]) * F["cpi"]
    for j, m in enumerate((A, D)):
        for s in range(4):
            bot = 0
            for c, (_, col) in enumerate(COMP):
                bx.bar(s + (j - 0.5) * w * 1.1, m[4 * s + c], w, bottom=bot, color=col, edgecolor="black", lw=0.4, hatch="////" if j else None, zorder=2); bot += m[4 * s + c]
    top = max(A[4 * s:4 * s + 4].sum() for s in range(4))
    for s in (1, 3):
        j0, j1 = jain(A[4 * s + 1:4 * s + 4]), jain(D[4 * s + 1:4 * s + 4]); print(f"{STAGES[s]} Jain {j0:.2f} to {j1:.2f}")
        bx.text(s, top * 1.04, f"J {j0:.2f}\n\u2192 {j1:.2f}", ha="center", va="bottom", fontsize=5.8, linespacing=0.95)
    for s in range(4):
        print(STAGES[s], "4 MSHRs", np.round(A[4 * s:4 * s + 4], 2), "16 MSHRs", np.round(D[4 * s:4 * s + 4], 2))
    bx.set_xticks(range(4)); bx.set_xticklabels(STAGES, fontsize=6.8); bx.set_ylabel("Cycles / inst", fontweight="bold", labelpad=2); bx.set_ylim(0, top * 1.46)
    bx.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
    bx.text(0.04, 0.96, "(b)", transform=bx.transAxes, ha="left", va="top", fontweight="bold")
    for x in (ax, bx):
        for sp in x.spines.values():
            sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)
    h = [plt.Line2D([], [], color=GEM5, lw=1.1, ls="--", marker="o", ms=2.6, mfc=GEM5, mec="black", mew=0.4, label="gem5"), plt.Line2D([], [], color=MODEL, lw=1.3, label="Model")]
    h += [Patch(facecolor=cc, edgecolor="black", lw=0.4, label=cn) for cn, cc in COMP[:3]] + [Patch(facecolor="white", edgecolor="black", lw=0.4, hatch="////", label="+16 MSHRs")]
    fig.tight_layout(w_pad=0.7, rect=(0, 0.1, 1, 1))
    fig.legend(handles=h, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=True, edgecolor="black", fancybox=False, framealpha=1,
               columnspacing=0.55, handlelength=1.3, handletextpad=0.25, borderpad=0.3, fontsize=6.3).get_frame().set_linewidth(0.8)
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
