#!/usr/bin/env python3
"""Case-study figure (single column, style of plot_balance_gem5_column.py) for an expert intervention on a greedy walk.
  (a) CPI along the search: original greedy vs expert-intervened greedy; lines = model, dashed + markers = gem5.
      The intervention counts as one step (expert step k is drawn at search step k + 1); both start from the same design.
  (b) gem5 four-stage stacks (cycles per instruction: Base / FP / BP / SP) of both walks at one search step.
Data: expert/<bench>_s<start>_<tag>.json (expert_walk.py), gem5_val4 (original), expert/gem5 (gem5_expert.py).
Writes expert/figs/fig_expert_<name>.{png,pdf}; refuses to overwrite.
  python plot_expert_case.py expert/631.deepsjeng_s_s761_lockI_L1up.json deepsjeng --at 4"""
import os, sys, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'figure.dpi': 600, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
})
HERE = os.path.dirname(os.path.abspath(__file__))
GREEDY, EXPERT = "#7F7F7F", "#C0392B"
COMP = [("Base", "#D9D9D9"), ("FP", "#2E86C1"), ("BP", "#E67E22"), ("SP", "#27AE60")]
STAGES = ["Dec", "Disp", "Iss", "Com"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("walk"); ap.add_argument("name"); ap.add_argument("--at", type=int, default=4)
    a = ap.parse_args()
    W = json.load(open(a.walk)); b, s0 = W["bench"], W["start"]
    fd = os.path.join(HERE, "expert", "figs"); os.makedirs(fd, exist_ok=True)
    outs = [os.path.join(fd, f"fig_expert_{a.name}.{e}") for e in ("png", "pdf")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    O, E = W["original"], W["expert"]
    xo = np.arange(len(O)); xe = np.r_[0, np.arange(len(E)) + 1]
    mo = np.array([r["cpi"] for r in O]); me = np.r_[O[0]["cpi"], [r["cpi"] for r in E]]
    go = {k: load(os.path.join(HERE, "gem5_val4", "runs", b, f"p{s0}_s{k:02d}", "result.json")) for k in range(len(O))}
    ge = {k + 1: load(os.path.join(HERE, "expert", "gem5", "runs", b, f"p{s0}_exp_s{k:02d}", "result.json")) for k in range(len(E))}
    ge[0] = go[0]
    go = {k: v for k, v in go.items() if v}; ge = {k: v for k, v in ge.items() if v}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.4, 1.75), gridspec_kw={"width_ratios": [1.25, 1]})
    ax.plot(xo, mo, color=GREEDY, lw=1.2, zorder=2); ax.plot(xe, me, color=EXPERT, lw=1.2, zorder=3)
    for g, col in ((go, GREEDY), (ge, EXPERT)):
        k = sorted(g); ax.plot(k, [g[i]["cpi"] for i in k], color=col, lw=0.9, ls="--", marker="o", ms=2.6, mfc=col, mec="black", mew=0.4, zorder=4)
    ax.axvline(a.at, color="black", lw=0.6, ls=":", zorder=1)
    ax.annotate("expert\ndecision", xy=(1, me[1] - 0.02), xytext=(0.5, 0.42), fontsize=6.5, color=EXPERT, va="bottom", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.6, color=EXPERT, shrinkA=0, shrinkB=1))
    ax.set_xlabel("Search step", fontweight="bold"); ax.set_ylabel("CPI", fontweight="bold")
    ax.set_xlim(-0.5, max(xo[-1], xe[-1]) + 0.5); ax.set_ylim(0.4, ax.get_ylim()[1])
    ax.text(0.96, 0.93, "(a)", transform=ax.transAxes, ha="right", va="top", fontweight="bold")
    ax.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)

    w = 0.36
    for j, (lab, g) in enumerate((("Greedy", go.get(a.at)), ("Expert", ge.get(a.at)))):
        if g is None:
            continue
        m = np.array(g["mscs16"]) * g["cpi"]
        for s in range(4):
            bot = 0
            for c, (cn, cc) in enumerate(COMP):
                v = m[4 * s + c]
                bx.bar(s + (j - 0.5) * w * 1.08, v, w, bottom=bot, color=cc, edgecolor="black", lw=0.4,
                       hatch="////" if lab == "Expert" else None, zorder=2)
                bot += v
            if s == 0:
                bx.text(s + (j - 0.5) * w * 1.08, bot + 0.02, lab[0], ha="center", va="bottom", fontsize=6.5)
        print(f"step {a.at} {lab:6s} gem5 CPI {g['cpi']:.3f} | " + "  ".join(f"{STAGES[s]} " + "/".join(f"{m[4*s+c]:.2f}" for c in range(4)) for s in range(4)))
    bx.set_xticks(range(4)); bx.set_xticklabels(STAGES); bx.set_ylabel("Cycles / inst", fontweight="bold")
    bx.set_ylim(0, bx.get_ylim()[1] * 1.22)
    bx.text(0.5, 0.95, f"step {a.at}", transform=bx.transAxes, ha="center", va="top", fontsize=7)
    bx.text(0.96, 0.93, "(b)", transform=bx.transAxes, ha="right", va="top", fontweight="bold")
    bx.yaxis.grid(True, linestyle=":", color="#A0A0A0", alpha=0.8, zorder=0)
    for x in (ax, bx):
        for sp in x.spines.values():
            sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)
    h = [plt.Line2D([], [], color=GREEDY, lw=1.2, label="Greedy"), plt.Line2D([], [], color=EXPERT, lw=1.2, label="Expert"),
         plt.Line2D([], [], color="black", lw=0.9, ls="--", marker="o", ms=2.6, mfc="white", mec="black", mew=0.4, label="gem5"),
         plt.Line2D([], [], color="black", lw=1.2, label="Model")]
    h += [Patch(facecolor=cc, edgecolor="black", lw=0.4, label=cn) for cn, cc in COMP]
    h += [Patch(facecolor="white", edgecolor="black", lw=0.4, hatch="////", label="Expert (b)")]
    fig.tight_layout(w_pad=0.8, rect=(0, 0.2, 1, 1))
    fig.legend(handles=h, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=True, edgecolor="black", fancybox=False,
               framealpha=1, columnspacing=0.7, handlelength=1.5, handletextpad=0.3, borderpad=0.3).get_frame().set_linewidth(0.8)
    for k in sorted(set(go) | set(ge)):
        print(f"  step {k:2d}  gem5 CPI greedy {go[k]['cpi'] if k in go else float('nan'):.3f}  expert {ge[k]['cpi'] if k in ge else float('nan'):.3f}")
    for p in outs:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
