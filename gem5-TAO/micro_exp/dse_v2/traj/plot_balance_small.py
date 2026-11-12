#!/usr/bin/env python3
"""Balance of Dispatch-stage stalls along walks from SMALL starts (results_small/), with the random-start
median as reference. Writes results_small/figs/fig_balance_small_{jain,entropy}.png|pdf (never overwrites).
  Jain fairness index J = (sum x)^2 / (n * sum x^2), x = Dispatch FP/BP/SP stall cycles (CPI x share), n = 3:
     J = 1 when the three stall types are equal, J = 1/3 when a single type remains.
  Normalised entropy H = -sum p ln p / ln 3 (as in fig_entropy)."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "results_small", "figs"); os.makedirs(FIG, exist_ok=True)
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})
GRIDX = np.linspace(0, 1, 21)


def load(root, b):
    fs = sorted(glob.glob(os.path.join(HERE, root, b, "part_*.npz")))
    return {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "cpi", "mscs")}


def x3(d):
    return np.stack([d["cpi"] * d["mscs"][:, 4 + k] for k in (1, 2, 3)], 1)


def jain(x):
    return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def entropy(x):
    p = x / x.sum(1, keepdims=True); return -(p * np.log(p + 1e-12)).sum(1) / np.log(3)


def curves(st, v):
    out = []
    for s in np.unique(st):
        h = v[st == s]
        out.append(np.interp(GRIDX, np.linspace(0, 1, len(h)), h) if len(h) > 1 else np.full_like(GRIDX, h[0]))
    return np.array(out)


def short(b):
    return b.split(".")[1].replace("_s", "").replace("_r", "") + ("*" if b in FP_SUITE else "")


def main():
    B = sorted(d for d in os.listdir(os.path.join(HERE, "results_small")) if d[0].isdigit())
    S = {b: load("results_small", b) for b in B}; Rr = {b: load("results", b) for b in B}
    summ = {}
    for metric, fn, ylo, ylab in (("jain", jain, 1 / 3, "Jain index (Dispatch FP/BP/SP)"),
                                  ("entropy", entropy, 0.0, "normalised entropy (Dispatch FP/BP/SP)")):
        fig, axes = plt.subplots(3, 6, figsize=(14, 7), sharex=True, sharey=True)
        rng = np.random.RandomState(1)
        for ax, b in zip(axes.ravel(), B):
            cs = curves(S[b]["start"], fn(x3(S[b]))); cr = curves(Rr[b]["start"], fn(x3(Rr[b])))
            for i in rng.choice(len(cs), 12, replace=False):
                ax.plot(GRIDX, cs[i], color=MUTED, lw=0.5, alpha=0.35)
            q25, q50, q75 = np.percentile(cs, [25, 50, 75], axis=0)
            ax.fill_between(GRIDX, q25, q75, color=BLUE, alpha=0.2, lw=0)
            ax.plot(GRIDX, q50, color=BLUE, lw=2)
            ax.plot(GRIDX, np.median(cr, 0), color=ORANGE, lw=1.4, ls="--")
            ax.set_title(short(b), fontsize=9, color=INK, loc="left")
            ax.grid(axis="y", color=GRID, lw=0.6); ax.set_ylim(ylo, 1)
            summ.setdefault(b, {})[metric] = (q50[0], q50[-1], np.median(cr, 0)[0], np.median(cr, 0)[-1])
        for ax in axes.ravel()[len(B):]:
            ax.axis("off")
        h = [plt.Line2D([], [], color=BLUE, lw=2, label="small starts: median (band = IQR, grey = 12 walks)"),
             plt.Line2D([], [], color=ORANGE, lw=1.4, ls="--", label="random starts: median")]
        axes.ravel()[-1].legend(handles=h, loc="center", frameon=False, fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("walk progress (start = 0, end = 1)")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylab, fontsize=8)
        note = "1/3 = one stall type, 1 = all equal" if metric == "jain" else "0 = one stall type, 1 = all equal"
        fig.suptitle(f"Balance of Dispatch-stage stalls along greedy walks from small designs, 1000 walks per benchmark "
                     f"({note}; * = SPECfp)", fontsize=10, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for ext in ("png", "pdf"):
            p = os.path.join(FIG, f"fig_balance_small_{metric}.{ext}")
            if os.path.exists(p):
                p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
            fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)
    print(f"\n{'bench':12s}  Jain small start->end | random start->end || entropy small start->end")
    for b in B:
        j, e = summ[b]["jain"], summ[b]["entropy"]
        print(f"{short(b):12s}  {j[0]:.2f}->{j[1]:.2f} | {j[2]:.2f}->{j[3]:.2f} || {e[0]:.2f}->{e[1]:.2f}")
    je = [summ[b]["jain"][1] for b in B]; ee = [summ[b]["entropy"][1] for b in B]
    print(f"Spearman(end Jain, end entropy) across benchmarks: {spearmanr(je, ee).correlation:.2f}; "
          f"Jain rises in {sum(summ[b]['jain'][1] > summ[b]['jain'][0] for b in B)}/17 (small starts)")


if __name__ == "__main__":
    main()
