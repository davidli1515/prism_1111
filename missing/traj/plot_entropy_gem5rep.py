#!/usr/bin/env python3
"""fig_entropy (random-start greedy walks, results/) with ONE gem5-checked walk per benchmark (gem5_val3/: the
walk closest to the benchmark's median entropy curve, 12 evenly spaced states). Per benchmark: model median and
IQR over 1000 walks (module globals R/FIG/V3/TAG/LABEL select the start set); the representative walk = model line + gem5 markers at the same states, x = step / walk
length. Writes results/figs/fig_{entropy,jain}_gem5rep.png|pdf (never overwrites)."""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results"); FIG = os.path.join(R, "figs")
V3 = os.path.join(HERE, "gem5_val3")
TAG, LABEL = "gem5rep", "random starts"
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})
GX = np.linspace(0, 1, 21)


def x3(cpi, m):
    return np.stack([cpi * m[:, 4 + k] for k in (1, 2, 3)], 1)


def entropy(x):
    p = x / x.sum(1, keepdims=True); return -(p * np.log(p + 1e-12)).sum(1) / np.log(3)


def jain(x):
    return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def short(b):
    return b.split(".")[1].replace("_s", "").replace("_r", "") + ("*" if b in FP_SUITE else "")


def main():
    jobs = [j for pf in ("plan.json", "plan_peak.json") if os.path.exists(os.path.join(V3, pf))
            for j in json.load(open(os.path.join(V3, pf)))]; W = {}
    for j in jobs:
        p = os.path.join(V3, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
        if os.path.exists(p):
            r = json.load(open(p)); W.setdefault(j["bench"], []).append((j["step"], j["walk_len"], j["model_cpi"], j["model_mscs"], r["cpi"], r["mscs16"]))
    B = sorted(d for d in os.listdir(R) if d[0].isdigit())
    for metric, fn, ylo in (("entropy", entropy, 0.0), ("jain", jain, 1 / 3)):
        fig, axes = plt.subplots(3, 6, figsize=(14, 7), sharex=True, sharey=True)
        for ax, b in zip(axes.ravel(), B):
            fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
            d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "cpi", "mscs")}
            v = fn(x3(d["cpi"], d["mscs"])); st = d["start"]; cs = []
            for s in np.unique(st):
                h = v[st == s]; cs.append(np.interp(GX, np.linspace(0, 1, len(h)), h) if len(h) > 1 else np.full_like(GX, h[0]))
            q25, q50, q75 = np.percentile(np.array(cs), [25, 50, 75], axis=0)
            ax.fill_between(GX, q25, q75, color=BLUE, alpha=0.15, lw=0); ax.plot(GX, q50, color=BLUE, lw=1.8)
            rows = sorted(W.get(b, []))
            if rows:
                x = np.array([r[0] for r in rows]) / max(rows[0][1] - 1, 1)
                mv = fn(x3(np.array([r[2] for r in rows]), np.array([r[3] for r in rows])))
                gv = fn(x3(np.array([r[4] for r in rows]), np.array([r[5] for r in rows])))
                ax.plot(x, mv, color=ORANGE, lw=1.0)
                ax.plot(x, gv, ls="none", marker="o", ms=3.5, mfc="white", mec=ORANGE, mew=1.1)
                ax.text(0.98, 0.04, f"MAE {np.mean(abs(mv-gv)):.2f}", transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
            ax.set_title(short(b), fontsize=9, color=INK, loc="left")
            ax.grid(axis="y", color=GRID, lw=0.6); ax.set_ylim(ylo, 1)
        for ax in axes.ravel()[len(B):]:
            ax.axis("off")
        h = [plt.Line2D([], [], color=BLUE, lw=1.8, label="model: median of 1000 walks (band = IQR)"),
             plt.Line2D([], [], color=ORANGE, lw=1.0, label="representative walk: model"),
             plt.Line2D([], [], color=ORANGE, ls="none", marker="o", mfc="white", ms=3.5, label="representative walk: gem5")]
        axes.ravel()[-1].legend(handles=h, loc="center", frameon=False, fontsize=7.5,
                                title="representative walk = closest\nto the median curve", title_fontsize=7.5)
        for ax in axes[-1]:
            ax.set_xlabel("walk progress (start = 0, end = 1)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Dispatch stall entropy" if metric == "entropy" else "Jain index (Dispatch FP/BP/SP)", fontsize=8)
        note = "0 = one stall type, 1 = all equal" if metric == "entropy" else "1/3 = one stall type, 1 = all equal"
        fig.suptitle(f"Balance of Dispatch-stage stalls along greedy walks ({LABEL}): model, and gem5 on one representative walk per "
                     f"benchmark ({note}; * = SPECfp)", fontsize=10, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for ext in ("png", "pdf"):
            p = os.path.join(FIG, f"fig_{metric}_{TAG}.{ext}")
            if os.path.exists(p):
                p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
            fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
