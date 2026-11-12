#!/usr/bin/env python3
"""fig_entropy (random-start greedy walks, results/) with the gem5-checked walks drawn on top:
gem5_val/ (8 benchmarks x 2 walks: largest and median entropy increase) and gem5_val2/ part B (4 cactuBSSN
rise-then-fall walks). Per benchmark: model median and IQR over 1000 walks (as fig_entropy); each checked
walk = its model values (thin line) and gem5 values at the same states (markers), x = step / walk length.
Writes results/figs/fig_{entropy,jain}_gem5.png|pdf (never overwrites)."""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results"); FIG = os.path.join(R, "figs")
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
BLUE, INK, MUTED, GRID = "#2a78d6", "#1f1f1e", "#6b6a63", "#e4e3dc"
WCOL = {"max_dH": "#eb6834", "median_dH": "#1baf7a", "cactus": "#e87ba4"}
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


def checked_walks():
    """-> {bench: [(kind, start, steps, model (cpi,mscs), gem5 (cpi,mscs))]}"""
    out = {}
    for root, kind_of in (("gem5_val", lambda j: j["why"]), ("gem5_val2", lambda j: "cactus")):
        jobs = [j for j in json.load(open(os.path.join(HERE, root, "plan.json"))) if root == "gem5_val" or j["part"] == "B"]
        walks = {}
        for j in jobs:
            p = os.path.join(HERE, root, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            walks.setdefault((j["bench"], kind_of(j), j["idx"]), []).append((j["step"], j["model_cpi"], j["model_mscs"], r["cpi"], r["mscs16"]))
        for (b, kind, s), rows in walks.items():
            rows.sort()
            out.setdefault(b, []).append((kind, s, np.array([r[0] for r in rows]),
                                          (np.array([r[1] for r in rows]), np.array([r[2] for r in rows])),
                                          (np.array([r[3] for r in rows]), np.array([r[4] for r in rows]))))
    return out


def main():
    B = sorted(d for d in os.listdir(R) if d[0].isdigit()); W = checked_walks()
    for metric, fn, ylo in (("entropy", entropy, 0.0), ("jain", jain, 1 / 3)):
        fig, axes = plt.subplots(3, 6, figsize=(14, 7), sharex=True, sharey=True)
        for ax, b in zip(axes.ravel(), B):
            fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
            d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "step", "cpi", "mscs")}
            v = fn(x3(d["cpi"], d["mscs"])); st = d["start"]; cs, nlen = [], {}
            for s in np.unique(st):
                h = v[st == s]; nlen[int(s)] = len(h)
                cs.append(np.interp(GX, np.linspace(0, 1, len(h)), h) if len(h) > 1 else np.full_like(GX, h[0]))
            q25, q50, q75 = np.percentile(np.array(cs), [25, 50, 75], axis=0)
            ax.fill_between(GX, q25, q75, color=BLUE, alpha=0.15, lw=0); ax.plot(GX, q50, color=BLUE, lw=1.8)
            for kind, s, steps, (mc, mm), (gc, gm) in W.get(b, []):
                x = steps / max(nlen[int(s)] - 1, 1)
                ax.plot(x, fn(x3(mc, mm)), color=WCOL[kind], lw=0.9, alpha=0.9)
                ax.plot(x, fn(x3(gc, np.array(gm))), ls="none", marker="o", ms=3, mfc="white", mec=WCOL[kind], mew=1.0)
            n = len(W.get(b, []))
            ax.set_title(short(b) + (f"  (gem5: {n} walks)" if n else ""), fontsize=9, color=INK if n else MUTED, loc="left")
            ax.grid(axis="y", color=GRID, lw=0.6); ax.set_ylim(ylo, 1)
        for ax in axes.ravel()[len(B):]:
            ax.axis("off")
        h = [plt.Line2D([], [], color=BLUE, lw=1.8, label="model: median of 1000 walks (band = IQR)"),
             plt.Line2D([], [], color=WCOL["max_dH"], lw=0.9, marker="o", mfc="white", ms=3, label="largest-increase walk: model line, gem5 dots"),
             plt.Line2D([], [], color=WCOL["median_dH"], lw=0.9, marker="o", mfc="white", ms=3, label="median-increase walk: model line, gem5 dots"),
             plt.Line2D([], [], color=WCOL["cactus"], lw=0.9, marker="o", mfc="white", ms=3, label="cactuBSSN rise-then-fall walks (4)")]
        axes.ravel()[-1].legend(handles=h, loc="center", frameon=False, fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("walk progress (start = 0, end = 1)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Dispatch stall entropy" if metric == "entropy" else "Jain index (Dispatch FP/BP/SP)", fontsize=8)
        note = "0 = one stall type, 1 = all equal" if metric == "entropy" else "1/3 = one stall type, 1 = all equal"
        fig.suptitle(f"Balance of Dispatch-stage stalls along greedy walks, model vs gem5 on checked walks ({note}; * = SPECfp)",
                     fontsize=10, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for ext in ("png", "pdf"):
            p = os.path.join(FIG, f"fig_{metric}_gem5.{ext}")
            if os.path.exists(p):
                p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
            fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
