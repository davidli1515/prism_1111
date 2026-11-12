#!/usr/bin/env python3
"""BO counterparts of the greedy balance / state-scatter figures (reads results_bo/, results_small/, results/;
writes results_bo/figs/*, never overwrites).
 fig_bo_balance_{jain,entropy}: Dispatch FP/BP/SP balance vs BO iteration, per benchmark (10 seeds).
   BO has no single walk, so its "trajectory" = the incumbent: at iteration t, the point of the current Pareto
   front (all points evaluated up to t) closest in area to the greedy small-start median end area (same
   matching as analyze_bo.py). Blue = median over seeds (band = IQR, grey = each seed); teal = median balance
   of the points BO samples (window of 25 iterations); orange dashed = greedy small-start end median;
   dotted = end of the 30 random initial points.
 fig_bo_balance_stages: Jain of each stage (Decode/Dispatch/Issue/Commit) of the same incumbent, median of seeds.
 fig_bo_state_scatter: same three views as fig_state_scatter (PCA fitted on the random-start greedy states as
   there); grey = greedy random-start states, coloured = BO-sampled points labelled with the reference
   centroids; second row = BO final Pareto-front points only.
  python plot_bo.py [--scatter-only]"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
RB, RS, RR = (os.path.join(HERE, x) for x in ("results_bo", "results_small", "results"))
FIG = os.path.join(RB, "figs"); os.makedirs(FIG, exist_ok=True)
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
BLUE, ORANGE, TEAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1e", "#6b6a63", "#e4e3dc"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STAGES, SCOL = ["Decode", "Dispatch", "Issue", "Commit"], ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})
TS = np.arange(0, 500, 5)                                   # iterations at which the incumbent is evaluated
N_INIT, WIN = 30, 25


def save(fig, name):
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"{name}.{ext}")
        if os.path.exists(p):
            p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)


def x3(cpi, m, s):
    return np.stack([cpi * m[:, 4 * s + k] for k in (1, 2, 3)], 1)


def jain(x):
    return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def entropy(x):
    p = x / x.sum(1, keepdims=True); return -(p * np.log(p + 1e-12)).sum(1) / np.log(3)


def feat6(cpi, m):
    X = np.stack([cpi * m[:, 4 * s + k] for s in (1, 3) for k in (1, 2, 3)], 1)
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def pareto(cpi, area):
    o = np.lexsort((cpi, area)); best, keep = np.inf, []
    for i in o:
        if cpi[i] < best:
            keep.append(i); best = cpi[i]
    return np.array(keep)


def greedy(root, b):
    fs = sorted(glob.glob(os.path.join(root, b, "part_*.npz")))
    d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "cpi", "mscs", "area")}
    st = d["start"]; d["last"] = np.r_[np.where(np.diff(st) != 0)[0], len(st) - 1]
    return d


def short(b):
    return b.split(".")[1].replace("_s", "").replace("_r", "") + ("*" if b in FP_SUITE else "")


def incumbent(z, a_ref):
    """index of the incumbent (current-front point closest to a_ref) at every t in TS"""
    out = []
    for t in TS:
        p = pareto(z["cpi"][:t + 1], z["area"][:t + 1])
        out.append(p[np.argmin(abs(z["area"][p] - a_ref))])
    return np.array(out)


def main():
    import sys
    only_sc = "--scatter-only" in sys.argv
    B = sorted(d for d in os.listdir(RB) if d[0].isdigit())
    D = {}
    for b in B:
        g = greedy(RS, b); a_ref = np.median(g["area"][g["last"]])
        runs = [np.load(f) for f in sorted(glob.glob(os.path.join(RB, b, "seed_*.npz")))]
        D[b] = {"g": g, "runs": runs, "inc": None if only_sc else [incumbent(z, a_ref) for z in runs]}

    if not only_sc:
        balance_figs(B, D)
    scatter_fig(B, D)


def balance_figs(B, D):
    # ---------------- balance vs BO iteration (Dispatch), Jain and entropy
    print(f"{'bench':11s} metric: incumbent it0 -> it499 (median seeds) | sampled 0-30 -> 475-500 | greedy small end")
    for metric, fn, ylo, ylab in (("jain", jain, 1 / 3, "Jain index (Dispatch FP/BP/SP)"),
                                  ("entropy", entropy, 0.0, "normalised entropy (Dispatch FP/BP/SP)")):
        fig, axes = plt.subplots(3, 6, figsize=(14, 7), sharex=True, sharey=True)
        for ax, b in zip(axes.ravel(), B):
            d = D[b]; g = d["g"]
            V = [fn(x3(z["cpi"], z["mscs"], 1)) for z in d["runs"]]
            inc = np.array([v[i] for v, i in zip(V, d["inc"])])                    # seeds x TS
            for row in inc:
                ax.plot(TS, row, color=MUTED, lw=0.5, alpha=0.35)
            q25, q50, q75 = np.percentile(inc, [25, 50, 75], axis=0)
            ax.fill_between(TS, q25, q75, color=BLUE, alpha=0.2, lw=0); ax.plot(TS, q50, color=BLUE, lw=2)
            S = np.array(V)                                                           # seeds x 500
            roll = [np.median(S[:, max(0, t - WIN + 1):t + 1]) for t in TS]
            ax.plot(TS, roll, color=TEAL, lw=1.2)
            ge = np.median(fn(x3(g["cpi"][g["last"]], g["mscs"][g["last"]], 1)))
            ax.axhline(ge, color=ORANGE, lw=1.4, ls="--"); ax.axvline(N_INIT, color=MUTED, lw=0.8, ls=":")
            ax.set_title(short(b), fontsize=9, color=INK, loc="left")
            ax.grid(axis="y", color=GRID, lw=0.6); ax.set_ylim(ylo, 1)
            print(f"{short(b):11s} {metric:7s} {q50[0]:.2f} -> {q50[-1]:.2f} | {np.median(S[:, :N_INIT]):.2f} -> "
                  f"{np.median(S[:, -WIN:]):.2f} | {ge:.2f}")
        for ax in axes.ravel()[len(B):]:
            ax.axis("off")
        h = [plt.Line2D([], [], color=BLUE, lw=2, label="BO incumbent at greedy end area:\nmedian of 10 seeds (band = IQR, grey = seeds)"),
             plt.Line2D([], [], color=TEAL, lw=1.2, label=f"points BO samples (median, window {WIN})"),
             plt.Line2D([], [], color=ORANGE, lw=1.4, ls="--", label="greedy end, small starts (median)"),
             plt.Line2D([], [], color=MUTED, lw=0.8, ls=":", label="end of 30 random initial points")]
        axes.ravel()[-1].legend(handles=h, loc="center", frameon=False, fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("BO iteration")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylab, fontsize=8)
        note = "1/3 = one stall type, 1 = all equal" if metric == "jain" else "0 = one stall type, 1 = all equal"
        fig.suptitle(f"Balance of Dispatch-stage stalls along BO runs (budget 500, 10 seeds per benchmark; {note}; "
                     f"* = SPECfp)", fontsize=10, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96)); save(fig, f"fig_bo_balance_{metric}")

    # ---------------- per-stage Jain of the incumbent
    fig, axes = plt.subplots(3, 6, figsize=(14, 7), sharex=True, sharey=True)
    print("\nincumbent Jain per stage, it0 -> it499 (Dec | Disp | Iss | Com)")
    for ax, b in zip(axes.ravel(), B):
        d = D[b]; row = []
        for s in range(4):
            inc = np.array([jain(x3(z["cpi"], z["mscs"], s))[i] for z, i in zip(d["runs"], d["inc"])])
            m = np.median(inc, 0); ax.plot(TS, m, color=SCOL[s], lw=2 if s == 1 else 1.6); row.append(f"{m[0]:.2f}->{m[-1]:.2f}")
        ax.axvline(N_INIT, color=MUTED, lw=0.8, ls=":")
        ax.set_title(short(b), fontsize=9, color=INK, loc="left"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_ylim(1 / 3, 1)
        print(f"  {short(b):11s} " + " | ".join(row))
    for ax in axes.ravel()[len(B):]:
        ax.axis("off")
    h = [plt.Line2D([], [], color=SCOL[s], lw=2, label=STAGES[s]) for s in range(4)]
    axes.ravel()[-1].legend(handles=h, loc="center", frameon=False, fontsize=9, title="stage (BO incumbent, median of 10 seeds)")
    for ax in axes[-1]:
        ax.set_xlabel("BO iteration")
    for ax in axes[:, 0]:
        ax.set_ylabel("Jain index of FP/BP/SP", fontsize=8)
    fig.suptitle("Stall balance at each pipeline stage along BO runs (incumbent at the greedy end area; 1/3 = one stall type, "
                 "1 = all equal; * = SPECfp)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); save(fig, "fig_bo_balance_stages")



def scatter_fig(B, D):
    # ---------------- state scatter
    Z = np.load(os.path.join(RR, "state_clusters_v1.npz")); cent, k = Z["centroids"], int(Z["k"])
    Xg = np.concatenate([feat6(g["cpi"], g["mscs"]) for g in (greedy(RR, b) for b in B)])
    rng = np.random.RandomState(0)
    pca = PCA(2).fit(Xg[rng.choice(len(Xg), 15000, replace=False)]); ev = pca.explained_variance_ratio_
    Xg = Xg[rng.choice(len(Xg), 40000, replace=False)]
    Xb, Xf = [], []
    for b in B:
        for z in D[b]["runs"]:
            F = feat6(z["cpi"], z["mscs"]); Xb.append(F); Xf.append(F[pareto(z["cpi"], z["area"])])
    Xb, Xf = np.concatenate(Xb), np.concatenate(Xf)
    near = lambda X: np.argmin(((X[:, None] - cent[None]) ** 2).sum(-1), 1)
    Lb, Lf = near(Xb), near(Xf)
    print(f"\nBO sampled points: {len(Xb)}; state shares " + ", ".join(f"S{s} {100*np.mean(Lb==s):.1f}%" for s in range(k)))
    print(f"BO final-front points: {len(Xf)}; state shares " + ", ".join(f"S{s} {100*np.mean(Lf==s):.1f}%" for s in range(k)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.4)); order = rng.permutation(len(Xb))
    tr = lambda X, cols: pca.transform(X) if cols is None else X[:, cols]
    views = [(None, f"PC1 ({100*ev[0]:.0f}% var.)", f"PC2 ({100*ev[1]:.0f}% var.)", "PCA of the 6-dim stall composition"),
             ([1, 2], "Dispatch BP share", "Dispatch SP share", "Dispatch: back-pressure vs speculation"),
             ([1, 3], "Dispatch BP share", "Commit FP share", "Dispatch back-pressure vs Commit starvation")]
    for r, (Xp, Lp, bg, tag) in enumerate(((Xb, Lb, Xg, "all BO-sampled points (grey = greedy random-start states)"),
                                           (Xf, Lf, Xb, "BO final Pareto fronts (grey = all BO-sampled points)"))):
        o = rng.permutation(len(Xp))
        for j, (ax, (cols, xl, yl, title)) in enumerate(zip(axes[r], views)):
            G, P, C = tr(bg, cols), tr(Xp, cols), tr(cent, cols)
            ax.scatter(G[:, 0], G[:, 1], c="#cfcdc4", s=1.5, alpha=0.25, lw=0, rasterized=True)
            ax.scatter(P[o, 0], P[o, 1], c=[PAL[i] for i in Lp[o]], s=1.5 if r == 0 else 5, alpha=0.25 if r == 0 else 0.6,
                       lw=0, rasterized=True)
            for s in range(k):
                ax.scatter(C[s, 0], C[s, 1], s=90, c=PAL[s], edgecolor="white", linewidth=2, zorder=5)
                ax.annotate(f"S{s}", (C[s, 0], C[s, 1]), xytext=(6, 6), textcoords="offset points", fontsize=10, fontweight="bold",
                            color=INK, zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(color=GRID, lw=0.6)
            ax.set_title(f"({'abcdef'[3 * r + j]}) {title}" + (f"\n{tag}" if j == 0 else ""), loc="left", fontsize=10, color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", color=PAL[s], label=f"S{s} (sampled {np.mean(Lb==s)*100:.0f}% / front {np.mean(Lf==s)*100:.0f}%)")
               for s in range(k)]
    fig.legend(handles=handles, loc="lower center", ncol=k, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"BO: {len(Xb)//1000}k sampled points (17 benchmarks x 10 seeds x 500) and {len(Xf)} final-front points, "
                 f"labelled with the greedy random-start centroids", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96)); save(fig, "fig_bo_state_scatter")


if __name__ == "__main__":
    main()
