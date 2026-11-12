#!/usr/bin/env python3
"""GA and random-walker (RW) results next to greedy and BO (reads results_ga/, results_rw/, results_bo/,
results_small/, results/; writes results_rw/analysis_ga_rw_v1.txt and results_rw/figs/*, never overwrites).
States = nearest reference centroid (results/state_clusters_v1.npz); balance = Jain of Dispatch FP/BP/SP
stall cycles. A_ref = median end area of the greedy small-start walks (GA's area budget, BO's matched area).
 1. at A_ref, per benchmark: state mode and Jain of
      greedy small-start ends | BO front point nearest A_ref (10 seeds) | GA best feasible (10 seeds) |
      RW-512 best with area <= A_ref (10 blocks of 512 = the same budget) | RW-100k Pareto point nearest A_ref
    and best CPI with area <= A_ref of each method relative to RW-100k.
 2. control (RW-100k): in area bins, Jain of the best designs of the bin (lowest 2% CPI) vs all designs of
    the bin; at A_ref (+-5% area) the same plus the state mix -> is balance a property of optimality or of area?
 3. figures: fig_ga_balance_jain (GA incumbent = best feasible so far vs generation, like fig_bo_balance_jain),
    fig_rw_balance_vs_area (control 2 per benchmark, with greedy/BO/GA points at A_ref)."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = {k: os.path.join(HERE, v) for k, v in (("ga", "results_ga"), ("rw", "results_rw"), ("bo", "results_bo"),
                                           ("sm", "results_small"), ("rr", "results"))}
FIG = os.path.join(R["rw"], "figs"); os.makedirs(FIG, exist_ok=True)
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
BLUE, ORANGE, TEAL, PINK, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})
RW_BLOCK, N_BLOCKS, TOP = 512, 10, 0.02
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def save(fig, name):
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"{name}.{ext}")
        if os.path.exists(p):
            p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)


def jain(cpi, m):
    x = np.stack([cpi * m[:, 4 + k] for k in (1, 2, 3)], 1); return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def feat6(cpi, m):
    X = np.stack([cpi * m[:, 4 * s + k] for s in (1, 3) for k in (1, 2, 3)], 1)
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def pareto(cpi, area):
    o = np.lexsort((cpi, area)); best, keep = np.inf, []
    for i in o:
        if cpi[i] < best:
            keep.append(i); best = cpi[i]
    return np.array(keep)


def cat(fs, keys):
    return {k: np.concatenate([np.load(f)[k] for f in fs]) for k in keys}


def short(b):
    return b.split(".")[1].replace("_s", "").replace("_r", "") + ("*" if b in FP_SUITE else "")


def mode_share(lab, k):
    h = np.bincount(lab, minlength=k) / max(len(lab), 1); return np.argmax(h), h.max()


def main():
    out = os.path.join(R["rw"], "analysis_ga_rw_v1.txt")
    if os.path.exists(out):
        out = out.replace(".txt", f".{os.getpid()}.txt")
    cent = np.load(os.path.join(R["rr"], "state_clusters_v1.npz"))["centroids"]; k = len(cent)
    lab = lambda cpi, m: np.argmin(((feat6(cpi, m)[:, None] - cent[None]) ** 2).sum(-1), 1)
    B = sorted(d for d in os.listdir(R["rw"]) if d[0].isdigit())
    say("1. at A_ref: state mode (share) and mean Jain | best CPI with area <= A_ref relative to RW-100k (median of seeds)")
    say(f"{'bench':11s} {'A_ref':>5s} | {'greedy':>15s} | {'BO':>15s} | {'GA':>15s} | {'RW-512':>15s} | {'RW-100k':>10s} || "
        f"CPI: greedy  BO    GA    RW512")
    agree = {m: 0 for m in ("BO", "GA", "RW-512", "RW-100k")}
    ctrl, panel = [], {}
    for b in B:
        fs = sorted(glob.glob(os.path.join(R["sm"], b, "part_*.npz"))); g = cat(fs, ("start", "cpi", "mscs", "area"))
        last = np.r_[np.where(np.diff(g["start"]) != 0)[0], len(g["start"]) - 1]
        A = float(np.median(g["area"][last])); gc, gm, ga = g["cpi"][last], g["mscs"][last], g["area"][last]
        rw = cat(sorted(glob.glob(os.path.join(R["rw"], b, "part_*.npz"))), ("cpi", "mscs", "area"))
        rows = {"greedy": (lab(gc, gm), jain(gc, gm))}
        best = {"greedy": gc[ga <= A].min() if (ga <= A).any() else np.nan}
        # BO: front point nearest A_ref per seed
        L, J, Cb = [], [], []
        for f in sorted(glob.glob(os.path.join(R["bo"], b, "seed_*.npz"))):
            z = np.load(f); p = pareto(z["cpi"], z["area"]); i = p[np.argmin(abs(z["area"][p] - A))]
            L.append(lab(z["cpi"][[i]], z["mscs"][[i]])[0]); J.append(jain(z["cpi"][[i]], z["mscs"][[i]])[0])
            ok = z["area"] <= A; Cb.append(z["cpi"][ok].min() if ok.any() else np.nan)
        rows["BO"] = (np.array(L), np.array(J)); best["BO"] = np.nanmedian(Cb)
        # GA: best feasible per seed
        L, J, Cg, gcurves = [], [], [], []
        for f in sorted(glob.glob(os.path.join(R["ga"], b, "seed_*.npz"))):
            z = np.load(f); ok = z["feasible"]
            if ok.any():
                i = np.where(ok)[0][np.argmin(z["cpi"][ok])]
                L.append(lab(z["cpi"][[i]], z["mscs"][[i]])[0]); J.append(jain(z["cpi"][[i]], z["mscs"][[i]])[0]); Cg.append(z["cpi"][i])
            jj = jain(z["cpi"], z["mscs"]); cur, inc = np.inf, []
            for t in range(len(jj)):                                       # incumbent = best feasible so far
                if ok[t] and z["cpi"][t] < cur:
                    cur, ji = z["cpi"][t], jj[t]
                inc.append(ji if np.isfinite(cur) else np.nan)
            gcurves.append(inc)
        rows["GA"] = (np.array(L, int), np.array(J)); best["GA"] = np.median(Cg) if Cg else np.nan
        # RW-512: blocks of 512 designs, best with area <= A_ref
        L, J, Cr = [], [], []
        for s in range(N_BLOCKS):
            sl = slice(s * RW_BLOCK, (s + 1) * RW_BLOCK); c, m, a = rw["cpi"][sl], rw["mscs"][sl], rw["area"][sl]; ok = a <= A
            if ok.any():
                i = np.where(ok)[0][np.argmin(c[ok])]
                L.append(lab(c[[i]], m[[i]])[0]); J.append(jain(c[[i]], m[[i]])[0]); Cr.append(c[i])
        rows["RW-512"] = (np.array(L, int), np.array(J)); best["RW-512"] = np.median(Cr) if Cr else np.nan
        # RW-100k: Pareto point nearest A_ref
        p = pareto(rw["cpi"], rw["area"]); i = p[np.argmin(abs(rw["area"][p] - A))]
        rows["RW-100k"] = (lab(rw["cpi"][[i]], rw["mscs"][[i]]), jain(rw["cpi"][[i]], rw["mscs"][[i]]))
        okA = rw["area"] <= A; ref = rw["cpi"][okA].min() if okA.any() else np.nan
        gm_mode = mode_share(rows["greedy"][0], k)[0]
        cells = []
        for name in ("greedy", "BO", "GA", "RW-512", "RW-100k"):
            l, j = rows[name]
            if len(l) == 0:
                cells.append(f"{'-':>15s}" if name != "RW-100k" else f"{'-':>10s}"); continue
            mo, sh = mode_share(l, k)
            if name in agree:
                agree[name] += mo == gm_mode
            cells.append(f"S{mo} ({100*sh:3.0f}%) {np.mean(j):.2f}" if name != "RW-100k" else f"S{mo} {np.mean(j):.2f}")
        say(f"{short(b):11s} {A:5.1f} | " + " | ".join(f"{c:>15s}" if n < 4 else c for n, c in enumerate(cells)) + " || "
            + "  ".join(f"{best[n]/ref:.3f}" for n in ("greedy", "BO", "GA", "RW-512")))
        # control 2: area bins (quantiles of RW area), best 2% CPI of the bin vs all designs of the bin
        Jrw = jain(rw["cpi"], rw["mscs"]); Lrw = lab(rw["cpi"], rw["mscs"])
        edges = np.quantile(rw["area"], np.linspace(0, 1, 21)); mids, jall, jq, jtop = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (rw["area"] >= lo) & (rw["area"] < hi)
            if m.sum() < 200:
                continue
            c = rw["cpi"][m]; top = c <= np.quantile(c, TOP)
            mids.append(np.median(rw["area"][m])); jall.append(np.median(Jrw[m])); jq.append(np.percentile(Jrw[m], [25, 75]))
            jtop.append(np.median(Jrw[m][top]))
        m = np.abs(rw["area"] / A - 1) <= 0.05
        if m.sum() >= 50:
            c = rw["cpi"][m]; top = c <= np.quantile(c, TOP)
            ctrl.append((b, m.sum(), np.median(Jrw[m]), np.median(Jrw[m][top]), mode_share(Lrw[m], k), mode_share(Lrw[m][top], k)))
        panel[b] = dict(A=A, mids=np.array(mids), jall=np.array(jall), jq=np.array(jq), jtop=np.array(jtop),
                        pts={n: np.mean(rows[n][1]) for n in rows if len(rows[n][1])}, gcurves=np.array(gcurves, float))
    say(f"\nstate mode == greedy(small) end mode: " + ", ".join(f"{n} {v}/{len(B)}" for n, v in agree.items()))

    say(f"\n2. control on RW-100k at A_ref (+-5% area): all designs vs best {100*TOP:.0f}% CPI of the same area")
    say(f"{'bench':11s} {'n':>6s} | Jain all  best | state all       best")
    d = []
    for b, n, ja, jt, sa, st in ctrl:
        d.append(jt - ja)
        say(f"{short(b):11s} {n:6d} | {ja:.2f}  {jt:.2f} | S{sa[0]} ({100*sa[1]:3.0f}%)  S{st[0]} ({100*st[1]:3.0f}%)")
    if d:
        d = np.array(d)
        say(f"best - all Jain at A_ref: median {np.median(d):+.3f}; best more balanced in {np.sum(d > 0)}/{len(d)}")
    say("across area bins (20 RW quantile bins), best-2% minus all-designs median Jain, per benchmark (median over bins):")
    say("  " + ", ".join(f"{short(b)} {np.median(panel[b]['jtop'] - panel[b]['jall']):+.2f}" for b in B))

    # ---------------- fig_ga_balance_jain
    fig, axes = plt.subplots(3, 6, figsize=(14, 7), sharex=True, sharey=True)
    for ax, b in zip(axes.ravel(), B):
        P = panel[b]; G = P["gcurves"]
        if G.size:
            x = np.arange(G.shape[1])
            for row in G:
                ax.plot(x, row, color=MUTED, lw=0.5, alpha=0.35)
            with np.errstate(all="ignore"):
                q25, q50, q75 = np.nanpercentile(G, [25, 50, 75], axis=0)
            ax.fill_between(x, q25, q75, color=BLUE, alpha=0.2, lw=0); ax.plot(x, q50, color=BLUE, lw=2)
        ax.axhline(P["pts"]["greedy"], color=ORANGE, lw=1.4, ls="--")
        if "BO" in P["pts"]:
            ax.axhline(P["pts"]["BO"], color=TEAL, lw=1.2, ls="-.")
        ax.set_title(short(b), fontsize=9, color=INK, loc="left"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_ylim(1 / 3, 1)
    for ax in axes.ravel()[len(B):]:
        ax.axis("off")
    h = [plt.Line2D([], [], color=BLUE, lw=2, label="GA incumbent (best feasible so far):\nmedian of 10 seeds (band = IQR, grey = seeds)"),
         plt.Line2D([], [], color=ORANGE, lw=1.4, ls="--", label="greedy end, small starts (mean)"),
         plt.Line2D([], [], color=TEAL, lw=1.2, ls="-.", label="BO front at A_ref (mean of seeds)")]
    axes.ravel()[-1].legend(handles=h, loc="center", frameon=False, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("GA evaluation (16 per generation)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Jain index (Dispatch FP/BP/SP)", fontsize=8)
    fig.suptitle("Balance of Dispatch-stage stalls along ArchGym GA runs (min CPI s.t. area <= greedy end area; 512 evaluations, "
                 "10 seeds; * = SPECfp)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); save(fig, "fig_ga_balance_jain")

    # ---------------- fig_rw_balance_vs_area
    fig, axes = plt.subplots(3, 6, figsize=(14, 7), sharex=False, sharey=True)
    for ax, b in zip(axes.ravel(), B):
        P = panel[b]
        ax.fill_between(P["mids"], P["jq"][:, 0], P["jq"][:, 1], color=MUTED, alpha=0.18, lw=0)
        ax.plot(P["mids"], P["jall"], color=MUTED, lw=1.4)
        ax.plot(P["mids"], P["jtop"], color=BLUE, lw=2)
        for n, col, mk in (("greedy", ORANGE, "*"), ("BO", TEAL, "o"), ("GA", PINK, "s")):
            if n in P["pts"]:
                ax.scatter([P["A"]], [P["pts"][n]], color=col, marker=mk, s=40 if mk == "*" else 22, zorder=5, edgecolor="white", lw=0.5)
        ax.axvline(P["A"], color=MUTED, lw=0.7, ls=":")
        ax.set_title(short(b), fontsize=9, color=INK, loc="left"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_ylim(1 / 3, 1)
    for ax in axes.ravel()[len(B):]:
        ax.axis("off")
    h = [plt.Line2D([], [], color=MUTED, lw=1.4, label="all random designs of the area bin\n(median, band = IQR)"),
         plt.Line2D([], [], color=BLUE, lw=2, label=f"best {100*TOP:.0f}% CPI of the area bin (median)"),
         plt.Line2D([], [], marker="*", ls="", color=ORANGE, label="greedy end (small starts)"),
         plt.Line2D([], [], marker="o", ls="", color=TEAL, label="BO front at A_ref"),
         plt.Line2D([], [], marker="s", ls="", color=PINK, label="GA best feasible"),
         plt.Line2D([], [], color=MUTED, lw=0.7, ls=":", label="A_ref = greedy end area")]
    axes.ravel()[-1].legend(handles=h, loc="center", frameon=False, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("area (mm$^2$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Jain index (Dispatch FP/BP/SP)", fontsize=8)
    fig.suptitle("Optimality vs area: Dispatch-stall balance of 100k random designs per benchmark, best designs vs all designs "
                 "of the same area (* = SPECfp)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); save(fig, "fig_rw_balance_vs_area")
    open(out, "w").write("\n".join(OUT) + "\n"); print("->", out)


if __name__ == "__main__":
    main()
