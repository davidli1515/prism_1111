#!/usr/bin/env python3
"""Bottleneck-state figures for the SMALL-start walks (results_small/), using the SAME 6 centroids as the
random-start clustering (results/state_clusters_v1.npz; nearest centroid) so the two are comparable, plus a
fresh k-means (elbow) on the small-start states as a check. Writes (never overwrites):
  results_small/state_labels_v1centroids.npz, results_small/state_clusters_small.txt,
  results_small/figs/fig_states_small.png|pdf, results_small/figs/fig_state_scatter_small.png|pdf"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

HERE = os.path.dirname(os.path.abspath(__file__))
RS, RR = os.path.join(HERE, "results_small"), os.path.join(HERE, "results")
FIG = os.path.join(RS, "figs"); os.makedirs(FIG, exist_ok=True)
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
COL = {"FP": "#2a78d6", "BP": "#eb6834", "SP": "#1baf7a"}
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})
OUT = []


def say(s=""):
    print(s); OUT.append(s)


def save(fig, name):
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"{name}.{ext}")
        if os.path.exists(p):
            p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)


def load(root, b):
    fs = sorted(glob.glob(os.path.join(root, b, "part_*.npz")))
    d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "cpi", "mscs")}
    X = np.stack([d["cpi"] * d["mscs"][:, 4 * s + k] for s in (1, 3) for k in (1, 2, 3)], 1)
    d["X"] = X / np.maximum(X.sum(1, keepdims=True), 1e-12)
    return d


def ends(st):
    f = np.r_[0, np.where(np.diff(st) != 0)[0] + 1]; return f, np.r_[f[1:] - 1, len(st) - 1]


def short(b):
    return b.split(".")[1].replace("_s", "").replace("_r", "") + ("*" if b in FP_SUITE else "")


def main():
    lab_path = os.path.join(RS, "state_labels_v1centroids.npz")
    if os.path.exists(lab_path):
        raise SystemExit(f"exists: {lab_path}")
    Z = np.load(os.path.join(RR, "state_clusters_v1.npz")); cent, k = Z["centroids"], int(Z["k"])
    B = sorted(d for d in os.listdir(RS) if d[0].isdigit())
    D = {b: load(RS, b) for b in B}
    L = {b: np.argmin(((D[b]["X"][:, None] - cent[None]) ** 2).sum(-1), 1) for b in B}
    LR = {b: Z[f"labels_{b}"] for b in B}
    Xall, Lall = np.concatenate([D[b]["X"] for b in B]), np.concatenate([L[b] for b in B])
    say(f"small-start states: {len(Xall)}; share in v1 states: " +
        ", ".join(f"S{s} {100*np.mean(Lall==s):.1f}%" for s in range(k)))
    rng = np.random.RandomState(0)
    samp = Xall[rng.choice(len(Xall), min(120000, len(Xall)), replace=False)]
    ks = list(range(2, 15)); inert = [KMeans(kk, n_init=4, random_state=0).fit(samp).inertia_ for kk in ks]
    x = (np.array(ks) - 2) / 12; y = (np.array(inert) - inert[-1]) / (inert[0] - inert[-1])
    kf = ks[int(np.argmax((1 - x) - y))]
    km = KMeans(kf, n_init=10, random_state=0).fit(samp)
    dist = np.sqrt(((km.cluster_centers_[:, None] - cent[None]) ** 2).sum(-1))
    say(f"fresh k-means on small-start states: elbow k = {kf}; nearest v1 centroid for each fresh centroid: " +
        ", ".join(f"S{int(np.argmin(r))} (dist {r.min():.3f})" for r in dist))
    sidx = rng.choice(len(Xall), 15000, replace=False)
    sil = silhouette_samples(Xall[sidx], Lall[sidx])
    say(f"silhouette with v1 centroids: {sil.mean():.2f}")

    # ---------------- fig_states_small
    fig = plt.figure(figsize=(13, 5.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.7, 1.0], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    comps = [("D", "FP"), ("D", "BP"), ("D", "SP"), ("C", "FP"), ("C", "BP"), ("C", "SP")]
    for s in range(k):
        bottom = 0
        for j, (stg, cat) in enumerate(comps):
            v = 100 * cent[s, j]
            ax.bar(s, v, bottom=bottom, color=COL[cat], width=0.72, edgecolor="white", linewidth=1.0,
                   hatch="////" if stg == "C" else None, alpha=1.0 if stg == "D" else 0.55)
            bottom += v
    ax.set_xticks(range(k)); ax.set_xticklabels([f"S{s}\n{100*np.mean(Lall==s):.0f}%" for s in range(k)])
    ax.set_ylabel("share of stall cycles (%)"); ax.set_ylim(0, 100)
    ax.set_title("(a) bottleneck states (same centroids\n as random starts; % of small-start states)", loc="left", fontsize=10, color=INK)
    leg = [Patch(facecolor=COL[c], label=f"Dispatch {c}") for c in ("FP", "BP", "SP")] + \
          [Patch(facecolor=COL[c], alpha=0.55, hatch="////", edgecolor="white", label=f"Commit {c}") for c in ("FP", "BP", "SP")]
    ax.legend(handles=leg, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)

    ax = fig.add_subplot(gs[1]); rows, labels = [], []
    say("\nper benchmark: share of walks in each state at start / end (small starts) | end (random starts)")
    for b in B:
        f, l = ends(D[b]["start"]); h = lambda lab, idx: np.bincount(lab[idx], minlength=k) / len(idx)
        fr, lr = ends(np.concatenate([np.load(p)["start"] for p in sorted(glob.glob(os.path.join(RR, b, "part_*.npz")))]))
        rows += [h(L[b], f), h(L[b], l)]; labels += [f"{short(b)}  start", "end"]
        say(f"  {short(b):11s} start " + " ".join(f"{100*v:3.0f}" for v in h(L[b], f)) + " | end " +
            " ".join(f"{100*v:3.0f}" for v in h(L[b], l)) + " | random end " + " ".join(f"{100*v:3.0f}" for v in h(LR[b], lr)))
    M = np.array(rows)
    ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k)); ax.set_xticklabels([f"S{s}" for s in range(k)])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6.5)
    for i in range(0, len(labels), 2):
        ax.axhline(i - 0.5, color="white", lw=1.5)
    for i in range(M.shape[0]):
        for j in range(k):
            if M[i, j] >= 0.15:
                ax.text(j, i, f"{100*M[i,j]:.0f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] > 0.6 else INK)
    ax.set_title("(b) share of walks in each state, small starts (* = SPECfp)", loc="left", fontsize=10, color=INK)
    ax.tick_params(length=0); [s_.set_visible(False) for s_ in ax.spines.values()]

    ax = fig.add_subplot(gs[2]); T = np.zeros((k, k))
    for b in B:
        st, lab = D[b]["start"], L[b]; same = st[1:] == st[:-1]; a, c = lab[:-1][same], lab[1:][same]; ch = a != c
        np.add.at(T, (a[ch], c[ch]), 1)
    Tp = T / np.maximum(T.sum(1, keepdims=True), 1)
    ax.imshow(Tp, cmap="Oranges", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if T[i, j] > 0:
                ax.text(j, i, f"{100*Tp[i,j]:.0f}\n({int(T[i,j])})", ha="center", va="center", fontsize=6,
                        color="white" if Tp[i, j] > 0.6 else INK)
    ax.set_xticks(range(k)); ax.set_xticklabels([f"S{s}" for s in range(k)]); ax.set_xlabel("to")
    ax.set_yticks(range(k)); ax.set_yticklabels([f"S{s}" for s in range(k)]); ax.set_ylabel("from")
    ax.set_title("(c) transitions, small starts:\n% of exits from a state (count)", loc="left", fontsize=10, color=INK)
    ax.tick_params(length=0); [s_.set_visible(False) for s_ in ax.spines.values()]
    say("\npooled transition counts (small starts), row = from:")
    for i in range(k):
        say(f"  S{i} " + " ".join(f"{int(x):6d}" for x in T[i]))
    save(fig, "fig_states_small")

    # ---------------- fig_state_scatter_small
    idx = rng.choice(len(Xall), 60000, replace=False); Xs, Ls = Xall[idx], Lall[idx]
    pca = PCA(2).fit(Xall[sidx]); P, C = pca.transform(Xs), pca.transform(cent); ev = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9)); order = rng.permutation(len(Xs))
    views = [(P, C, f"PC1 ({100*ev[0]:.0f}% var.)", f"PC2 ({100*ev[1]:.0f}% var.)", "(a) PCA of the 6-dim stall composition"),
             (Xs[:, [1, 2]], cent[:, [1, 2]], "Dispatch BP share", "Dispatch SP share", "(b) Dispatch: back-pressure vs speculation"),
             (Xs[:, [1, 3]], cent[:, [1, 3]], "Dispatch BP share", "Commit FP share", "(c) Dispatch back-pressure vs Commit starvation")]
    for ax, (pts, cp, xl, yl, title) in zip(axes, views):
        ax.scatter(pts[order, 0], pts[order, 1], c=[PAL[i] for i in Ls[order]], s=1.5, alpha=0.18, lw=0, rasterized=True)
        for s in range(k):
            ax.scatter(cp[s, 0], cp[s, 1], s=90, c=PAL[s], edgecolor="white", linewidth=2, zorder=5)
            ax.annotate(f"S{s}", (cp[s, 0], cp[s, 1]), xytext=(6, 6), textcoords="offset points", fontsize=10,
                        fontweight="bold", color=INK, zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, loc="left", fontsize=10, color=INK); ax.grid(color=GRID, lw=0.6)
    handles = [plt.Line2D([], [], marker="o", ls="", color=PAL[s], label=f"S{s} ({np.mean(Lall==s)*100:.0f}% of states)") for s in range(k)]
    fig.legend(handles=handles, loc="lower center", ncol=k, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Small-start walks: {len(Xall)//1000}k states (60k shown), labelled with the random-start centroids; "
                 f"silhouette {sil.mean():.2f}", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); save(fig, "fig_state_scatter_small")
    np.savez_compressed(lab_path, **{f"labels_{b}": L[b] for b in B})
    open(os.path.join(RS, "state_clusters_small.txt"), "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
