#!/usr/bin/env python3
"""
Four-stage variant of the Pesto-style state clustering (cluster_states.py, which uses {FP,BP,SP} x
{Dispatch, Commit}). Here: {FP,BP,SP} x {Decode, Dispatch, Issue, Commit} = 12 stall-cycle components
(CPI x share, Base removed), normalised to sum 1; same sample (120k, seed 0), elbow (kneedle, k = 2..14) and
k-means as cluster_states.py. Compared with the 6-dim reference (results/state_clusters_v1.npz):
elbow k, PC1 share, silhouette, ARI and contingency vs the reference labels (at the elbow k and at k = 6),
which stages carry the between-cluster separation, and per-benchmark end-state modes.
Writes (never overwrites): results/state_clusters12_v1.npz, results/state_clusters12_v1.txt,
results/figs/fig_states12.png|pdf
"""
import os, glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
FIG = os.path.join(R, "figs")
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
STG, CAT = ["Dec", "Disp", "Iss", "Com"], ["FP", "BP", "SP"]
COMP = [f"{s}-{c}" for s in STG for c in CAT]
COL = {"FP": "#2a78d6", "BP": "#eb6834", "SP": "#1baf7a"}
INK, MUTED = "#1f1f1e", "#6b6a63"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 300})
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def load(b):
    fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
    return {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "cpi", "mscs")}


def feat(cpi, m, stages):
    X = np.stack([cpi * m[:, 4 * s + k] for s in stages for k in (1, 2, 3)], 1)
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def kneedle(ks, inertia):
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0]); y = (np.array(inertia) - inertia[-1]) / (inertia[0] - inertia[-1])
    return ks[int(np.argmax((1 - x) - y))]


def short(b):
    return b.split(".")[1].replace("_s", "").replace("_r", "") + ("*" if b in FP_SUITE else "")


def main():
    out_npz = os.path.join(R, "state_clusters12_v1.npz")
    if os.path.exists(out_npz):
        raise SystemExit(f"exists: {out_npz}")
    Z = np.load(os.path.join(R, "state_clusters_v1.npz")); k6 = int(Z["k"])
    B = sorted(d for d in os.listdir(R) if d[0].isdigit())
    D = {b: load(b) for b in B}
    F12 = {b: feat(D[b]["cpi"], D[b]["mscs"], range(4)) for b in B}
    X12 = np.concatenate([F12[b] for b in B]); X6 = np.concatenate([feat(D[b]["cpi"], D[b]["mscs"], (1, 3)) for b in B])
    Lref = np.concatenate([Z[f"labels_{b}"] for b in B])
    rng = np.random.RandomState(0)
    si = rng.choice(len(X12), min(120000, len(X12)), replace=False); samp = X12[si]
    ks = list(range(2, 15)); inert = [KMeans(k, n_init=4, random_state=0).fit(samp).inertia_ for k in ks]
    k12 = kneedle(ks, inert)
    say(f"states pooled: {len(X12)}; 12-dim elbow k = {k12} (6-dim reference k = {k6})")
    say("inertia: " + ", ".join(f"k{k}:{x:.0f}" for k, x in zip(ks, inert)))
    sub = rng.choice(len(X12), 15000, replace=False)
    for nm, X in (("6-dim", X6), ("12-dim", X12)):
        ev = PCA(3).fit(X[sub]).explained_variance_ratio_
        say(f"PCA {nm}: PC1 {100*ev[0]:.0f}%, PC2 {100*ev[1]:.0f}%, PC3 {100*ev[2]:.0f}%")
    say(f"silhouette of the 6-dim reference labels (in 6-dim space): {silhouette_samples(X6[sub], Lref[sub]).mean():.2f}")

    res = {}
    for k in sorted({k12, k6}):
        km = KMeans(k, n_init=10, random_state=0).fit(samp)
        order = np.argsort(-km.cluster_centers_[:, 10])            # sort by Commit-BP share, as cluster_states.py
        cent = km.cluster_centers_[order]; remap = np.argsort(order)
        L = remap[km.predict(X12)]
        res[k] = (cent, L)
        say(f"\n===== 12-dim k-means, k = {k}{' (elbow)' if k == k12 else ''}")
        say(f"silhouette (12-dim space): {silhouette_samples(X12[sub], L[sub]).mean():.2f};  "
            f"ARI vs 6-dim reference labels: {adjusted_rand_score(Lref, L):.3f}")
        say("centroids (% of stall cycles):")
        say("        " + "".join(f"{c:>8s}" for c in COMP))
        for i, c in enumerate(cent):
            say(f"  T{i:<4d} " + "".join(f"{100*v:8.1f}" for v in c) + f"   ({100*np.mean(L==i):.1f}% of states)")
        tot = ((X12 - X12.mean(0)) ** 2)
        bss = np.zeros(12)
        for i in range(k):
            m = L == i; bss += m.sum() * (X12[m].mean(0) - X12.mean(0)) ** 2
        say("between-cluster separation carried by each stage (share of between-cluster SS): " +
            ", ".join(f"{s} {100*bss[3*j:3*j+3].sum()/bss.sum():.0f}%" for j, s in enumerate(STG)) +
            "  | share of total variance per stage: " +
            ", ".join(f"{s} {100*tot[:, 3*j:3*j+3].sum()/tot.sum():.0f}%" for j, s in enumerate(STG)))
        C = np.zeros((k6, k), int); np.add.at(C, (Lref, L), 1)
        say("contingency (rows = 6-dim reference S0..S5, cols = 12-dim T0..; % of the row):")
        for i in range(k6):
            say(f"  S{i} " + "".join(f"{100*C[i,j]/max(C[i].sum(),1):6.0f}" for j in range(k)) + f"   n={C[i].sum()}")
        say("per benchmark end-state mode: 6-dim reference | 12-dim")
        off, Lb = 0, {}
        for b in B:
            n = len(D[b]["start"]); Lb[b] = L[off:off + n]; off += n
        rows = []
        for b in B:
            st = D[b]["start"]; last = np.r_[np.where(np.diff(st) != 0)[0], len(st) - 1]
            h6 = np.bincount(Z[f"labels_{b}"][last], minlength=k6) / len(last); h12 = np.bincount(Lb[b][last], minlength=k) / len(last)
            rows.append((np.argmax(h6), np.argmax(h12)))
            say(f"  {short(b):11s} S{np.argmax(h6)} ({100*h6.max():3.0f}%) | T{np.argmax(h12)} ({100*h12.max():3.0f}%)")
        a6, a12 = np.array(rows).T
        same6 = a6[:, None] == a6[None]; same12 = a12[:, None] == a12[None]; iu = np.triu_indices(len(B), 1)
        say(f"benchmark grouping by end-state mode: ARI(6-dim, 12-dim) = {adjusted_rand_score(a6, a12):.2f}; "
            f"pairs grouped together in 6-dim that stay together in 12-dim: {np.sum(same6[iu] & same12[iu])}/{np.sum(same6[iu])}; "
            f"new pairs together in 12-dim only: {np.sum(~same6[iu] & same12[iu])}")
        res[k] = (cent, L, Lb)

    # ---------------- figure: 12-dim centroids at the elbow k, and contingency vs the 6-dim states
    cent, L, Lb = res[k12]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]; w = 0.8 / 4
    for i in range(k12):
        for j, s in enumerate(STG):
            bottom = 0
            for c_i, c in enumerate(CAT):
                v = 100 * cent[i, 3 * j + c_i]
                ax.bar(i + (j - 1.5) * w, v, bottom=bottom, width=w * 0.92, color=COL[c], alpha=[0.45, 1, 0.75, 0.6][j],
                       edgecolor="white", linewidth=0.4)
                bottom += v
            if i == 0:
                ax.text(i + (j - 1.5) * w, bottom + 0.5, s, ha="center", va="bottom", fontsize=6, rotation=90, color=MUTED)
    ax.set_xticks(range(k12)); ax.set_xticklabels([f"T{i}\n{100*np.mean(L==i):.0f}%" for i in range(k12)])
    ax.set_ylabel("share of all stall cycles (%)")
    ax.set_title(f"(a) 12-dim states (k = {k12} by elbow): bars per state = Dec / Disp / Iss / Com", loc="left", fontsize=10, color=INK)
    ax.legend(handles=[Patch(facecolor=COL[c], label=c) for c in CAT], frameon=False, fontsize=8, loc="upper right")
    ax = axes[1]
    C = np.zeros((k6, k12)); np.add.at(C, (Lref, L), 1); Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
    ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(k6):
        for j in range(k12):
            if Cn[i, j] >= 0.05:
                ax.text(j, i, f"{100*Cn[i,j]:.0f}", ha="center", va="center", fontsize=7, color="white" if Cn[i, j] > 0.6 else INK)
    ax.set_xticks(range(k12)); ax.set_xticklabels([f"T{j}" for j in range(k12)]); ax.set_xlabel("12-dim state")
    ax.set_yticks(range(k6)); ax.set_yticklabels([f"S{i}" for i in range(k6)]); ax.set_ylabel("6-dim reference state (Dispatch + Commit)")
    ax.set_title(f"(b) where each 6-dim state goes (% of row); ARI {adjusted_rand_score(Lref, L):.2f}", loc="left", fontsize=10, color=INK)
    ax.tick_params(length=0); [s_.set_visible(False) for s_ in ax.spines.values()]
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"fig_states12.{ext}")
        if os.path.exists(p):
            p = p.replace(f".{ext}", f".{os.getpid()}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor="white"); print("saved", p)
    np.savez_compressed(out_npz, k=k12, comp=np.array(COMP), centroids=cent,
                        **{f"labels_{b}": Lb[b] for b in B},
                        centroids_k6=res[k6][0], **{f"labels_k6_{b}": res[k6][2][b] for b in B})
    open(os.path.join(R, "state_clusters12_v1.txt"), "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
