#!/usr/bin/env python3
"""
Pesto-style clustering of trajectory states (single-benchmark runs, all 17 benchmarks pooled).
Feature per state (Pesto): stall cycles {FP, BP, SP} x {Dispatch, Commit} = CPI x share (Base removed),
normalised to sum 1 (composition). k-means; k by the elbow (kneedle) over k = 2..14, fitted on a sample,
then every state is assigned. A state is named by the components needed to cover 80% of its centroid
(D/C = Dispatch/Commit; FP = starve, BP = stall, SP = flush).
Outputs (never overwritten): results/state_clusters_v1.npz (centroids, per-benchmark labels per state) and
results/state_clusters_v1.txt:
  (a) centroids; (b) per benchmark: share of all states / starts / ends in each bottleneck state;
  (c) transitions: pooled matrix, per-benchmark most frequent collapsed state sequences, #states per walk;
  (d) benchmark clusters by state distribution vs by transition matrix.
"""
import os, glob, json
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
COMP = ["D-FP", "D-BP", "D-SP", "C-FP", "C-BP", "C-SP"]
OUT = []


def say(s=""):
    print(s); OUT.append(s)


def load(tag, keys=("start", "cpi", "mscs", "kind")):
    fs = sorted(glob.glob(os.path.join(R, tag, "part_*.npz")))
    return {k: np.concatenate([np.load(f)[k] for f in fs]) for k in keys}


def feat(cpi, m):
    X = np.stack([cpi * m[:, 4 * s + k] for s in (1, 3) for k in (1, 2, 3)], 1)      # D-FP,D-BP,D-SP,C-FP,C-BP,C-SP
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12), X.sum(1)


def kneedle(ks, inertia):
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0]); y = (np.array(inertia) - inertia[-1]) / (inertia[0] - inertia[-1])
    return ks[int(np.argmax((1 - x) - y))]


def name(c):
    o = np.argsort(-c); acc, parts = 0, []
    for i in o:
        parts.append(COMP[i]); acc += c[i]
        if acc >= 0.8:
            break
    return "+".join(parts)


def main():
    out_npz = os.path.join(R, "state_clusters_v1.npz")
    if os.path.exists(out_npz):
        raise SystemExit(f"exists: {out_npz}")
    B = sorted(d for d in os.listdir(R) if d[0].isdigit())
    D = {b: load(b) for b in B}
    F = {b: feat(D[b]["cpi"], D[b]["mscs"])[0] for b in B}
    Xall = np.concatenate([F[b] for b in B])
    rng = np.random.RandomState(0)
    samp = Xall[rng.choice(len(Xall), min(120000, len(Xall)), replace=False)]
    ks = list(range(2, 15)); inert = []
    for k in ks:
        inert.append(KMeans(k, n_init=4, random_state=0).fit(samp).inertia_)
    k = kneedle(ks, inert)
    km = KMeans(k, n_init=10, random_state=0).fit(samp)
    order = np.argsort(-km.cluster_centers_[:, 4])                 # sort states by Commit-BP share
    cent = km.cluster_centers_[order]; remap = np.argsort(order)
    names = [f"S{i}:{name(c)}" for i, c in enumerate(cent)]
    say(f"states pooled: {len(Xall)}; elbow k = {k} (inertia {', '.join(f'{x:.0f}' for x in inert)})")
    say("\n(a) bottleneck states (centroid composition of stall cycles, %):")
    say("      " + "".join(f"{c:>7s}" for c in COMP))
    for n_, c in zip(names, cent):
        say(f"  {n_:22s}" + "".join(f"{100*v:7.1f}" for v in c))
    L = {b: remap[km.predict(F[b])] for b in B}

    say("\n(b) share of states in each bottleneck state  [all states | walk starts | walk ends]")
    say("  bench           " + "".join(f"{'S'+str(i):>6s}" for i in range(k)))
    dist_all = {}
    for b in B:
        st = D[b]["start"]; first = np.r_[0, np.where(np.diff(st) != 0)[0] + 1]; last = np.r_[first[1:] - 1, len(st) - 1]
        h = lambda idx: np.bincount(L[b][idx], minlength=k) / len(idx)
        dist_all[b] = h(np.arange(len(st)))
        for tag, v in (("all", dist_all[b]), ("start", h(first)), ("end", h(last))):
            say(f"  {b.split('.')[1]:10s} {tag:5s}" + "".join(f"{100*x:6.0f}" for x in v))

    say("\n(c) transitions along walks (consecutive repeats collapsed)")
    T = np.zeros((k, k)); seqs, nvis, Tb = {}, {}, {}
    for b in B:
        st, lab, kind = D[b]["start"], L[b], D[b]["kind"]
        Tb[b] = np.zeros((k, k)); seqs[b] = Counter(); nv = []
        for s in np.unique(st):
            l = lab[st == s]
            col = [l[0]] + [x for p, x in zip(l[:-1], l[1:]) if x != p]
            for a, c in zip(col[:-1], col[1:]):
                Tb[b][a, c] += 1
            seqs[b]["→".join(f"S{x}" for x in col)] += 1; nv.append(len(set(col)))
        T += Tb[b]; nvis[b] = np.mean(nv)
    say("  pooled transition counts (row = from, col = to):")
    say("        " + "".join(f"{'S'+str(i):>7s}" for i in range(k)))
    for i in range(k):
        say(f"  S{i:<4d} " + "".join(f"{int(x):7d}" for x in T[i]))
    say("\n  per benchmark: mean #distinct states per walk; share of walks that never change state; top-3 sequences")
    for b in B:
        tot = sum(seqs[b].values()); stay = sum(n for s_, n in seqs[b].items() if "→" not in s_) / tot
        top = ", ".join(f"{s_} ({n/tot:.2f})" for s_, n in seqs[b].most_common(3))
        say(f"  {b.split('.')[1]:10s} {nvis[b]:.2f}  stay {stay:.2f} | {top}")

    say("\n(d) benchmark clusters (k=4)")
    Bs = [b.split('.')[1] + ("*" if b in FP_SUITE else "") for b in B]
    A = np.array([dist_all[b] for b in B])
    Tn = np.array([(Tb[b] / max(Tb[b].sum(), 1)).ravel() for b in B])
    for nm, X, met in (("by state distribution", A, "euclidean"), ("by transition matrix", Tn, "euclidean")):
        c4 = fcluster(linkage(X, "average", metric=met), 4, "maxclust")
        say(f"  {nm}: " + " | ".join(", ".join(x for x, y in zip(Bs, c4) if y == c) for c in sorted(set(c4))))
    say("  (* = SPECfp)")
    np.savez_compressed(out_npz, centroids=cent, names=np.array(names), k=k, comp=np.array(COMP),
                        **{f"labels_{b}": L[b] for b in B})
    open(out_npz.replace(".npz", ".txt"), "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
