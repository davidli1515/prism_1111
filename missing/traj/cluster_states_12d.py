#!/usr/bin/env python3
"""Reference bottleneck states from ALL FOUR stages (12-dim feature: FP/BP/SP stall cycles at Decode, Dispatch, Issue,
Commit, normalised to sum 1), greedy small-start designs. Procedure as cluster_states.py: 120k sample (seed 0),
elbow (kneedle) over k = 2..14 (k-means n_init 4), final k-means n_init 10.
State order: regular states by Dispatch BP fraction (frontend -> backend), then the squash state (largest SP share),
then rare states (< 2% of designs).
Also: stability (re-cluster on another 120k sample, ARI), PC1/PC2, per-state composition, independent clustering of
BO / GA / random-18k (elbow k, nearest reference state and distance), per-benchmark states (>= 5%), redundancy.
Writes results_small/state_clusters_12d.{npz,txt}; never overwrites."""
import os, sys, glob, json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results_small")
OUT = os.path.join(R, "state_clusters_12d")
ST, CAT = ("Dec", "Disp", "Iss", "Com"), ("FP", "BP", "SP")
L = []


def say(s=""):
    print(s, flush=True); L.append(s)


def feat12(cpi, m):
    X = np.stack([cpi * m[:, 4 * s + k] for s in range(4) for k in (1, 2, 3)], 1)
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def kneedle(ks, inert):
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0]); y = (np.array(inert) - inert[-1]) / (inert[0] - inert[-1])
    return ks[int(np.argmax((1 - x) - y))]


def elbow(X):
    ks = list(range(2, 15)); return kneedle(ks, [KMeans(k, n_init=4, random_state=0).fit(X).inertia_ for k in ks])


def nearest(X, C):
    out_l, out_d = [], []
    for i in range(0, len(X), 200000):
        d = np.sqrt(((X[i:i + 200000, None] - C[None]) ** 2).sum(2)); out_l.append(d.argmin(1)); out_d.append(d.min(1))
    return np.concatenate(out_l), np.concatenate(out_d)


def dbp(C):                                   # Dispatch BP fraction of Dispatch stalls (cols 3..5 = Disp FP/BP/SP)
    return C[:, 4] / np.maximum(C[:, 3:6].sum(1), 1e-12)


def load(root, b, pat):
    Z = [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, root, b, pat)))]
    return feat12(np.concatenate([z["cpi"] for z in Z]), np.concatenate([z["mscs"] for z in Z]))


def main():
    if os.path.exists(OUT + ".npz") or os.path.exists(OUT + ".txt"):
        sys.exit(f"exists, refusing to overwrite: {OUT}.*")
    B = sorted(d for d in os.listdir(R) if d[0].isdigit())
    F = {b: load("results_small", b, "part_*.npz") for b in B}; X = np.concatenate([F[b] for b in B])
    rng = np.random.RandomState(0); samp = X[rng.choice(len(X), 120000, replace=False)]
    k = elbow(samp)
    km = KMeans(k, n_init=10, random_state=0).fit(samp); C0 = km.cluster_centers_
    lab0, _ = nearest(X, C0); share0 = np.bincount(lab0, minlength=k) / len(X)
    rare = [c for c in range(k) if share0[c] < 0.02]
    rest = [c for c in range(k) if c not in rare]
    sp = {c: C0[c, [2, 5, 8, 11]].sum() for c in rest}; squash = max(rest, key=lambda c: sp[c])
    regular = sorted([c for c in rest if c != squash], key=lambda c: dbp(C0[[c]])[0])
    order = regular + [squash] + sorted(rare, key=lambda c: -share0[c])
    cent = C0[order]
    lab, dist = nearest(X, cent); share = np.bincount(lab, minlength=k) / len(X)
    say(f"greedy small-start designs: {len(X)}; 12-dim feature; elbow k = {k}")
    say("state order: regular by Dispatch BP fraction, then squash, then rare (<2%)")

    say("\ncentroid composition (% of the 12-component stall total):")
    say("        " + "".join(f"{s[:3]}-{c:<3s}" for s in ST for c in CAT))
    for s in range(k):
        say(f"  S{s} {100*share[s]:4.1f}% " + "".join(f"{100*v:7.1f}" for v in cent[s]))
    say("per state, within-stage share of BP (%), Dec/Disp/Iss/Com: " + "; ".join(
        f"S{s} " + "/".join(f"{100*cent[s, 3*i+1]/cent[s, 3*i:3*i+3].sum():.0f}" for i in range(4)) for s in range(k)))
    say("per state, within-stage share of FP (%), Dec/Disp/Iss/Com: " + "; ".join(
        f"S{s} " + "/".join(f"{100*cent[s, 3*i]/cent[s, 3*i:3*i+3].sum():.0f}" for i in range(4)) for s in range(k)))
    say("per state, SP share of the 12-component total (%): " + ", ".join(f"S{s} {100*cent[s, [2,5,8,11]].sum():.1f}" for s in range(k)))
    say(f"distance to own centroid: median {np.median(dist):.3f}, p95 {np.percentile(dist, 95):.3f}")

    l1 = KMeans(k, n_init=10, random_state=0).fit(X[np.random.RandomState(1).choice(len(X), 120000, replace=False)]).predict(X)
    l2 = KMeans(k, n_init=10, random_state=1).fit(samp).predict(X)
    say(f"stability: ARI vs reference, other 120k sample {adjusted_rand_score(lab, l1):.3f}, other k-means seed {adjusted_rand_score(lab, l2):.3f}")
    pca = PCA(2).fit(X[np.random.RandomState(3).choice(len(X), 15000, replace=False)])
    say(f"PCA: PC1 {100*pca.explained_variance_ratio_[0]:.0f}%, PC2 {100*pca.explained_variance_ratio_[1]:.0f}%; "
        "centroid PC1: " + ", ".join(f"S{s} {v:+.3f}" for s, v in enumerate(pca.transform(cent)[:, 0])))
    D = np.sqrt(((cent[:, None] - cent[None]) ** 2).sum(2))
    say("centroid distances:\n" + "\n".join("  " + " ".join(f"{v:.3f}" for v in row) for row in D))

    old = np.load(os.path.join(R, "state_clusters_v1.npz"))["centroids"]
    f6 = lambda Y: None
    say("\nindependent clustering (12-dim, 120k sample, elbow k), centroids -> nearest reference state (distance):")
    r2 = np.random.RandomState(0); ind = {}
    for name, root, pat in (("BO", "results_bo", "seed_*.npz"), ("GA", "results_ga", "seed_*.npz"), ("Random-18k", "results_rw18k", "part_*.npz")):
        Xm = np.concatenate([load(root, b, pat) for b in B])
        sm = Xm[r2.choice(len(Xm), min(120000, len(Xm)), replace=False)]
        kk = elbow(sm); cm = KMeans(kk, n_init=10, random_state=0).fit(sm).cluster_centers_
        nl, nd = nearest(cm, cent); lm, dm = nearest(Xm, cent); ind[name] = cm
        say(f"  {name}: {len(Xm)} designs, elbow k = {kk}; " + ", ".join(f"S{a} ({d:.3f})" for a, d in sorted(zip(nl, nd)))
            + f"; missing: {sorted(set(range(k)) - set(nl.tolist()))}; designs nearest each state: "
            + ", ".join(f"S{s} {100*np.mean(lm == s):.0f}%" for s in range(k))
            + f"; distance to nearest ref centroid median {np.median(dm):.3f}")

    say("\nper benchmark (greedy): share per state (%) and states with >= 5%")
    groups = {}; Lb = {}
    for b in B:
        lb, _ = nearest(F[b], cent); Lb[b] = lb; f = np.bincount(lb, minlength=k) / len(lb); n = int((f >= 0.05).sum())
        groups.setdefault(n, []).append(b.split(".")[1])
        say(f"  {b.split('.')[1]:12s} " + " ".join(f"S{s}:{100*f[s]:5.1f}" for s in range(k)) + f"  -> {n}")
    for n in sorted(groups):
        say(f"  {n} states: {', '.join(groups[n])}")
    say("benchmarks with >= 5% of their designs in each state: " + ", ".join(
        f"S{s} {sum(np.mean(Lb[b] == s) >= 0.05 for b in B)}" for s in range(k)))

    np.savez_compressed(OUT + ".npz", centroids=cent, k=k, pca_mean=pca.mean_, pca_comp=pca.components_,
                        pca_ev=pca.explained_variance_ratio_, **{f"labels_{b}": Lb[b] for b in B},
                        **{f"ind_{n}": c for n, c in ind.items()})
    open(OUT + ".txt", "w").write("\n".join(L) + "\n"); print("wrote", OUT + ".{npz,txt}")


if __name__ == "__main__":
    main()
