#!/usr/bin/env python3
"""Robustness of the bottleneck-state clustering (reference = results/state_clusters_v1.npz, k = 6).
 A. Bootstrap: 30 x (100 random start ids out of the 1000 random-start walks, all 17 benchmarks) ->
    elbow k, k=6 k-means, centroids matched to the reference (Hungarian), ARI of the induced labels vs the
    reference labels on ALL states, agreement of each benchmark's end-state mode.
 B. Fresh data: 100 NEW random starts per benchmark (results_seed2/, seed 3026): labels with the reference
    centroids, independent elbow k-means matched to the reference, silhouette, per-benchmark end-state
    distribution vs the 1000-walk run (mode agreement, total-variation distance), transition structure.
Writes results_seed2/robust_clusters.txt (never overwrites)."""
import os, glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
R, R2 = os.path.join(HERE, "results"), os.path.join(HERE, "results_seed2")
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def load(root, b):
    fs = sorted(glob.glob(os.path.join(root, b, "part_*.npz")))
    d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "cpi", "mscs")}
    X = np.stack([d["cpi"] * d["mscs"][:, 4 * s + k] for s in (1, 3) for k in (1, 2, 3)], 1)
    d["X"] = X / np.maximum(X.sum(1, keepdims=True), 1e-12)
    return d


def ends(st):
    f = np.r_[0, np.where(np.diff(st) != 0)[0] + 1]; return f, np.r_[f[1:] - 1, len(st) - 1]


def elbow(X, rs):
    ks = list(range(2, 15)); inert = [KMeans(k, n_init=3, random_state=rs).fit(X).inertia_ for k in ks]
    x = (np.array(ks) - 2) / 12; y = (np.array(inert) - inert[-1]) / (inert[0] - inert[-1])
    return ks[int(np.argmax((1 - x) - y))]


def match(C, ref):
    D = np.sqrt(((C[:, None] - ref[None]) ** 2).sum(-1)); r, c = linear_sum_assignment(D)
    perm = np.empty(len(ref), int); perm[r] = c                # fitted cluster r -> reference state c
    return D[r, c], perm


def nearest(X, C):
    return np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)


def trans(st, lab, k):
    T = np.zeros((k, k)); same = st[1:] == st[:-1]; a, c = lab[:-1][same], lab[1:][same]; ch = a != c
    np.add.at(T, (a[ch], c[ch]), 1); return T / np.maximum(T.sum(1, keepdims=True), 1)


def main():
    out = os.path.join(R2, "robust_clusters.txt")
    if os.path.exists(out):
        out = out.replace(".txt", f".{os.getpid()}.txt")
    Z = np.load(os.path.join(R, "state_clusters_v1.npz")); ref, k = Z["centroids"], int(Z["k"])
    B = sorted(d for d in os.listdir(R) if d[0].isdigit())
    D = {b: load(R, b) for b in B}
    Lref = {b: Z[f"labels_{b}"] for b in B}
    Xall, Lall = np.concatenate([D[b]["X"] for b in B]), np.concatenate([Lref[b] for b in B])
    mode_ref = {b: np.argmax(np.bincount(Lref[b][ends(D[b]["start"])[1]], minlength=k)) for b in B}
    Tref = np.zeros((k, k))
    for b in B:
        st, lab = D[b]["start"], Lref[b]; same = st[1:] == st[:-1]; a, c = lab[:-1][same], lab[1:][same]; ch = a != c
        np.add.at(Tref, (a[ch], c[ch]), 1)
    Tref = Tref / np.maximum(Tref.sum(1, keepdims=True), 1)

    # ---------------- A. bootstrap of 100 starts
    rng = np.random.RandomState(0); reps = []
    say("A. bootstrap: 30 x 100 random start ids (of 1000) from the random-start walks, all 17 benchmarks")
    for r in range(30):
        ids = set(rng.choice(1000, 100, replace=False).tolist())
        Xs = np.concatenate([D[b]["X"][np.isin(D[b]["start"], list(ids))] for b in B])
        sub = Xs[rng.choice(len(Xs), min(40000, len(Xs)), replace=False)]
        kk = elbow(sub, r)
        km = KMeans(k, n_init=6, random_state=r).fit(sub)
        dist, perm = match(km.cluster_centers_, ref)
        lab_all = perm[km.predict(Xall)]
        ari = adjusted_rand_score(Lall, lab_all)
        modes = sum(np.argmax(np.bincount(perm[km.predict(D[b]["X"][ends(D[b]["start"])[1]])], minlength=k)) == mode_ref[b]
                    for b in B)
        reps.append((kk, dist.mean(), dist.max(), ari, modes))
    A = np.array(reps)
    say(f"   elbow k: {dict(zip(*np.unique(A[:,0].astype(int), return_counts=True)))}  (reference k = {k})")
    say(f"   matched centroid distance: mean {A[:,1].mean():.3f}, worst-state mean {A[:,2].mean():.3f} (max {A[:,2].max():.3f})")
    say(f"   ARI vs reference labels on all 540k states: median {np.median(A[:,3]):.3f} (min {A[:,3].min():.3f})")
    say(f"   benchmarks whose end-state mode matches the reference: median {np.median(A[:,4]):.0f}/17 (min {A[:,4].min():.0f})")

    # ---------------- B. fresh starts
    B2 = sorted(d for d in os.listdir(R2) if d[0].isdigit() and glob.glob(os.path.join(R2, d, "part_*.npz")))
    say(f"\nB. fresh data: 100 new random starts (seed 3026), {len(B2)} benchmarks")
    if not B2:
        open(out, "w").write("\n".join(OUT) + "\n"); return
    E = {b: load(R2, b) for b in B2}
    X2 = np.concatenate([E[b]["X"] for b in B2]); L2 = nearest(X2, ref)
    say(f"   states: {len(X2)}; share per reference state: " + ", ".join(f"S{s} {100*np.mean(L2==s):.1f}%" for s in range(k))
        + "   (1000-walk run: " + ", ".join(f"S{s} {100*np.mean(Lall==s):.1f}%" for s in range(k)) + ")")
    sub = X2[rng.choice(len(X2), min(40000, len(X2)), replace=False)]
    kk = elbow(sub, 0); km = KMeans(k, n_init=10, random_state=0).fit(sub); dist, perm = match(km.cluster_centers_, ref)
    say(f"   independent k-means on the new data: elbow k = {kk}; k=6 centroids matched to reference: "
        + ", ".join(f"S{perm[i]} {dist[i]:.3f}" for i in np.argsort(perm)))
    say(f"   ARI(independent clustering, reference-centroid labels) on the new states: {adjusted_rand_score(L2, perm[km.predict(X2)]):.3f}")
    sidx = rng.choice(len(X2), min(15000, len(X2)), replace=False)
    say(f"   silhouette with reference centroids: {silhouette_samples(X2[sidx], L2[sidx]).mean():.2f} (1000-walk run 0.45)")
    say("   per benchmark end-state distribution: new 100 starts vs 1000-walk run (mode match, total-variation distance)")
    tv, mm = [], 0
    for b in B2:
        l2 = nearest(E[b]["X"], ref)
        p = np.bincount(l2[ends(E[b]["start"])[1]], minlength=k) / len(ends(E[b]["start"])[1])
        q = np.bincount(Lref[b][ends(D[b]["start"])[1]], minlength=k) / len(ends(D[b]["start"])[1])
        t = 0.5 * np.abs(p - q).sum(); tv.append(t); m = np.argmax(p) == np.argmax(q); mm += m
        say(f"     {b.split('.')[1]:12s} new end mode S{np.argmax(p)} ({100*p.max():.0f}%) vs S{np.argmax(q)} ({100*q.max():.0f}%)  "
            f"match {'yes' if m else 'NO '}  TV {t:.2f}")
    say(f"   end-state mode matches: {mm}/{len(B2)}; total-variation distance median {np.median(tv):.2f}")
    T2 = np.zeros((k, k))
    for b in B2:
        st, lab = E[b]["start"], nearest(E[b]["X"], ref); same = st[1:] == st[:-1]; a, c = lab[:-1][same], lab[1:][same]; ch = a != c
        np.add.at(T2, (a[ch], c[ch]), 1)
    T2n = T2 / np.maximum(T2.sum(1, keepdims=True), 1)
    m = (Tref > 0) | (T2n > 0)
    say(f"   transition probabilities vs 1000-walk run: correlation {np.corrcoef(Tref[m], T2n[m])[0,1]:.2f}; "
        f"dominant exit per state same: {sum(np.argmax(Tref[i]) == np.argmax(T2n[i]) for i in range(k) if T2[i].sum() > 0)}"
        f"/{sum(T2[i].sum() > 0 for i in range(k))}")
    open(out, "w").write("\n".join(OUT) + "\n"); print("->", out)


if __name__ == "__main__":
    main()
