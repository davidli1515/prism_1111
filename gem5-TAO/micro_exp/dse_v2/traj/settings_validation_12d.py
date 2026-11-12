#!/usr/bin/env python3
"""State agreement between PRISM and gem5 with the four-stage states (results_small/state_clusters_12d.npz), on the
same pooled validation set as settings_validation_pooled.py (1,273 unique designs; loader imported, unchanged).
Each design is assigned to the nearest 12-dim centroid once from the PRISM MSCS and once from the gem5 MSCS.
Reports exact agreement, Cohen's kappa, exact-or-adjacent along S0-S1-S2 (frontend -> backend), per method,
confusion matrix, and a 95% bootstrap over benchmarks. Writes gem5_val4/settings_validation_12d.txt (never overwrites)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import settings_validation_pooled as P
import cluster_states_12d as Q
OUT = os.path.join(HERE, "gem5_val4", "settings_validation_12d.txt")
ORDER = {0: 0, 1: 1, 2: 2}


def labels(rows, cent):
    lab = lambda c, m: np.argmin(((Q.feat12(c, m)[:, None] - cent[None]) ** 2).sum(2), 1)
    return (lab(np.array([r["mc"] for r in rows]), np.stack([r["mm"] for r in rows])),
            lab(np.array([r["gc"] for r in rows]), np.stack([r["gm"] for r in rows])))


def stats(lm, lg, k):
    po = np.mean(lm == lg); pe = sum(np.mean(lm == s) * np.mean(lg == s) for s in range(k))
    adj = np.mean([(a == b) or (a in ORDER and b in ORDER and abs(ORDER[a] - ORDER[b]) == 1) for a, b in zip(lm, lg)])
    return 100 * po, (po - pe) / (1 - pe), 100 * adj


def main():
    if os.path.exists(OUT):
        sys.exit(f"exists, refusing to overwrite: {OUT}")
    Z = np.load(Q.OUT + ".npz"); cent, k = Z["centroids"], int(Z["k"])
    rows = P.load(); lm, lg = labels(rows, cent)
    L = [f"{len(rows)} designs; four-stage states, k = {k}",
         f"{'method':8s}{'n':>6s}{'exact':>8s}{'kappa':>8s}{'adjacent':>10s}"]
    for name in ["Greedy", "BO", "GA", "Random", "All"]:
        ii = np.arange(len(rows)) if name == "All" else np.array([i for i, r in enumerate(rows) if r["method"] == name])
        e, kp, a = stats(lm[ii], lg[ii], k)
        L.append(f"{name:8s}{len(ii):6d}{e:7.0f}%{kp:8.2f}{a:9.0f}%")
    B = sorted({r["bench"] for r in rows}); by = {b: [i for i, r in enumerate(rows) if r["bench"] == b] for b in B}
    rng = np.random.RandomState(0); bs = []
    for _ in range(1000):
        ii = np.array([i for b in rng.choice(B, len(B)) for i in by[b]]); bs.append(stats(lm[ii], lg[ii], k))
    bs = np.array(bs)
    L.append("95%% bootstrap over benchmarks: exact [%.0f, %.0f]%%, kappa [%.2f, %.2f], adjacent [%.0f, %.0f]%%"
             % (*np.percentile(bs[:, 0], [2.5, 97.5]), *np.percentile(bs[:, 1], [2.5, 97.5]), *np.percentile(bs[:, 2], [2.5, 97.5])))
    conf = np.zeros((k, k), int)
    for a, b in zip(lm, lg):
        conf[a, b] += 1
    L += ["confusion (rows PRISM, cols gem5):", "      " + "".join(f"S{s:<5d}" for s in range(k))]
    L += [f"  S{s}  " + "".join(f"{conf[s, t]:<6d}" for t in range(k)) for s in range(k)]
    L.append("per benchmark exact agreement: " + ", ".join(
        f"{b.split('.')[1]} {100*np.mean(lm[by[b]] == lg[by[b]]):.0f}%" for b in B))
    print("\n".join(L)); open(OUT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
