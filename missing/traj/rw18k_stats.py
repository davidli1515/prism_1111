#!/usr/bin/env python3
"""Numbers that depended on the 100k random designs, recomputed with the first 18k per benchmark (results_rw18k),
next to the 100k value for reference:
  1. per benchmark: states with >= 5% of the designs, pooling greedy small-start + BO + GA + random-18k (each design once)
  2. static balance control: at A_ref (+-5% area), median Dispatch Jain of the best 2% CPI minus all designs
     (same definition as analyze_ga_rw.py control 2)
  3. cactuBSSN case: CPI floor, designs within 5% of the floor, their ROB range, median Commit BP share of CPI,
     their states; at the largest L1, CPI gain of ROB top vs bottom quartile (median CPI)
Read-only. Writes results_rw18k/rw18k_stats.txt (never overwrites)."""
import os, sys, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_rw18k", "rw18k_stats.txt")
CENT = np.load(os.path.join(HERE, "results_small", "state_clusters_v1.npz"))["centroids"]; K = len(CENT)
P17 = json.load(open(os.path.join(HERE, "results_ga", "644.nab_s", "meta.json")))["params"]
ROB, L1 = P17.index("numROBEntries"), P17.index("l1_size")
L = []


def say(s=""):
    print(s, flush=True); L.append(s)


def feat(cpi, m):
    X = np.stack([cpi * m[:, 4 * s + k] for s in (1, 3) for k in (1, 2, 3)], 1)
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def label(X):
    return np.concatenate([np.argmin(((X[i:i + 200000, None] - CENT[None]) ** 2).sum(2), 1) for i in range(0, len(X), 200000)])


def jain(cpi, m):
    x = np.stack([cpi * m[:, 4 + k] for k in (1, 2, 3)], 1); return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def cat(root, b, pat, keys=("cpi", "mscs", "area", "cfg")):
    Z = [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, root, b, pat)))]
    return {k: np.concatenate([z[k] for z in Z]) for k in keys}


def main():
    if os.path.exists(OUT):
        sys.exit(f"exists, refusing to overwrite: {OUT}")
    B = sorted(d for d in os.listdir(os.path.join(HERE, "results_small")) if d[0].isdigit())

    say("1. states with >= 5% of designs per benchmark (greedy + BO + GA + random-18k)")
    groups, tot = {}, 0
    for b in B:
        labs = []
        for root, pat in (("results_small", "part_*.npz"), ("results_bo", "seed_*.npz"), ("results_ga", "seed_*.npz"), ("results_rw18k", "part_*.npz")):
            d = cat(root, b, pat, ("cpi", "mscs")); labs.append(label(feat(d["cpi"], d["mscs"])))
        lab = np.concatenate(labs); tot += len(lab); f = np.bincount(lab, minlength=K) / len(lab); n = int((f >= 0.05).sum())
        groups.setdefault(n, []).append(b.split(".")[1])
        say(f"  {b.split('.')[1]:12s} n={len(lab):6d} " + " ".join(f"S{s}:{100*f[s]:5.1f}" for s in range(K)) + f"  -> {n}")
    say(f"  total designs {tot}")
    for n in sorted(groups):
        say(f"  {n} states: {len(groups[n])} benchmarks: {', '.join(groups[n])}")

    say("\n2. static balance at A_ref (+-5% area): Dispatch Jain, best 2% CPI minus all (median over benchmarks)")
    for root, name in (("results_rw", "100k"), ("results_rw18k", "18k")):
        d_, ns = [], []
        for b in B:
            A = json.load(open(os.path.join(HERE, "results_ga", b, "meta.json")))["A_ref"]
            d = cat(root, b, "part_*.npz"); m = np.abs(d["area"] / A - 1) <= 0.05
            if m.sum() < 50:
                continue
            c = d["cpi"][m]; top = c <= np.quantile(c, 0.02); J = jain(d["cpi"][m], d["mscs"][m])
            d_.append(np.median(J[top]) - np.median(J)); ns.append(int(m.sum()))
        d_ = np.array(d_)
        say(f"  {name:5s}: median {np.median(d_):+.3f}; best more balanced in {np.sum(d_ > 0)}/{len(d_)} benchmarks; "
            f"designs at A_ref per benchmark: median {int(np.median(ns))}, min {min(ns)}")

    say("\n3. cactuBSSN case")
    b = "607.cactuBSSN_s"
    for root, name in (("results_rw", "100k"), ("results_rw18k", "18k")):
        d = cat(root, b, "part_*.npz"); cpi = d["cpi"]; fl = cpi.min(); m = cpi <= 1.05 * fl
        lab = label(feat(cpi[m], d["mscs"][m])); rob = d["cfg"][m, ROB]
        say(f"  {name:5s}: floor {fl:.3f}; within 5%: {m.sum()} designs, ROB {int(rob.min())}-{int(rob.max())}, "
            f"Commit BP share of CPI median {100*np.median(d['mscs'][m, 14]):.0f}%, states "
            + ", ".join(f"S{s} {c}" for s, c in enumerate(np.bincount(lab, minlength=K)) if c))
        big = d["cfg"][:, L1] == d["cfg"][:, L1].max(); r = d["cfg"][big, ROB]; q1, q3 = np.quantile(r, [0.25, 0.75])
        lo, hi = np.median(cpi[big][r <= q1]), np.median(cpi[big][r >= q3])
        say(f"         largest L1 ({int(d['cfg'][:, L1].max())}): {big.sum()} designs; ROB top vs bottom quartile median CPI "
            f"{hi:.3f} vs {lo:.3f} ({100*(1-hi/lo):.1f}% gain)")
    open(OUT, "w").write("\n".join(L) + "\n"); print("wrote", OUT)


if __name__ == "__main__":
    main()
