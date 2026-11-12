#!/usr/bin/env python3
"""Negative control for When (iii), model only (PRISM surrogate, CPU).
Groups (global floor F = lowest predicted CPI of the 18k random designs; N = random designs with CPI <= 1.05 F;
states = nearest 12-dim centroid; same rule as when_exhausted_area.py, global floor, concentration >= 90%):
  exhausted  greedy end d with CPI(d) <= 1.05 F, one state holds >= 90% of N and d is in it
  open       CPI(d) <= 1.05 F otherwise
  far        CPI(d) >  1.05 F   (reference; 30 random walks per benchmark)
Re-search from every d: M random restarts, each applies 1..KMAX random legal moves (traj_common.move) to d.
Per walk: gain = 1 - min CPI found / CPI(d); below = min CPI found < 0.95 F (a design clearly below the floor);
moved = among found designs with CPI <= CPI(d), share in a state other than state(d); any_moved = at least one.
Read-only on existing results. Writes feature/negctl_when.{txt,npz} (never overwrites)."""
import os, sys, glob, random
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import traj_common as T
import cluster_states_12d as Q
OUT = os.path.join(HERE, "feature", "negctl_when")
C = np.load(Q.OUT + ".npz")["centroids"]; M, KMAX, NFAR, TH = 400, 8, 30, 0.9


def lab(c, m):
    f = Q.feat12(np.atleast_1d(c).astype(float), np.atleast_2d(m).astype(float))
    return np.argmin(((f[:, None] - C[None]) ** 2).sum(2), 1)


def load(root, b, keys):
    Z = [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, root, b, "part_*.npz")))]
    return {k: np.concatenate([z[k] for z in Z]) for k in keys}


def run(b):
    R = load("results_rw18k", b, ("cpi", "mscs")); Rl = lab(R["cpi"], R["mscs"])
    F = R["cpi"].min(); near = R["cpi"] <= 1.05 * F
    cnt = np.bincount(Rl[near], minlength=len(C)); top = cnt.argmax(); conc = cnt[top] / near.sum()
    W = load("results_small", b, ("start", "step", "cpi", "mscs", "cfg"))
    ends = np.array([np.where(W["start"] == s)[0][np.argmax(W["step"][W["start"] == s])] for s in np.unique(W["start"])])
    El = lab(W["cpi"][ends], W["mscs"][ends]); grp = []
    for i, e in enumerate(ends):
        if W["cpi"][e] > 1.05 * F:
            grp.append("far")
        else:
            grp.append("exhausted" if conc >= TH and El[i] == top else "open")
    grp = np.array(grp); rng = np.random.RandomState(0)
    far = np.where(grp == "far")[0]; keep = np.concatenate([np.where(grp != "far")[0], rng.choice(far, min(NFAR, len(far)), replace=False)])
    P = T.WLPredictor(b); res = []
    for i in sorted(keep):
        e = ends[i]; c0 = T.S.project17({p: int(v) for p, v in zip(T.SEARCH, W["cfg"][e])})
        cd, md = P(c0); sd = lab(cd, md)[0]; r = random.Random(int(e)); cs, ms = [], []
        for _ in range(M):
            c = dict(c0)
            for _ in range(r.randint(1, KMAX)):
                p, d = r.choice(T.MOVES); cn, _ = T.move(c, p, d)
                if cn is not None:
                    c = cn
            y, m = P(c); cs.append(y); ms.append(m)
        cs, ms = np.array(cs), np.array(ms); L = lab(cs, ms); ok = cs <= cd
        res.append((b, grp[i], cd, cs.min(), 1 - cs.min() / cd, cs.min() < 0.95 * F,
                    (L[ok] != sd).mean() if ok.any() else 0.0, bool((L[ok] != sd).any()), ok.sum()))
    print(b, "done", len(res), flush=True)
    return res


def main():
    for ext in (".txt", ".npz"):
        if os.path.exists(OUT + ext):
            sys.exit(f"exists, refusing to overwrite: {OUT + ext}")
    benches = sorted(d for d in os.listdir(os.path.join(HERE, "results_rw18k")) if os.path.isdir(os.path.join(HERE, "results_rw18k", d)))
    with Pool(len(benches)) as pool:
        rows = [r for rs in pool.map(run, benches) for r in rs]
    b = np.array([r[0] for r in rows]); g = np.array([r[1] for r in rows]); v = np.array([r[2:] for r in rows], float)
    L = [f"negative control for When: M={M} random restarts (1..{KMAX} moves) from each greedy end; model values",
         "group | walks | gain median (p90) | below 0.95F | moved share median | any moved"]

    def row(name, k):
        if k.sum() == 0:
            return f"{name:22s} |     0"
        return (f"{name:22s} | {k.sum():5d} | {np.median(v[k, 2]):.1%} ({np.percentile(v[k, 2], 90):.1%}) | "
                f"{v[k, 3].mean():.1%} | {np.median(v[k, 4]):.1%} | {v[k, 5].mean():.1%}")
    for name in ("exhausted", "open", "far"):
        L.append(row(name, g == name))
    for bb in sorted(set(b[g == "exhausted"]) & set(b[g == "open"])):
        L += [row(f"{bb} {name}", (b == bb) & (g == name)) for name in ("exhausted", "open")]
    print("\n".join(L)); os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT + ".txt", "w").write("\n".join(L) + "\n")
    np.savez_compressed(OUT + ".npz", bench=b, group=g, vals=v,
                        cols=np.array(["cpi_d", "cpi_min", "gain", "below", "moved_share", "any_moved", "n_ok"]))
    print("wrote", OUT + ".txt")


if __name__ == "__main__":
    main()
