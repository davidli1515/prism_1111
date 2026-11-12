#!/usr/bin/env python3
"""Per-benchmark bottleneck states over ALL four search methods, and strict state transitions.
Every design (greedy small-start, BO, GA, random search) is assigned to the nearest of the six small-start reference
centroids (results_small/state_clusters_v1.npz), as in Section V-B. Reported per benchmark:
  1. states visited (any design) per method and pooled; share of designs per state (pooled, each method weighted equally)
  2. strict transitions along searches:
       greedy: share of the 1,000 walks whose state changes at least once (consecutive steps)
       BO / GA / random: incumbent trajectory (best CPI so far with area <= A_ref, A_ref from results_ga/<b>/meta.json;
       random search = 10 blocks of 512 in index order); share of runs whose incumbent state changes at least once
  3. variance of the six-component feature explained by the benchmark (between-benchmark / total; methods pooled)
Read-only. Writes results_opt_clusters/state_transitions_all.txt (never overwrites)."""
import os, sys, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_opt_clusters", "state_transitions_all.txt")
CENT = np.load(os.path.join(HERE, "results_small", "state_clusters_v1.npz"))["centroids"]; K = len(CENT)


def feat(cpi, m):
    X = np.stack([cpi * m[:, 4 * s + k] for s in (1, 3) for k in (1, 2, 3)], 1)
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def label(X):
    out = np.empty(len(X), int)
    for i in range(0, len(X), 200000):
        out[i:i + 200000] = np.argmin(((X[i:i + 200000, None] - CENT[None]) ** 2).sum(2), 1)
    return out


def changes(seq):
    return bool(np.any(seq[1:] != seq[:-1])) if len(seq) > 1 else False


def incumbent_states(cpi, area, lab, aref):
    best, out = np.inf, []
    for c, a, l in zip(cpi, area, lab):
        if a <= aref and c < best:
            best = c; out.append(l)
    return np.array(out)


def main():
    if os.path.exists(OUT):
        sys.exit(f"exists, refusing to overwrite: {OUT}")
    benches = sorted(d for d in os.listdir(os.path.join(HERE, "results_small")) if d[0].isdigit())
    L = [f"{'bench':12s}| states visited  G  BO GA RS all | pooled share S0..S5 (%)            "
         f"| runs with a state change: greedy  BO    GA    RS"]
    feats, owner = [], []
    for b in benches:
        aref = json.load(open(os.path.join(HERE, "results_ga", b, "meta.json")))["A_ref"]
        visited, shares, trans = {}, [], {}
        # greedy
        fs = sorted(glob.glob(os.path.join(HERE, "results_small", b, "part_*.npz")))
        Z = [np.load(f) for f in fs]
        X = np.concatenate([feat(z["cpi"], z["mscs"]) for z in Z]); lab = label(X)
        start = np.concatenate([z["start"] + 100000 * i for i, z in enumerate(Z)]); step = np.concatenate([z["step"] for z in Z])
        o = np.lexsort((step, start)); s_, l_ = start[o], lab[o]
        cut = np.flatnonzero(np.diff(s_)) + 1
        walks = np.split(l_, cut)
        trans["G"] = np.mean([changes(w) for w in walks]); visited["G"] = set(lab); shares.append(np.bincount(lab, minlength=K) / len(lab))
        feats.append(X); owner += [b] * len(X)
        # BO, GA, RS
        for name, root, pat in (("BO", "results_bo", "seed_*.npz"), ("GA", "results_ga", "seed_*.npz"), ("RS", "results_rw", "part_*.npz")):
            fs = sorted(glob.glob(os.path.join(HERE, root, b, pat)))
            Z = [np.load(f) for f in fs]
            if name == "RS":
                cat = {k: np.concatenate([z[k] for z in Z]) for k in ("idx", "cpi", "area", "mscs")}
                o = np.argsort(cat["idx"], kind="stable"); cat = {k: v[o] for k, v in cat.items()}
                Xm = feat(cat["cpi"], cat["mscs"]); lm = label(Xm)
                runs = [(cat["cpi"][i:i + 512], cat["area"][i:i + 512], lm[i:i + 512]) for i in range(0, 10 * 512, 512)]
            else:
                Xs = [feat(z["cpi"], z["mscs"]) for z in Z]; ls = [label(x) for x in Xs]
                Xm, lm = np.concatenate(Xs), np.concatenate(ls)
                runs = [(z["cpi"], z["area"], l) for z, l in zip(Z, ls)]
            trans[name] = np.mean([changes(incumbent_states(c, a, l, aref)) for c, a, l in runs])
            visited[name] = set(lm); shares.append(np.bincount(lm, minlength=K) / len(lm))
            feats.append(Xm); owner += [b] * len(Xm)
        allv = set().union(*visited.values()); sh = 100 * np.mean(shares, 0)
        L.append(f"{b.split('.')[1]:12s}| {len(visited['G']):>16d} {len(visited['BO']):3d}{len(visited['GA']):3d}{len(visited['RS']):3d}{len(allv):4d} | "
                 + " ".join(f"{x:5.1f}" for x in sh)
                 + f" | {100*trans['G']:22.0f}%{100*trans['BO']:5.0f}%{100*trans['GA']:5.0f}%{100*trans['RS']:5.0f}%")
        print(L[-1], flush=True)
    F = np.concatenate(feats); own = np.array(owner)
    tot = ((F - F.mean(0)) ** 2).sum()
    within = sum(((F[own == b] - F[own == b].mean(0)) ** 2).sum() for b in benches)
    L.append(f"variance of the six-component feature explained by the benchmark: {100*(1-within/tot):.0f}% "
             f"(all designs of all methods, n={len(F)})")
    print(L[-1]); os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
