#!/usr/bin/env python3
"""When (iii) over all 17 benchmarks with an area-matched floor; model values only.
Search = one greedy walk of results_small/<bench> (1000 per benchmark), d = its last design (area a_d).
Area-matched floor F_d = lowest predicted CPI among the 18k random designs (results_rw18k) with area <= a_d;
N_d = those designs with CPI <= 1.05 F_d.  States: nearest 12-dim centroid (results_small/state_clusters_12d.npz).
Outcome of a walk:
  far        CPI(d) > 1.05 F_d                                   (CPI alone says: not finished)
  exhausted  CPI(d) <= 1.05 F_d, one state holds >= TH of N_d and d is in it
  open       CPI(d) <= 1.05 F_d otherwise (near-floor designs spread over states, or d outside the main one)
  thin       |N_d| < NMIN (too few comparable designs; reported, not classified)
TH in {0.8, 0.9}; NMIN = 5.  The global-floor version (no area matching) is printed for comparison.
Read-only. Writes feature/when_exhausted_area.txt (never overwrites)."""
import os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cluster_states_12d as Q
OUT = os.path.join(HERE, "feature", "when_exhausted_area.txt")
C = np.load(Q.OUT + ".npz")["centroids"]; NMIN = 5; L = []


def say(s=""):
    print(s, flush=True); L.append(s)


def lab(c, m):
    f = Q.feat12(np.atleast_1d(c).astype(float), np.atleast_2d(m).astype(float))
    return np.argmin(((f[:, None] - C[None]) ** 2).sum(2), 1)


def load(root, b, keys):
    Z = [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, root, b, "part_*.npz")))]
    return {k: np.concatenate([z[k] for z in Z]) for k in keys}


def classify(cd, sd, ad, Rc, Ra, Rl, th, area_match):
    sel = Ra <= ad if area_match else np.ones_like(Ra, bool)
    if sel.sum() == 0:
        return "thin"
    F = Rc[sel].min(); n = sel & (Rc <= 1.05 * F)
    if n.sum() < NMIN:
        return "thin"
    if cd > 1.05 * F:
        return "far"
    cnt = np.bincount(Rl[n], minlength=len(C)); t = cnt.argmax()
    return "exhausted" if cnt[t] / n.sum() >= th and sd == t else "open"


def main():
    if os.path.exists(OUT):
        sys.exit(f"exists, refusing to overwrite: {OUT}")
    benches = sorted(d for d in os.listdir(os.path.join(HERE, "results_rw18k")) if os.path.isdir(os.path.join(HERE, "results_rw18k", d)))
    KEYS = ("far", "exhausted", "open", "thin")
    for area_match in (True, False):
        for th in (0.9, 0.8):
            say(f"=== {'area-matched' if area_match else 'global'} floor, concentration >= {th:.0%}")
            say("bench            |  far exhausted  open  thin")
            tot = dict.fromkeys(KEYS, 0); bex = bop = 0
            for b in benches:
                R = load("results_rw18k", b, ("cpi", "mscs", "area")); Rl = lab(R["cpi"], R["mscs"])
                W = load("results_small", b, ("start", "step", "cpi", "mscs", "area"))
                ends = []
                for s in np.unique(W["start"]):
                    k = np.where(W["start"] == s)[0]; ends.append(k[np.argmax(W["step"][k])])
                ends = np.array(ends); El = lab(W["cpi"][ends], W["mscs"][ends])
                c = dict.fromkeys(KEYS, 0)
                for i, e in enumerate(ends):
                    c[classify(W["cpi"][e], El[i], W["area"][e], R["cpi"], R["area"], Rl, th, area_match)] += 1
                for k in KEYS:
                    tot[k] += c[k]
                bex += c["exhausted"] > 0; bop += c["open"] > 0
                say(f"{b:16s} | {c['far']:4d} {c['exhausted']:9d} {c['open']:5d} {c['thin']:5d}")
            near = tot["exhausted"] + tot["open"]
            say(f"ALL              | {tot['far']:4d} {tot['exhausted']:9d} {tot['open']:5d} {tot['thin']:5d}  | near floor {near}: "
                f"exhausted {tot['exhausted']} ({tot['exhausted'] / max(near, 1):.0%}), open {tot['open']}; "
                f"benchmarks with exhausted {bex}, with open {bop}")
            say()
    os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w").write("\n".join(L) + "\n"); print("wrote", OUT)


if __name__ == "__main__":
    main()
