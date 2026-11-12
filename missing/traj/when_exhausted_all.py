#!/usr/bin/env python3
"""When (iii) over all 17 benchmarks, model values only.
Search = one greedy walk of results_small/<bench> (1000 per benchmark); d = its last design.
Floor F = lowest predicted CPI of the 18k random designs (results_rw18k/<bench>); N = random designs with CPI <= 1.05 F.
Residual of a walk = largest of the 12 stall components (FP/BP/SP x Decode/Dispatch/Issue/Commit, Base excluded) at d.
Exhausted (When) = CPI(d) <= 1.05 F  AND  the residual is the largest component in all designs of N.
Also reported: the looser 'in >= 90% of N', and the walks whose residual is the largest component at every step.
Read-only. Writes feature/when_exhausted_all.txt (never overwrites)."""
import os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "feature", "when_exhausted_all.txt")
COMP = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
NAME = [f"{s}-{c}" for s in ("Dec", "Disp", "Iss", "Com") for c in ("FP", "BP", "SP")]
L = []


def say(s=""):
    print(s, flush=True); L.append(s)


def load(root, b, keys):
    Z = [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, root, b, "part_*.npz")))]
    return {k: np.concatenate([z[k] for z in Z]) for k in keys}


def main():
    if os.path.exists(OUT):
        sys.exit(f"exists, refusing to overwrite: {OUT}")
    benches = sorted(d for d in os.listdir(os.path.join(HERE, "results_rw18k")) if os.path.isdir(os.path.join(HERE, "results_rw18k", d)))
    say("bench | floor | N | dominant in N (share of N) | walks: end<=1.05F | +all-N | +>=90%-N | +all-N & residual largest every step")
    tot = np.zeros(4, int); nw = 0; nb_ex = 0
    for b in benches:
        R = load("results_rw18k", b, ("cpi", "mscs")); W = load("results_small", b, ("start", "step", "cpi", "mscs"))
        F = R["cpi"].min(); near = R["cpi"] <= 1.05 * F
        domN = np.argmax(R["mscs"][near][:, COMP], 1); cnt = np.bincount(domN, minlength=12)
        top = int(cnt.argmax())
        n = [0, 0, 0, 0]; starts = np.unique(W["start"])
        for s in starts:
            k = np.where(W["start"] == s)[0]; k = k[np.argsort(W["step"][k])]
            dm = W["mscs"][k[-1]]; r = int(np.argmax(dm[COMP]))
            if W["cpi"][k[-1]] > 1.05 * F:
                continue
            n[0] += 1
            share = cnt[r] / near.sum()
            if share == 1.0:
                n[1] += 1
                if np.all(np.argmax(W["mscs"][k][:, COMP], 1) == r):
                    n[3] += 1
            if share >= 0.9:
                n[2] += 1
        tot += n; nw += len(starts); nb_ex += n[1] > 0
        say(f"{b:16s} | {F:.3f} | {near.sum():4d} | {NAME[top]} ({cnt[top] / near.sum():.0%}) | "
            f"{n[0]:4d} | {n[1]:4d} | {n[2]:4d} | {n[3]:4d}  (of {len(starts)})")
    say(f"ALL: walks {nw}; end<=1.05F {tot[0]}; exhausted (all N) {tot[1]} ({tot[1] / nw:.1%}); >=90% N {tot[2]} "
        f"({tot[2] / nw:.1%}); all N & residual largest every step {tot[3]}; benchmarks with >=1 exhausted walk {nb_ex}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w").write("\n".join(L) + "\n"); print("wrote", OUT)


if __name__ == "__main__":
    main()
