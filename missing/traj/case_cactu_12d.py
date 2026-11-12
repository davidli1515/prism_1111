#!/usr/bin/env python3
"""cactuBSSN case study with the four-stage states (results_small/state_clusters_12d.npz).
  1. state of every gem5 point of the walk (gem5_val4/runs/607.cactuBSSN_s/p840_s*), of the end design with 16 MSHRs
     (feature/gem5/runs/.../p840_s23_mshr16), and of the model walk (expert/607.cactuBSSN_s_s840_at10_port2.json)
  2. random designs within 5% of the CPI floor (18k and 100k): states
  3. lowest CPI of a random design in the state reached with 16 MSHRs (18k and 100k)
  4. Decode / Dispatch stall cycles per instruction (FP, BP, SP) and within-stage FP share of the end design,
     gem5 with 4 and 16 MSHRs, and the model at the same design
Read-only. Writes feature/case_cactu_12d.txt (never overwrites)."""
import os, sys, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cluster_states_12d as Q
B = "607.cactuBSSN_s"; OUT = os.path.join(HERE, "feature", "case_cactu_12d.txt")
CENT = np.load(Q.OUT + ".npz")["centroids"]; L = []


def say(s=""):
    print(s, flush=True); L.append(s)


def state(cpi, m):
    m = np.atleast_2d(np.asarray(m, float)); f = Q.feat12(np.atleast_1d(cpi).astype(float), m)
    return np.argmin(((f[:, None] - CENT[None]) ** 2).sum(2), 1)


def main():
    if os.path.exists(OUT):
        sys.exit(f"exists, refusing to overwrite: {OUT}")
    G = {}
    for f in glob.glob(os.path.join(HERE, "gem5_val4", "runs", B, "p840_s*", "result.json")):
        G[int(os.path.basename(os.path.dirname(f)).split("_s")[1])] = json.load(open(f))
    F = json.load(open(os.path.join(HERE, "feature", "gem5", "runs", B, "p840_s23_mshr16", "result.json")))
    W = json.load(open(os.path.join(HERE, "expert", "607.cactuBSSN_s_s840_at10_port2.json")))["original"]
    say("1. states (four-stage, k = 5)")
    say("   gem5 walk: " + " ".join(f"{k}:S{state(G[k]['cpi'], G[k]['mscs16'])[0]}({G[k]['cpi']:.2f})" for k in sorted(G)))
    say(f"   gem5 end + 16 MSHRs: S{state(F['cpi'], F['mscs16'])[0]} (CPI {F['cpi']:.3f})")
    say("   model walk: " + " ".join(f"{w['step']}:S{state(w['cpi'], w['mscs'])[0]}" for w in W))
    s16 = state(F["cpi"], F["mscs16"])[0]

    for root, name in (("results_rw18k", "18k"), ("results_rw", "100k")):
        Z = [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, root, B, "part_*.npz")))]
        cpi = np.concatenate([z["cpi"] for z in Z]); m = np.concatenate([z["mscs"] for z in Z]); lab = state(cpi, m)
        near = cpi <= 1.05 * cpi.min()
        say(f"2/3. random {name}: floor {cpi.min():.3f}; within 5%: {near.sum()} designs, states "
            + ", ".join(f"S{s} {c}" for s, c in enumerate(np.bincount(lab[near], minlength=len(CENT))) if c)
            + f"; lowest CPI in S{s16}: {cpi[lab == s16].min():.3f} ({np.sum(lab == s16)} designs in S{s16})")

    say("4. end design (step 23): stall cycles per instruction [FP, BP, SP] and FP share within the stage")
    for name, cpi, m in (("gem5, 4 MSHRs", G[23]["cpi"], G[23]["mscs16"]), ("gem5, 16 MSHRs", F["cpi"], F["mscs16"]),
                         ("model, 4 MSHRs", W[23]["cpi"], W[23]["mscs"])):
        m = np.asarray(m, float)
        parts = []
        for s, st in ((0, "Dec"), (1, "Disp"), (2, "Iss"), (3, "Com")):
            c = cpi * m[4 * s + 1:4 * s + 4]
            parts.append(f"{st} [{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}] FP {100*c[0]/max(c.sum(), 1e-12):.0f}%")
        say(f"   {name:15s} CPI {cpi:.3f} | " + " | ".join(parts))
    open(OUT, "w").write("\n".join(L) + "\n"); print("wrote", OUT)


if __name__ == "__main__":
    main()
