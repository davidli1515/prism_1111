#!/usr/bin/env python3
"""BO-as-optimiser vs greedy trajectories (writes results_bo/analysis_bo_v1.txt, never overwrites).
States are labelled with the reference centroids (results/state_clusters_v1.npz); balance = Jain index of
Dispatch FP/BP/SP stall cycles.
 1. per benchmark: BO final Pareto fronts (each of 10 seeds) -> state distribution and Jain on the front;
    the front point closest in area to the greedy (small-start) median end area -> state / Jain, compared with
    the greedy end states (small starts and random starts)
 2. quality: share of greedy end designs dominated by the pooled BO fronts; BO best CPI vs greedy best CPI
 3. BO over iterations (pooled seeds, all benchmarks): state shares and Jain of sampled points by phase."""
import os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RB, RS, RR = (os.path.join(HERE, x) for x in ("results_bo", "results_small", "results"))
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
OUT = []


def say(s=""):
    print(s); OUT.append(s)


def feats(cpi, m):
    X = np.stack([cpi * m[:, 4 * s + k] for s in (1, 3) for k in (1, 2, 3)], 1)
    Xn = X / np.maximum(X.sum(1, keepdims=True), 1e-12)
    x = X[:, :3]; J = x.sum(1) ** 2 / (3 * (x ** 2).sum(1))
    return Xn, J


def label(Xn, C):
    return np.argmin(((Xn[:, None] - C[None]) ** 2).sum(-1), 1)


def pareto(cpi, area):
    o = np.lexsort((cpi, area)); best, keep = np.inf, []
    for i in o:
        if cpi[i] < best:
            keep.append(i); best = cpi[i]
    return np.array(keep)


def greedy_ends(root, b):
    fs = sorted(glob.glob(os.path.join(root, b, "part_*.npz")))
    d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "cpi", "mscs", "area")}
    st = d["start"]; f = np.r_[0, np.where(np.diff(st) != 0)[0] + 1]; l = np.r_[f[1:] - 1, len(st) - 1]
    return {k: d[k][l] for k in ("cpi", "mscs", "area")}


def short(b):
    return b.split(".")[1].replace("_s", "").replace("_r", "") + ("*" if b in FP_SUITE else "")


def main():
    out = os.path.join(RB, "analysis_bo_v1.txt")
    if os.path.exists(out):
        out = out.replace(".txt", f".{os.getpid()}.txt")
    C = np.load(os.path.join(RR, "state_clusters_v1.npz"))["centroids"]; k = len(C)
    B = sorted(d for d in os.listdir(RB) if d[0].isdigit())
    say(f"{'bench':11s} | BO front: pts/seed  state mode (share)  Jain | BO front @greedy area: state Jain | "
        f"greedy end small: mode Jain | random: mode | greedy ends dominated by BO | best CPI BO/greedy")
    agree_mode, agree_area, ph = 0, 0, {}
    for b in B:
        runs = [np.load(f) for f in sorted(glob.glob(os.path.join(RB, b, "seed_*.npz")))]
        F_lab, F_J, near_lab, near_J, npts, Pc, Pa = [], [], [], [], [], [], []
        gs, gr = greedy_ends(RS, b), greedy_ends(RR, b)
        a_ref = np.median(gs["area"])
        for z in runs:
            p = pareto(z["cpi"], z["area"]); Xn, J = feats(z["cpi"][p], z["mscs"][p]); lab = label(Xn, C)
            F_lab += lab.tolist(); F_J += J.tolist(); npts.append(len(p)); Pc += z["cpi"][p].tolist(); Pa += z["area"][p].tolist()
            i = np.argmin(abs(z["area"][p] - a_ref)); near_lab.append(lab[i]); near_J.append(J[i])
            Xa, Ja = feats(z["cpi"], z["mscs"]); la = label(Xa, C)
            for name, msk in (("init", z["phase"] == 0), ("bo 31-150", (z["it"] >= 30) & (z["it"] < 150)),
                              ("bo 150-300", (z["it"] >= 150) & (z["it"] < 300)), ("bo 300-500", z["it"] >= 300)):
                ph.setdefault(name, [[], []]); ph[name][0] += la[msk].tolist(); ph[name][1] += Ja[msk].tolist()
        F_lab = np.array(F_lab); fm = np.bincount(F_lab, minlength=k) / len(F_lab)
        Xs, Js = feats(gs["cpi"], gs["mscs"]); ls_ = label(Xs, C); sm = np.bincount(ls_, minlength=k) / len(ls_)
        Xr, _ = feats(gr["cpi"], gr["mscs"]); rm = np.bincount(label(Xr, C), minlength=k) / len(gr["cpi"])
        nm = np.bincount(near_lab, minlength=k) / len(near_lab)
        Pc, Pa = np.array(Pc), np.array(Pa); fr = pareto(Pc, Pa); Pc, Pa = Pc[fr], Pa[fr]
        dom = np.mean([np.any((Pc <= c) & (Pa <= a) & ((Pc < c) | (Pa < a))) for c, a in zip(gs["cpi"], gs["area"])])
        agree_mode += np.argmax(fm) == np.argmax(sm); agree_area += np.argmax(nm) == np.argmax(sm)
        say(f"{short(b):11s} | {np.mean(npts):5.1f}  S{np.argmax(fm)} ({100*fm.max():3.0f}%)  {np.mean(F_J):.2f} | "
            f"S{np.argmax(nm)} ({100*nm.max():3.0f}%) {np.mean(near_J):.2f} | S{np.argmax(sm)} ({100*sm.max():3.0f}%) {np.mean(Js):.2f} | "
            f"S{np.argmax(rm)} | {dom:.2f} | {np.min(Pc)/min(gs['cpi'].min(), gr['cpi'].min()):.3f}")
    say(f"\nBO front mode == greedy(small) end mode: {agree_mode}/{len(B)};  "
        f"BO front point at the greedy end area has the same state: {agree_area}/{len(B)}")
    say("\nBO sampled points over iterations (all benchmarks, 10 seeds): state shares S0..S5 | mean Dispatch Jain")
    for name in ("init", "bo 31-150", "bo 150-300", "bo 300-500"):
        lab, J = np.array(ph[name][0]), np.array(ph[name][1])
        say(f"  {name:11s} " + " ".join(f"{100*np.mean(lab==s):5.1f}" for s in range(k)) + f" | {J.mean():.3f}")
    open(out, "w").write("\n".join(OUT) + "\n"); print("->", out)


if __name__ == "__main__":
    main()
