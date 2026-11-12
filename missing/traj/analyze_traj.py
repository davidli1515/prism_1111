#!/usr/bin/env python3
"""First-pass analysis of the greedy trajectories (reads results/, writes results/analysis_v1.txt).
Stacks are compared in cycles per instruction (CPI x share). Commit-stage dominance is structural (BP by
position-based accounting), so dominance/migration/balance are measured at the Dispatch stage."""
import os, sys, glob, json
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
CAT = ["FP", "BP", "SP"]
OUT = []


def say(s=""):
    print(s); OUT.append(s)


def load(tag):
    z = [np.load(f) for f in sorted(glob.glob(os.path.join(R, tag, "part_*.npz")))]
    return {k: np.concatenate([x[k] for x in z]) for k in z[0].files}


def ends(start):
    first = np.r_[0, np.where(np.diff(start) != 0)[0] + 1]
    last = np.r_[first[1:] - 1, len(start) - 1]
    return first, last


def cyc(cpi, m, stage):                      # (n,3) FP,BP,SP cycles at a stage
    return np.stack([cpi * m[:, 4 * stage + k] for k in (1, 2, 3)], 1)


def entropy(x):
    p = x / np.maximum(x.sum(1, keepdims=True), 1e-12)
    return -(p * np.log(p + 1e-12)).sum(1) / np.log(3)


def main():
    meta = json.load(open(os.path.join(R, "602.gcc_s", "meta.json")))
    P, MV = meta["params"], meta["moves"]
    B = sorted(d for d in os.listdir(R) if d[0].isdigit())
    adv, rec, endv, startdom, enddom = {}, {}, {}, {}, {}
    say(f"{'bench':16s} walks st/walk adv/walk  CPI end/start  area start->end | Dispatch dominant FP/BP/SP start -> end  migr | entropy@Disp start->end (up%)")
    for b in B:
        d = load(b); f, l = ends(d["start"])
        cd = cyc(d["cpi"], d["mscs"], 1)
        ds, de = cd[f].argmax(1), cd[l].argmax(1)
        e0, e1 = entropy(cd[f]), entropy(cd[l])
        k, a = d["kind"], d["act"]
        adv[b] = np.bincount(a[k == 1], minlength=34)[0::2]; rec[b] = np.bincount(a[k == 2], minlength=34)[1::2]
        endv[b] = (d["cpi"][l, None] * d["mscs"][l]).mean(0); startdom[b], enddom[b] = ds, de
        fmt = lambda x: "/".join(f"{np.mean(x == j):.2f}" for j in range(3))
        say(f"{b:16s} {len(f):5d} {len(k)/len(f):6.1f} {adv[b].sum()/len(f):7.1f}  {np.median(d['cpi'][l]/d['cpi'][f]):10.3f}"
            f"   {np.median(d['area'][f]):5.1f}->{np.median(d['area'][l]):5.1f} | {fmt(ds)} -> {fmt(de)}  {np.mean(ds!=de):.2f}"
            f" | {e0.mean():.2f}->{e1.mean():.2f} ({100*np.mean(e1>e0):.0f})")

    say("\n== routes: advance-move histograms (share of advance steps), top-4")
    for b in B:
        v = adv[b] / max(adv[b].sum(), 1); top = np.argsort(-v)[:4]
        say(f"  {b:16s} " + "  ".join(f"{P[i]}:{v[i]:.2f}" for i in top))
    V = np.array([adv[b] / max(adv[b].sum(), 1) for b in B])
    Rv = np.array([rec[b] / max(rec[b].sum(), 1) for b in B])
    E = np.array([endv[b] / endv[b].sum() for b in B])
    truth = [int(b in FP_SUITE) for b in B]
    for name, X, met in [("advance route", V, "cosine"), ("reclaim route", Rv, "cosine"),
                         ("endpoint MSCS", E, "euclidean")]:
        Z = linkage(X, "average", metric=met)
        c2 = fcluster(Z, 2, "maxclust"); c4 = fcluster(Z, 4, "maxclust")
        say(f"\n== clusters by {name}: ARI(k=2 vs SPECint/fp) = {adjusted_rand_score(truth, c2):.2f}")
        for c in sorted(set(c4)):
            say(f"   k=4 cluster {c}: " + ", ".join(b.split('.')[1] + ("*" if b in FP_SUITE else "") for b, x in zip(B, c4) if x == c))
    say("   (* = SPECfp)")

    # ---------------- joint run
    j = load("joint_17"); W = json.load(open(os.path.join(R, "joint_17", "meta.json")))["workloads"]
    f, l = ends(j["start"])
    say(f"\n== joint_17: {len(f)} walks, {len(j['kind'])/len(f):.1f} states/walk, adv/walk {np.sum(j['kind']==1)/len(f):.1f}, "
        f"geomean CPI end/start {np.median(j['cpi'][l]/j['cpi'][f]):.3f}, area {np.median(j['area'][f]):.1f}->{np.median(j['area'][l]):.1f}")
    v = np.bincount(j["act"][j["kind"] == 1], minlength=34)[0::2]; v = v / v.sum(); top = np.argsort(-v)[:6]
    say("   joint advance moves: " + "  ".join(f"{P[i]}:{v[i]:.2f}" for i in top))
    # per-workload relative gain on every advance step
    idx = np.where(j["kind"] == 1)[0]
    g = np.array([(j["cpi_wl"][i] - j["k_cpi_wl"][i, j["act"][i]]) / j["cpi_wl"][i] for i in idx])   # (steps, W)
    say(f"   advance steps: {len(idx)}; per-workload gain on a joint advance step: median {np.median(g)*100:.2f}%, "
        f"share of (step,workload) with a loss {np.mean(g < 0):.2f}")
    C = np.corrcoef(g.T)
    Zc = linkage(1 - C[np.triu_indices(len(W), 1)], "average")
    cc = fcluster(Zc, 4, "maxclust")
    say("   co-movement clusters (correlation of per-step gains):")
    for c in sorted(set(cc)):
        say("     " + ", ".join(w.split('.')[1] + ("*" if w in FP_SUITE else "") for w, x in zip(W, cc) if x == c))
    mean_g = g.mean(0); loss = (g < 0).mean(0)
    order = np.argsort(mean_g)
    say("   workloads by mean gain per joint advance step (least helped first): " +
        ", ".join(f"{W[i].split('.')[1]} {mean_g[i]*100:.2f}% (loss {loss[i]:.2f})" for i in order[:6]))
    # who is left behind at the end: joint-end CPI vs the median single-benchmark end CPI
    single_end = {}
    for b in B:
        d = load(b); fb, lb = ends(d["start"]); single_end[b] = np.median(d["cpi"][lb])
    ratio = np.median(j["cpi_wl"][l], 0) / np.array([single_end[w] for w in W])
    o = np.argsort(-ratio)
    say("   joint-end CPI / single-benchmark-end CPI (highest = most compromised): " +
        ", ".join(f"{W[i].split('.')[1]} {ratio[i]:.2f}" for i in o[:6]))
    say("   ... least compromised: " + ", ".join(f"{W[i].split('.')[1]} {ratio[i]:.2f}" for i in o[-4:]))

    # ---------------- by-product: masking events in single-benchmark walks
    say("\n== by-product: masked parameters (gain < 0.1% at t, >= 0.5% at t+1 after an advance of another parameter)")
    tot, ev, pairs = 0, 0, {}
    for b in B:
        d = load(b); k, a, st = d["kind"], d["act"], d["start"]
        gup = (d["cpi"][:, None] - d["k_cpi"][:, 0::2]) / d["cpi"][:, None]          # (n,17) up-move gains
        for t in np.where(k == 1)[0]:
            if t + 1 >= len(k) or st[t + 1] != st[t]:
                continue
            moved = a[t] // 2
            m = (gup[t] < 0.001) & (gup[t + 1] >= 0.005); m[moved] = False
            tot += 1; ev += m.any()
            for p in np.where(m)[0]:
                pairs[(P[moved], P[p])] = pairs.get((P[moved], P[p]), 0) + 1
    say(f"   advance steps followed by >=1 unmasked parameter: {ev}/{tot} = {ev/max(tot,1):.2f}")
    say("   most frequent (moved -> unmasked): " + ", ".join(f"{a_}->{b_} {n}" for (a_, b_), n in
                                                            sorted(pairs.items(), key=lambda x: -x[1])[:8]))
    out = os.path.join(R, "analysis_v1.txt")
    if os.path.exists(out):
        out = out.replace(".txt", f".{os.getpid()}.txt")
    open(out, "w").write("\n".join(OUT) + "\n"); print("->", out)


if __name__ == "__main__":
    main()
