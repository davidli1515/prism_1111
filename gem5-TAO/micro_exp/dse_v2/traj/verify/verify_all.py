#!/usr/bin/env python3
"""
Read-only verification of the trajectory study (writes only verify/report_v1.txt and verify/scratch/).
 L1 shared pieces: project17 invariants on 10k fresh samples and on every stored design; idempotence.
 L2 stored data: completeness / NaN; re-evaluation of random stored points (CPI, MSCS, area) with a fresh
    predictor + area model, every source; bit-exact re-run of small pieces (greedy walks, GA seed, RW designs).
 L3 algorithms: greedy replay (next state = chosen candidate; decision rule re-applied to the stored candidate
    values; candidate validity = move()); BO / GA / RW budget, phases, constraint flags, identical RW designs.
 L4 analysis: unit tests (Pareto vs brute force, Jain/entropy values and scale invariance, walk-end
    extraction), independent re-computation of the headline numbers with code written here, and a
    benchmark-shuffle null for the cross-optimiser state agreement.
"""
import os, sys, glob, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); sys.path.insert(0, TR)
import traj_common as T
from config_dse import get_param_values
sys.path.insert(0, os.path.join(os.path.dirname(TR), "dse17"))
from area17 import FastArea

OUT, FAILS = [], []
RNG = np.random.RandomState(12345)
SCR = os.path.join(HERE, "scratch")
B17 = list(T.CM.SPEC17)
FIXED = T.S.FIXED


def say(s=""):
    print(s, flush=True); OUT.append(s)


def check(name, ok, detail=""):
    say(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def cat(fs, keys=None):
    zs = [np.load(f) for f in fs]; keys = keys or zs[0].files
    return {k: np.concatenate([z[k] for z in zs]) for k in keys}


def cfg_of(row):
    c = {p: int(v) for p, v in zip(T.SEARCH, row)}; c.update(FIXED); return c


def invariants(c):
    bad = []
    if c["decodeWidth"] > c["fetchWidth"]: bad.append("decode>fetch")
    if c["dispatchWidth"] > c["decodeWidth"]: bad.append("dispatch>decode")
    if c["wbWidth"] < c["dispatchWidth"] or c["commitWidth"] < c["dispatchWidth"]: bad.append("wb/commit<dispatch")
    if c["numROBEntries"] < min(2 * c["dispatchWidth"] + c["numIQEntries"] + c["LQEntries"] + c["SQEntries"], 1024): bad.append("rob")
    mx = max(1, c["l1_size"] * 1024 // (16 * 64))
    if c["l1d_assoc"] > mx or c["l1i_assoc"] > mx: bad.append("l1assoc")
    for p in T.SEARCH:
        if c[p] not in get_param_values(p): bad.append(f"{p}∉values")
    for k, v in FIXED.items():
        if c[k] != v: bad.append(f"fixed {k}")
    return bad


def jain(x):
    return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def x3(cpi, m, s=1):
    return np.stack([cpi * m[:, 4 * s + k] for k in (1, 2, 3)], 1)


def ends_idx(st):
    """last row of each walk, written independently of the analysis scripts (groupby on start id)"""
    out = {}
    for i, s in enumerate(st):
        out[int(s)] = i
    return np.array([out[s] for s in sorted(out)]), np.array([min(np.where(st == s)[0]) for s in sorted(out)])


def pareto_bf(cpi, area):
    n = len(cpi); keep = []
    for i in range(n):
        dom = np.any((cpi <= cpi[i]) & (area <= area[i]) & ((cpi < cpi[i]) | (area < area[i])))
        if not dom: keep.append(i)
    return np.array(keep)


# ------------------------------------------------------------------ L1
def L1():
    say("\nL1. shared pieces")
    rng = np.random.RandomState(0); bad = 0
    for _ in range(10000):
        c = T.S.sample17(rng); bad += bool(invariants(c)) or T.ckey(T.S.project17(c)) != T.ckey(c)
    check("10k fresh sample17 designs satisfy all invariants and project17 is idempotent", bad == 0, f"{bad} bad")
    srcs = {"greedy random": "results/*/part_*.npz", "greedy small": "results_small/*/part_*.npz",
            "greedy seed2": "results_seed2/*/part_*.npz", "BO": "results_bo/*/seed_*.npz", "GA": "results_ga/*/seed_*.npz",
            "RW": "results_rw/*/part_*.npz"}
    for name, pat in srcs.items():
        fs = sorted(glob.glob(os.path.join(TR, pat)))
        rows = np.unique(np.concatenate([np.load(f)["cfg"] for f in fs]), axis=0)
        sub = rows[RNG.choice(len(rows), min(20000, len(rows)), replace=False)]
        nb = sum(bool(invariants(cfg_of(r))) or T.ckey(T.S.project17(cfg_of(r))) != T.ckey(cfg_of(r)) for r in sub)
        check(f"stored designs valid + projection-stable: {name}", nb == 0, f"{nb}/{len(sub)} of {len(rows)} unique")


# ------------------------------------------------------------------ L2
def L2():
    say("\nL2. stored data")
    exp = {"results": 1000, "results_small": 1000, "results_seed2": 100}
    for root, n in exp.items():
        for b in B17:
            fs = sorted(glob.glob(os.path.join(TR, root, b, "part_*.npz")))
            d = cat(fs, ("start", "cpi", "mscs", "area", "stop"))
            ok = (len(np.unique(d["start"])) == n and np.isfinite(d["cpi"]).all() and np.isfinite(d["mscs"]).all()
                  and np.isfinite(d["area"]).all() and np.sum(d["stop"] != "") == n)
            if not ok:
                check(f"{root}/{b}: {n} complete walks, finite values", False, f"{len(np.unique(d['start']))} walks")
        say(f"  checked {root}: 17 benchmarks x {n} walks")
    j = cat(sorted(glob.glob(os.path.join(TR, "results", "joint_17", "part_*.npz"))), ("start", "cpi", "cpi_wl", "stop"))
    check("joint_17: 1000 walks, CPI = geomean of the 17 workload CPIs",
          len(np.unique(j["start"])) == 1000 and np.allclose(j["cpi"], np.exp(np.log(j["cpi_wl"]).mean(1)), rtol=1e-5),
          f"{len(np.unique(j['start']))} walks")
    for root, pat, n, ln in (("results_bo", "seed_*.npz", 10, 500), ("results_ga", "seed_*.npz", 10, 512), ("results_rw", "part_*.npz", 10, 10000)):
        bad = []
        for b in B17:
            fs = sorted(glob.glob(os.path.join(TR, root, b, pat)))
            if len(fs) != n or any(len(np.load(f)["cpi"]) != ln or not np.isfinite(np.load(f)["cpi"]).all() for f in fs):
                bad.append(b)
        check(f"{root}: 17 x {n} files x {ln} finite evaluations", not bad, ",".join(bad))
    # re-evaluation with fresh predictors
    fa = FastArea(); worst = {}
    for b in RNG.choice(B17, 6, replace=False):
        P = T.WLPredictor(b)
        for name, pat in (("greedy random", "results"), ("greedy small", "results_small"), ("greedy seed2", "results_seed2"),
                          ("BO", "results_bo"), ("GA", "results_ga"), ("RW", "results_rw")):
            fs = sorted(glob.glob(os.path.join(TR, pat, b, "*.npz")))
            z = np.load(fs[RNG.randint(len(fs))]); idx = RNG.choice(len(z["cpi"]), 40, replace=False)
            e = []
            for i in idx:
                c = T.S.project17(cfg_of(z["cfg"][i])); cpi, m = P(c)
                e.append((abs(cpi - z["cpi"][i]) / z["cpi"][i], np.abs(m - z["mscs"][i]).max(), abs(fa(c) - z["area"][i]) / z["area"][i]))
            w = np.max(e, 0); worst[name] = np.maximum(worst.get(name, 0), w)
        if b == B17[0] or True:
            # greedy candidate values (k_cpi) of one state
            z = np.load(sorted(glob.glob(os.path.join(TR, "results", b, "part_*.npz")))[0]); i = RNG.randint(len(z["cpi"]))
            c = T.S.project17(cfg_of(z["cfg"][i])); kerr = 0
            for k, (p, dd) in enumerate(T.MOVES):
                cn, _ = T.move(c, p, dd)
                if (cn is None) != (not z["k_valid"][i, k]):
                    kerr = np.inf; break
                if cn is not None:
                    kerr = max(kerr, abs(P(cn)[0] - z["k_cpi"][i, k]) / z["k_cpi"][i, k], abs(fa(cn) - z["k_area"][i, k]) / z["k_area"][i, k])
            worst["greedy candidates"] = max(worst.get("greedy candidates", 0), kerr)
    for name, w in worst.items():
        w = np.atleast_1d(w)
        check(f"re-evaluation matches stored values: {name}", np.all(w < 1e-5), "max rel/abs err " + " ".join(f"{x:.1e}" for x in w))
    # bit-exact re-runs of small pieces
    os.makedirs(SCR, exist_ok=True)
    b = "602.gcc_s"; P = T.WLPredictor(b); ev = lambda cs: [{"cpi": a, "mscs": m} for a, m in (P(c) for c in cs)]
    import run_traj as RT
    st = RT.starts(1000)
    z = cat(sorted(glob.glob(os.path.join(TR, "results", b, "part_*.npz")))[:1], ("start", "cpi", "cfg", "kind", "act"))
    ok = True
    for s in (0, 7, 31):
        recs = T.walk(st[s], ev, fa.many); arr = T.pack([recs], [s]); r = z["start"] == s
        ok &= np.array_equal(arr["cfg"], z["cfg"][r]) and np.array_equal(arr["cpi"], z["cpi"][r]) and np.array_equal(arr["act"], z["act"][r])
    check("greedy walks 0, 7, 31 of gcc re-run bit-exact", ok)
    import run_ga_traj as RG
    g = RG.run_one(P, fa, RG.a_ref(b), 3); zg = np.load(os.path.join(TR, "results_ga", b, "seed_03.npz"))
    check("GA gcc seed 3 re-run bit-exact", all(np.array_equal(g[k], zg[k]) for k in ("cfg", "cpi", "area", "feasible")))
    import run_rw_traj as RR
    rng = np.random.RandomState(RR.RW_SEED); cs = [T.S.sample17(rng) for _ in range(300)]
    zr = np.load(os.path.join(TR, "results_rw", b, "part_000.npz"))
    check("RW gcc first 300 designs re-generated bit-exact", np.array_equal(np.array([[c[p] for p in T.SEARCH] for c in cs]), zr["cfg"][:300]))


# ------------------------------------------------------------------ L3
def L3():
    say("\nL3. algorithms")
    for root in ("results", "results_small"):
        n_chk = n_bad_next = n_bad_rule = n_bad_valid = n_tie = 0
        for b in B17:
            z = cat(sorted(glob.glob(os.path.join(TR, root, b, "part_*.npz"))),
                    ("start", "step", "cfg", "cpi", "area", "kind", "act", "stop", "k_valid", "k_cpi", "k_area"))
            st = z["start"]
            for s in RNG.choice(np.unique(st), 40, replace=False):
                r = np.where(st == s)[0]; visited = set()
                for t, i in enumerate(r):
                    c = cfg_of(z["cfg"][i]); visited.add(T.ckey(c)); n_chk += 1
                    cands = [T.move(c, p, dd)[0] for p, dd in T.MOVES]
                    n_bad_valid += any((cn is None) == z["k_valid"][i, k] for k, cn in enumerate(cands))
                    if z["stop"][i] == "max_steps":
                        continue
                    cpi0, a0 = float(z["cpi"][i]), float(z["area"][i]); kc, ka = z["k_cpi"][i].astype(float), z["k_area"][i].astype(float)
                    rc = [(k, a0 - ka[k]) for k in range(34) if cands[k] is not None and T.MOVES[k][1] < 0
                          and (kc[k] - cpi0) / cpi0 <= T.L_MAX and ka[k] < a0 and T.ckey(cands[k]) not in visited]
                    up = [(k, ((cpi0 - kc[k]) / cpi0) / max((ka[k] - a0) / a0, 1e-3)) for k in range(34)
                          if cands[k] is not None and T.MOVES[k][1] > 0 and (cpi0 - kc[k]) / cpi0 >= T.G_MIN and T.ckey(cands[k]) not in visited]
                    want = (max(rc, key=lambda x: x[1])[0], 2) if rc else ((max(up, key=lambda x: x[1])[0], 1) if up else (-1, 0))
                    got = (int(z["act"][i]), int(z["kind"][i]))
                    if want != got:
                        # float32 storage can flip a decision only at a near-tie / threshold edge
                        near = any(abs((kc[k] - cpi0) / cpi0 - T.L_MAX) < 1e-5 or abs((cpi0 - kc[k]) / cpi0 - T.G_MIN) < 1e-5 for k in range(34) if cands[k] is not None)
                        vals = sorted([v for _, v in (rc or up)], reverse=True)
                        near |= len(vals) > 1 and abs(vals[0] - vals[1]) <= 1e-5 * max(abs(vals[0]), 1e-9)
                        n_tie += near; n_bad_rule += not near
                    if got[0] >= 0 and t + 1 < len(r):
                        n_bad_next += T.ckey(cfg_of(z["cfg"][r[t + 1]])) != T.ckey(cands[got[0]]) or \
                                      abs(z["cpi"][r[t + 1]] - z["k_cpi"][i, got[0]]) > 1e-6 * z["cpi"][r[t + 1]]
        check(f"greedy replay {root} (40 walks x 17, {n_chk} states): next state = chosen candidate", n_bad_next == 0, f"{n_bad_next} bad")
        check(f"greedy replay {root}: decision rule reproduced", n_bad_rule == 0, f"{n_bad_rule} bad, {n_tie} float32 near-ties")
        check(f"greedy replay {root}: candidate validity = move()", n_bad_valid == 0, f"{n_bad_valid} bad")
    bad = []
    for b in B17:
        for f in sorted(glob.glob(os.path.join(TR, "results_bo", b, "seed_*.npz"))):
            z = np.load(f)
            if not (np.array_equal(z["phase"], (np.arange(500) >= 30).astype(np.int8)) and np.allclose(z["cpi"], z["cpi_bo"], rtol=1e-5)):
                bad.append(f)
    check("BO: 30 random initial points then 500 total; recorded CPI = CPI the optimiser saw", not bad, f"{len(bad)} files")
    bad = []
    for b in B17:
        A = json.load(open(os.path.join(TR, "results_ga", b, "meta.json")))["A_ref"]
        for f in sorted(glob.glob(os.path.join(TR, "results_ga", b, "seed_*.npz"))):
            z = np.load(f)
            if not (np.array_equal(z["feasible"], z["area"] <= A) and np.array_equal(z["gen"], np.arange(512) // 16)):
                bad.append(f)
    check("GA: feasible flag = area <= A_ref, 32 generations x 16", not bad, f"{len(bad)} files")
    ref = np.load(os.path.join(TR, "results_rw", B17[0], "part_003.npz"))["cfg"]
    check("RW: every benchmark sees the same designs", all(np.array_equal(np.load(os.path.join(TR, "results_rw", b, "part_003.npz"))["cfg"], ref) for b in B17))


# ------------------------------------------------------------------ L4
def L4():
    say("\nL4. analysis")
    import analyze_ga_rw as AG
    rng = np.random.RandomState(1); ok = True
    for _ in range(200):
        n = rng.randint(1, 60); c = rng.rand(n).round(2); a = rng.rand(n).round(2)
        p1 = set(AG.pareto(c, a).tolist()); p2 = set(pareto_bf(c, a).tolist())
        # analysis keeps one representative of exact duplicates; compare as (cpi, area) sets
        ok &= {(c[i], a[i]) for i in p1} == {(c[i], a[i]) for i in p2}
    check("Pareto routine = brute force on 200 random sets (incl. ties)", ok)
    check("Jain: equal -> 1, single -> 1/3, scale-invariant",
          np.isclose(jain(np.array([[1., 1, 1]]))[0], 1) and np.isclose(jain(np.array([[2., 0, 0]]))[0], 1 / 3)
          and np.allclose(jain(np.array([[1., 2, 3]])), jain(np.array([[7., 14, 21]]))))
    z = cat(sorted(glob.glob(os.path.join(TR, "results_small", "602.gcc_s", "part_*.npz"))), ("start",))
    last, first = ends_idx(z["start"]); st = z["start"]
    l2 = np.r_[np.where(np.diff(st) != 0)[0], len(st) - 1]
    check("walk-end extraction used in the analyses = groupby", np.array_equal(np.sort(last), l2))

    # independent re-computation of headline numbers
    cent = np.load(os.path.join(TR, "results", "state_clusters_v1.npz"))["centroids"]
    def lab(cpi, m):
        X = np.stack([cpi * m[:, 4 * s + k] for s in (1, 3) for k in (1, 2, 3)], 1); X = X / X.sum(1, keepdims=True)
        return np.argmin(((X[:, None] - cent[None]) ** 2).sum(-1), 1)
    rise, agree, ctrl, modes = 0, {"BO": 0, "GA": 0, "RW-100k": 0}, [], {}
    for b in B17:
        g = cat(sorted(glob.glob(os.path.join(TR, "results_small", b, "part_*.npz"))), ("start", "cpi", "mscs", "area"))
        last, first = ends_idx(g["start"]); Jg = jain(x3(g["cpi"], g["mscs"]))
        rise += np.median(Jg[last]) > np.median(Jg[first])
        A = float(np.median(g["area"][last])); gm = np.bincount(lab(g["cpi"][last], g["mscs"][last]), minlength=6).argmax()
        modes[b] = {"greedy": gm}
        L = []
        for f in sorted(glob.glob(os.path.join(TR, "results_bo", b, "seed_*.npz"))):
            zz = np.load(f); p = pareto_bf(zz["cpi"].astype(float), zz["area"].astype(float)); i = p[np.argmin(abs(zz["area"][p] - A))]
            L.append(lab(zz["cpi"][[i]], zz["mscs"][[i]])[0])
        modes[b]["BO"] = np.bincount(L, minlength=6).argmax()
        L = []
        for f in sorted(glob.glob(os.path.join(TR, "results_ga", b, "seed_*.npz"))):
            zz = np.load(f); ok_ = zz["area"] <= A
            if ok_.any():
                i = np.where(ok_)[0][np.argmin(zz["cpi"][ok_])]; L.append(lab(zz["cpi"][[i]], zz["mscs"][[i]])[0])
        modes[b]["GA"] = np.bincount(L, minlength=6).argmax()
        rw = cat(sorted(glob.glob(os.path.join(TR, "results_rw", b, "part_*.npz"))), ("cpi", "mscs", "area"))
        o = np.argsort(rw["area"], kind="stable"); best = np.minimum.accumulate(rw["cpi"][o]); fr = o[np.r_[True, best[1:] < best[:-1]]]
        i = fr[np.argmin(abs(rw["area"][fr] - A))]; modes[b]["RW-100k"] = lab(rw["cpi"][[i]], rw["mscs"][[i]])[0]
        for k in agree:
            agree[k] += modes[b][k] == gm
        m = np.abs(rw["area"] / A - 1) <= 0.05; Jr = jain(x3(rw["cpi"][m], rw["mscs"][m])); c = rw["cpi"][m]
        ctrl.append(np.median(Jr[c <= np.quantile(c, 0.02)]) - np.median(Jr))
    ctrl = np.array(ctrl)
    say(f"  independent: Dispatch Jain end > start (median over small-start walks) in {rise}/17   [reported 17/17]")
    say(f"  independent: state mode at A_ref == greedy: " + ", ".join(f"{k} {v}/17" for k, v in agree.items())
        + "   [reported BO 14, GA 13, RW-100k 13]")
    say(f"  independent: best-2% vs all at A_ref+-5%: best more balanced {np.sum(ctrl > 0)}/17, median {np.median(ctrl):+.3f}   [reported 16/17, +0.059]")
    check("headline numbers reproduced by independent code",
          rise == 17 and (agree["BO"], agree["GA"], agree["RW-100k"]) == (14, 13, 13) and np.sum(ctrl > 0) == 16 and abs(np.median(ctrl) - 0.059) < 0.002)
    # shuffle null: how much agreement would chance give, given how concentrated the states are?
    say("  benchmark-shuffle null (BO/GA/RW mode of benchmark i vs greedy mode of benchmark pi(i)), 10k permutations:")
    for k in agree:
        a = np.array([modes[b][k] for b in B17]); g_ = np.array([modes[b]["greedy"] for b in B17])
        null = np.array([np.sum(a == g_[np.random.RandomState(s).permutation(17)]) for s in range(10000)])
        say(f"    {k:8s} observed {agree[k]}/17, null mean {null.mean():.1f}, p(null >= observed) = {np.mean(null >= agree[k]):.4f}")


def main():
    out = os.path.join(HERE, "report_v1.txt")
    if os.path.exists(out):
        out = out.replace(".txt", f".{os.getpid()}.txt")
    t0 = time.time()
    for f in (L1, L2, L3, L4):
        f(); say(f"  ({time.time()-t0:.0f}s)")
    say(f"\nSUMMARY: {len(FAILS)} FAIL" + (": " + "; ".join(FAILS) if FAILS else ""))
    open(out, "w").write("\n".join(OUT) + "\n"); print("->", out)


if __name__ == "__main__":
    main()
