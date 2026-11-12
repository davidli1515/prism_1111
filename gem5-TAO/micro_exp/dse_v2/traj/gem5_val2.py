#!/usr/bin/env python3
"""
gem5 round 2 (writes gem5_val2/, never overwrites; same runner/command line as gem5_val and gem5_bounds).
 A. static control "best designs are more balanced than ordinary designs of the same area" (results_rw/):
    benchmarks with a large model effect (blender, nab, cactuBSSN, imagick) + one with ~0 effect (xz, negative
    control). Area window = A_ref +-5% (A_ref = greedy small-start median end area). top = the 30 lowest
    model-CPI designs of the window; ord = 30 random designs from the rest of the window (seed 0).
    Pre-registered test: metric = Dispatch Jain (FP/BP/SP stall cycles); per benchmark diff of means
    (top - ord) with a two-sided permutation p-value (10k); pooled over the 4 effect benchmarks with a
    stratified permutation; success = gem5 diff > 0 and p < 0.05 on the pool, sign agreement per benchmark.
    Reported next to the model diff on the SAME designs, and a CPI sanity check (top faster on gem5).
 B. cactuBSSN rise-then-fall of Dispatch balance (results/ random-start greedy walks): 4 walks with model
    Jain peak - start >= 0.10 and peak - end >= 0.05 (largest drop first); up to 14 states per walk
    (evenly spaced + peak and its neighbours + first + last).
  python gem5_val2.py [--analyze-only]      (GEM5_WORKERS, default 16)
"""
import os, sys, json, glob, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DSE_V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DSE_V2, "gem5_bounds")); sys.path.insert(0, HERE)
import traj_common as T
from run_gem5_bounds import run_job, surrogate_insts

OUT = os.path.join(HERE, "gem5_val2")
A_BENCH = ["526.blender_r", "644.nab_s", "607.cactuBSSN_s", "638.imagick_s", "657.xz_s"]
NEG = {"657.xz_s"}
N_SET, WIN, PER_WALK, N_WALKS = 30, 0.05, 14, 4


def jain(cpi, m):
    x = np.stack([cpi * m[:, 4 + k] for k in (1, 2, 3)], 1); return x.sum(1) ** 2 / (3 * (x ** 2).sum(1))


def cat(fs, keys):
    return {k: np.concatenate([np.load(f)[k] for f in fs]) for k in keys}


def a_ref(b):
    g = cat(sorted(glob.glob(os.path.join(HERE, "results_small", b, "part_*.npz"))), ("start", "area"))
    return float(np.median(g["area"][np.r_[np.where(np.diff(g["start"]) != 0)[0], len(g["start"]) - 1]]))


def job(b, idx, tag, cfg_row, extra):
    c = T.S.project17(dict(zip(T.SEARCH, map(int, cfg_row))))
    return {"bench": b, "idx": int(idx), "tag": tag, "config": c, "insts": surrogate_insts(b, c["l1_size"])} | extra


def plan():
    jobs = []
    for b in A_BENCH:                                                   # ---- A
        rw = cat(sorted(glob.glob(os.path.join(HERE, "results_rw", b, "part_*.npz"))), ("idx", "cfg", "area", "cpi", "mscs"))
        A = a_ref(b); w = np.where(np.abs(rw["area"] / A - 1) <= WIN)[0]
        top = w[np.argsort(rw["cpi"][w])[:N_SET]]
        rest = np.setdiff1d(w, top); ordi = np.random.RandomState(0).choice(rest, N_SET, replace=False)
        for grp, ids in (("top", top), ("ord", ordi)):
            for i in ids:
                jobs.append(job(b, rw["idx"][i], grp, rw["cfg"][i],
                                {"part": "A", "group": grp, "A_ref": A, "area": float(rw["area"][i]),
                                 "model_cpi": float(rw["cpi"][i]), "model_mscs": rw["mscs"][i].tolist()}))
    b = "607.cactuBSSN_s"                                               # ---- B
    d = cat(sorted(glob.glob(os.path.join(HERE, "results", b, "part_*.npz"))), ("start", "step", "cfg", "cpi", "mscs"))
    J = jain(d["cpi"], d["mscs"]); cand = []
    for s in np.unique(d["start"]):
        r = np.where(d["start"] == s)[0]; j = J[r]; p = int(np.argmax(j))
        if j[p] - j[0] >= 0.10 and j[p] - j[-1] >= 0.05:
            cand.append((j[p] - j[-1], s, r, p))
    cand.sort(key=lambda x: -x[0])
    for drop, s, r, p in cand[:N_WALKS]:
        keep = set(np.linspace(0, len(r) - 1, min(PER_WALK, len(r))).round().astype(int)) | {0, len(r) - 1, p, max(p - 1, 0), min(p + 1, len(r) - 1)}
        for k in sorted(keep):
            i = r[k]
            jobs.append(job(b, s, f"s{int(d['step'][i]):02d}", d["cfg"][i],
                            {"part": "B", "step": int(d["step"][i]), "peak_step": int(d["step"][r[p]]), "model_drop": float(drop),
                             "model_cpi": float(d["cpi"][i]), "model_mscs": d["mscs"][i].tolist()}))
    return jobs


def perm_p(x, y, rng, n=10000):
    obs = x.mean() - y.mean(); z = np.r_[x, y]; c = 0
    for _ in range(n):
        rng.shuffle(z); c += abs(z[:len(x)].mean() - z[len(x):].mean()) >= abs(obs) - 1e-12
    return obs, (c + 1) / (n + 1)


def analyze(jobs):
    rows = []
    for j in jobs:
        p = os.path.join(OUT, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
        if os.path.exists(p):
            r = json.load(open(p)); rows.append((j, r["cpi"], np.array(r["mscs16"])))
    L = [f"{len(rows)}/{len(jobs)} gem5 runs available"]
    if not rows:
        return L
    mc = np.array([j["model_cpi"] for j, _, _ in rows]); gc = np.array([c for _, c, _ in rows])
    mm = np.array([j["model_mscs"] for j, _, _ in rows]); gm = np.array([m for _, _, m in rows])
    Jm, Jg = jain(mc, mm), jain(gc, gm)
    L.append(f"all runs: CPI MAPE {100*np.mean(abs(mc-gc)/gc):.1f}%, MSCS MAE {np.mean(abs(mm-gm)):.3f}, "
             f"Dispatch Jain MAE {np.mean(abs(Jm-Jg)):.3f}")
    rng = np.random.RandomState(0)
    L.append("\nA. static control at A_ref +-5%: Dispatch Jain top-30 (lowest model CPI) vs 30 ordinary designs")
    L.append(f"  {'bench':12s} n(top,ord) | model diff | gem5 diff  p(perm) | gem5 CPI top/ord | gem5 Jain top / ord")
    pool = []
    for b in A_BENCH:
        it = [i for i, (j, _, _) in enumerate(rows) if j["part"] == "A" and j["bench"] == b]
        t = np.array([i for i in it if rows[i][0]["group"] == "top"]); o = np.array([i for i in it if rows[i][0]["group"] == "ord"])
        if len(t) < 3 or len(o) < 3:
            continue
        dg, p = perm_p(Jg[t], Jg[o], rng)
        L.append(f"  {b.split('.')[1]:12s} {len(t):3d},{len(o):3d} | {Jm[t].mean()-Jm[o].mean():+.3f}    | {dg:+.3f}   {p:.4f} | "
                 f"{np.mean(gc[t])/np.mean(gc[o]):.3f} | {Jg[t].mean():.2f} / {Jg[o].mean():.2f}{'   (negative control)' if b in NEG else ''}")
        if b not in NEG:
            pool.append((Jg[t], Jg[o]))
    if pool:
        obs = np.mean([x.mean() - y.mean() for x, y in pool]); c = 0
        for _ in range(10000):
            s = []
            for x, y in pool:
                z = np.r_[x, y].copy(); rng.shuffle(z); s.append(z[:len(x)].mean() - z[len(x):].mean())
            c += abs(np.mean(s)) >= abs(obs) - 1e-12
        L.append(f"  pooled over the {len(pool)} effect benchmarks: mean gem5 diff {obs:+.3f}, stratified permutation p = {(c+1)/10001:.4f}")
    L.append("\nB. cactuBSSN walks: Dispatch Jain along the walk, model / gem5")
    for s in sorted({j["idx"] for j, _, _ in rows if j["part"] == "B"}):
        ii = sorted([i for i, (j, _, _) in enumerate(rows) if j["part"] == "B" and j["idx"] == s], key=lambda i: rows[i][0]["step"])
        pk = rows[ii[0]][0]["peak_step"]; steps = [rows[i][0]["step"] for i in ii]
        L.append(f"  walk {s} (model peak at step {pk}, {len(ii)} states)")
        L.append("    step  " + " ".join(f"{x:5d}" for x in steps))
        L.append("    model " + " ".join(f"{Jm[i]:5.2f}" for i in ii))
        L.append("    gem5  " + " ".join(f"{Jg[i]:5.2f}" for i in ii))
        g = Jg[ii]; pg = int(np.argmax(g))
        L.append(f"    gem5: start {g[0]:.2f} -> max {g[pg]:.2f} (step {steps[pg]}) -> end {g[-1]:.2f}; "
                 f"rise {g[pg]-g[0]:+.2f}, fall {g[-1]-g[pg]:+.2f}; CPI end/start gem5 {gc[ii[-1]]/gc[ii[0]]:.3f}")
    return L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--analyze-only", action="store_true"); args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jp = os.path.join(OUT, "plan.json")
    if os.path.exists(jp):
        jobs = json.load(open(jp))
    else:
        jobs = plan(); json.dump(jobs, open(jp, "w"), indent=1)
    if not args.analyze_only:
        workers = int(os.environ.get("GEM5_WORKERS", 16))
        print(f"{len(jobs)} gem5 runs ({sum(j['part']=='A' for j in jobs)} A, {sum(j['part']=='B' for j in jobs)} B), "
              f"{workers} workers", flush=True)
        t0 = time.time(); ok = 0
        order = np.argsort([i for b in sorted({j["bench"] for j in jobs})            # round-robin over benchmarks,
                            for i in range(sum(j["bench"] == b for j in jobs))], kind="stable")  # so failures show early
        grouped = [j for b in sorted({j["bench"] for j in jobs}) for j in jobs if j["bench"] == b]
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(run_job, {k: j[k] for k in ("bench", "idx", "tag", "config", "insts")}, OUT)
                    for j in (grouped[i] for i in order)]
            for f in as_completed(futs):
                r = f.result(); ok += r.get("error") is None
                print(f"  {r['bench']:16s} p{r['idx']}_{r['tag']} {'skip' if r.get('skipped') else r.get('error') or 'ok'} "
                      f"[{time.time()-t0:.0f}s] ok {ok}/{len(jobs)}", flush=True)
    sp = os.path.join(OUT, "summary.txt")
    if os.path.exists(sp):
        sp = sp.replace(".txt", f".{os.getpid()}.txt")
    L = analyze(jobs); open(sp, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
