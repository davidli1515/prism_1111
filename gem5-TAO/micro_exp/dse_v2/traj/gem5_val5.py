#!/usr/bin/env python3
"""
gem5 round 5: start / end incumbents of BO, GA and the random walker (RW) for the appendix dumbbell figure
(greedy is covered by gem5_val4). Definitions are those of plot_stage_jain_optimizers.py (imported):
  A_ref     = median end area of the greedy small-start walks of the benchmark
  incumbent = lowest-CPI design evaluated so far with area <= A_ref; start = its first appearance in the
              run, end = the incumbent at the end of the run
  runs      = 10 BO seeds / 10 GA seeds / 10 RW blocks of 512 random designs
Representative run (per optimiser and benchmark): among runs with an incumbent, the one whose 8-vector
(start and end Jain at Decode, Dispatch, Issue, Commit; model) is closest (L2) to the element-wise median
over those runs. Its start and end designs are simulated (one run if start == end): <= 3 x 17 x 2 = 102.
Same runner and command line as gem5_val..gem5_val4. Writes gem5_val5/ (plan.json, runs/, summary.txt);
never overwrites.
  python gem5_val5.py --dry-run          # build/print the plan only (no gem5)
  python gem5_val5.py                    # run (GEM5_WORKERS, default 16), then summary
  python gem5_val5.py --analyze-only
"""
import os, sys, json, glob, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DSE_V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DSE_V2, "gem5_bounds")); sys.path.insert(0, HERE)
import traj_common as T
from run_gem5_bounds import run_job, surrogate_insts
from plot_stage_jain_optimizers import jain4, cat, first_last_incumbent

OUT = os.path.join(HERE, "gem5_val5")
OPTS = ["BO", "GA", "Random"]
STAGES = ["Decode", "Dispatch", "Issue", "Commit"]


def runs_of(b, rw):
    return {"BO": [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "results_bo", b, "seed_*.npz")))],
            "GA": [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "results_ga", b, "seed_*.npz")))],
            "Random": [{k: rw[k][s * 512:(s + 1) * 512] for k in rw} for s in range(10)]}


def plan():
    jobs = []
    for b in T.CM.SPEC17:
        g = cat(sorted(glob.glob(os.path.join(HERE, "results_small", b, "part_*.npz"))), ("start", "area"))
        A = float(np.median(g["area"][np.r_[np.where(np.diff(g["start"]) != 0)[0], len(g["start"]) - 1]]))
        rw = cat(sorted(glob.glob(os.path.join(HERE, "results_rw", b, "part_*.npz"))), ("cfg", "cpi", "mscs", "area"))
        for opt, rs in runs_of(b, rw).items():
            cand = []
            for r, z in enumerate(rs):
                f, l = first_last_incumbent(z["cpi"], z["area"], A)
                if f >= 0:
                    J = jain4(z["cpi"][[f, l]], z["mscs"][[f, l]]); cand.append((r, f, l, J.ravel()))
            if not cand:
                print(f"  {b} {opt}: no run with a design of area <= A_ref -> skipped"); continue
            V = np.array([c[3] for c in cand]); med = np.median(V, 0)
            r, f, l, _ = cand[int(np.argmin(((V - med) ** 2).sum(1)))]
            z = rs[r]
            for which, i in (("start", f), ("end", l)):
                c = T.S.project17(dict(zip(T.SEARCH, map(int, z["cfg"][i]))))
                jobs.append({"bench": b, "opt": opt, "run": int(r), "which": which, "eval_idx": int(i), "same_as_start": bool(which == "end" and l == f),
                             "idx": int(r), "tag": f"{opt.lower()}_{which}", "config": c, "A_ref": A, "area": float(z["area"][i]),
                             "model_cpi": float(z["cpi"][i]), "model_mscs": z["mscs"][i].tolist(), "n_runs_with_incumbent": len(cand),
                             "insts": surrogate_insts(b, c["l1_size"])})
    return jobs


def result_path(j):
    return os.path.join(OUT, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")


def gem5_of(j, jobs):
    """result for a job; an end incumbent identical to its start reuses the start run"""
    if j["same_as_start"]:
        j = next(x for x in jobs if x["bench"] == j["bench"] and x["opt"] == j["opt"] and x["which"] == "start")
    p = result_path(j)
    return json.load(open(p)) if os.path.exists(p) else None


def analyze(jobs):
    L = []; rows = []
    for j in jobs:
        r = gem5_of(j, jobs)
        if r is not None:
            rows.append((j, r["cpi"], np.array(r["mscs16"])))
    L.append(f"{len(rows)}/{len(jobs)} designs with gem5 results")
    if not rows:
        return L
    mc = np.array([j["model_cpi"] for j, _, _ in rows]); gc = np.array([c for _, c, _ in rows])
    mm = np.array([j["model_mscs"] for j, _, _ in rows]); gm = np.array([m for _, _, m in rows])
    L.append(f"CPI MAPE {100*np.mean(abs(mc-gc)/gc):.1f}%, MSCS MAE {np.mean(abs(mm-gm)):.3f}")
    Jm, Jg = jain4(mc, mm), jain4(gc, gm)
    L.append("per optimiser: per-stage Jain start -> end, median over benchmarks with both designs, model | gem5")
    for opt in OPTS:
        for s, n in enumerate(STAGES):
            st = {j["bench"]: i for i, (j, _, _) in enumerate(rows) if j["opt"] == opt and j["which"] == "start"}
            en = {j["bench"]: i for i, (j, _, _) in enumerate(rows) if j["opt"] == opt and j["which"] == "end"}
            bb = sorted(set(st) & set(en))
            if not bb:
                continue
            m0, m1 = np.median([Jm[st[b], s] for b in bb]), np.median([Jm[en[b], s] for b in bb])
            g0, g1 = np.median([Jg[st[b], s] for b in bb]), np.median([Jg[en[b], s] for b in bb])
            L.append(f"  {opt:7s} {n:8s} model {m0:.2f}->{m1:.2f} ({m1-m0:+.3f}) | gem5 {g0:.2f}->{g1:.2f} ({g1-g0:+.3f})  n={len(bb)}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    jp = os.path.join(OUT, "plan.json")
    if os.path.exists(jp):
        jobs = json.load(open(jp))
    else:
        jobs = plan()
        if args.dry_run:
            sim = [j for j in jobs if not j["same_as_start"]]
            print(f"{len(jobs)} designs, {len(sim)} gem5 runs (end == start in {len(jobs) - len(sim)})")
            for opt in OPTS:
                x = [j for j in jobs if j["opt"] == opt and j["which"] == "start"]
                print(f"  {opt:7s} {len(x)} benchmarks; runs with an incumbent per benchmark: "
                      f"min {min(j['n_runs_with_incumbent'] for j in x)}, chosen run ids {[j['run'] for j in x]}")
            return
        os.makedirs(OUT, exist_ok=True); json.dump(jobs, open(jp, "w"), indent=1)
    if not args.analyze_only and not args.dry_run:
        sim = [j for j in jobs if not j["same_as_start"]]
        workers = int(os.environ.get("GEM5_WORKERS", 16))
        print(f"{len(jobs)} designs, {len(sim)} gem5 runs, {workers} workers", flush=True)
        t0 = time.time(); ok = 0
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(run_job, {k: j[k] for k in ("bench", "idx", "tag", "config", "insts")}, OUT) for j in sim]
            for f in as_completed(futs):
                r = f.result(); ok += r.get("error") is None
                print(f"  {r['bench']:16s} p{r['idx']}_{r['tag']} {'skip' if r.get('skipped') else r.get('error') or 'ok'} "
                      f"[{time.time()-t0:.0f}s] ok {ok}/{len(sim)}", flush=True)
    if args.dry_run:
        return
    sp = os.path.join(OUT, "summary.txt")
    if os.path.exists(sp):
        sp = sp.replace(".txt", f".{os.getpid()}.txt")
    L = analyze(jobs); open(sp, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
