#!/usr/bin/env python3
"""
gem5 round 6: incumbent TRAJECTORIES of BO, GA and the random walker (RW) for the per-optimiser Jain trajectory
figure (greedy trajectories are covered by gem5_val4).
  A_ref     = median end area of the greedy small-start walks of the benchmark
  incumbent = lowest-CPI design so far with area <= A_ref
  progress  = 0 at the first incumbent of the run, 1 at the end of the run (evaluations, linear)
  runs      = 10 BO seeds / 10 GA seeds / 10 RW blocks of 512 random designs
Representative run (same rule as the greedy walks of gem5_val3/4): the run whose model Dispatch-Jain incumbent
trajectory (21 progress points) is closest (RMS) to the benchmark's median trajectory. States: the incumbent at
K = 12 evenly spaced progress points; identical designs (same evaluation) are simulated once; designs already
simulated in gem5_val5 (same benchmark, optimiser, run, evaluation) are copied, never re-run.
Writes gem5_val6/ (plan.json, runs/, summary.txt); never overwrites.
  python gem5_val6.py --dry-run | (run) | --analyze-only        (GEM5_WORKERS, default 16)
"""
import os, sys, json, glob, time, shutil, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DSE_V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DSE_V2, "gem5_bounds")); sys.path.insert(0, HERE)
import traj_common as T
from run_gem5_bounds import run_job, surrogate_insts
from plot_stage_jain_optimizers import jain4, cat

OUT, V5 = os.path.join(HERE, "gem5_val6"), os.path.join(HERE, "gem5_val5")
OPTS, K, GX = ["BO", "GA", "Random"], 12, np.linspace(0, 1, 21)


def incumbent_index(cpi, area, A):
    idx = np.full(len(cpi), -1); cur = -1
    for i in range(len(cpi)):
        if area[i] <= A and (cur < 0 or cpi[i] < cpi[cur]):
            cur = i
        idx[i] = cur
    return idx


def at_progress(idx, grid):
    """incumbent eval index at each progress point (0 = first incumbent, 1 = last evaluation)"""
    f = int(np.argmax(idx >= 0)); n = len(idx)
    return idx[np.clip(np.round(f + grid * (n - 1 - f)).astype(int), f, n - 1)]


def runs_of(b, rw):
    return {"BO": [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "results_bo", b, "seed_*.npz")))],
            "GA": [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "results_ga", b, "seed_*.npz")))],
            "Random": [{k: rw[k][s * 512:(s + 1) * 512] for k in rw} for s in range(10)]}


def a_ref(b):
    g = cat(sorted(glob.glob(os.path.join(HERE, "results_small", b, "part_*.npz"))), ("start", "area"))
    return float(np.median(g["area"][np.r_[np.where(np.diff(g["start"]) != 0)[0], len(g["start"]) - 1]]))


def plan():
    jobs = []
    for b in T.CM.SPEC17:
        A = a_ref(b)
        rw = cat(sorted(glob.glob(os.path.join(HERE, "results_rw", b, "part_*.npz"))), ("cfg", "cpi", "mscs", "area"))
        for opt, rs in runs_of(b, rw).items():
            cand = []
            for r, z in enumerate(rs):
                idx = incumbent_index(z["cpi"], z["area"], A)
                if (idx >= 0).any():
                    Jd = jain4(z["cpi"], z["mscs"])[:, 1]; cand.append((r, idx, Jd[at_progress(idx, GX)]))
            if not cand:
                print(f"  {b} {opt}: no run with a design of area <= A_ref -> skipped"); continue
            C = np.array([c[2] for c in cand]); med = np.median(C, 0)
            k = int(np.argmin(np.sqrt(((C - med) ** 2).mean(1)))); r, idx, _ = cand[k]; z = rs[r]
            ev = at_progress(idx, np.linspace(0, 1, K))
            for p, i in zip(np.linspace(0, 1, K), ev):
                c = T.S.project17(dict(zip(T.SEARCH, map(int, z["cfg"][i]))))
                jobs.append({"bench": b, "opt": opt, "run": int(r), "progress": float(p), "eval_idx": int(i),
                             "idx": int(r), "tag": f"{opt.lower()}_e{int(i):03d}", "config": c, "A_ref": A,
                             "area": float(z["area"][i]), "model_cpi": float(z["cpi"][i]), "model_mscs": z["mscs"][i].tolist(),
                             "rms_to_median": float(np.sqrt(((C[k] - med) ** 2).mean())), "n_runs": len(cand),
                             "insts": surrogate_insts(b, c["l1_size"])})
    return jobs


def result_path(root, j):
    return os.path.join(root, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")


def v5_source(j):
    """same design already simulated in gem5_val5 (same benchmark, optimiser, run and evaluation)?"""
    jp = os.path.join(V5, "plan.json")
    if not os.path.exists(jp):
        return None
    for x in json.load(open(jp)):
        if (x["bench"], x["opt"], x["run"], x["eval_idx"]) == (j["bench"], j["opt"], j["run"], j["eval_idx"]):
            p = os.path.join(V5, "runs", x["bench"], f"p{x['idx']}_{x['tag']}", "result.json")
            return p if os.path.exists(p) else None
    return None


def unique_sims(jobs):
    seen, out = set(), []
    for j in jobs:
        k = (j["bench"], j["idx"], j["tag"])
        if k not in seen:
            seen.add(k); out.append(j)
    return out


def analyze(jobs):
    rows = [(j, json.load(open(result_path(OUT, j)))) for j in jobs if os.path.exists(result_path(OUT, j))]
    L = [f"{len(rows)}/{len(jobs)} trajectory points with gem5 results"]
    if not rows:
        return L
    mc = np.array([j["model_cpi"] for j, _ in rows]); gc = np.array([r["cpi"] for _, r in rows])
    mm = np.array([j["model_mscs"] for j, _ in rows]); gm = np.array([r["mscs16"] for _, r in rows])
    L.append(f"CPI MAPE {100*np.mean(abs(mc-gc)/gc):.1f}%, MSCS MAE {np.mean(abs(mm-gm)):.3f}")
    Jm, Jg = jain4(mc, mm)[:, 1], jain4(gc, gm)[:, 1]
    for opt in OPTS:
        for tag, J in (("model", Jm), ("gem5", Jg)):
            M = {}
            for (j, _), v in zip(rows, J):
                if j["opt"] == opt:
                    M.setdefault(j["bench"], {})[round(j["progress"], 3)] = v
            P = sorted({p for d in M.values() for p in d})
            med = [np.median([d[p] for d in M.values() if p in d]) for p in P]
            L.append(f"  {opt:7s} {tag:5s} Dispatch Jain (median over {len(M)} benchmarks): " + " ".join(f"{v:.2f}" for v in med))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    jp = os.path.join(OUT, "plan.json")
    jobs = json.load(open(jp)) if os.path.exists(jp) else plan()
    sims = unique_sims(jobs)
    reuse = [j for j in sims if v5_source(j)]
    if args.dry_run:
        print(f"{len(jobs)} trajectory points, {len(sims)} distinct designs, {len(reuse)} reusable from gem5_val5 "
              f"-> {len(sims) - len(reuse)} new gem5 runs")
        for opt in OPTS:
            per = [len({(j['tag']) for j in sims if j['opt'] == opt and j['bench'] == b}) for b in T.CM.SPEC17]
            per = [x for x in per if x]
            print(f"  {opt:7s} {len(per)} benchmarks; distinct designs per walk min {min(per)} median {int(np.median(per))} max {max(per)}")
        return
    if not os.path.exists(jp):
        os.makedirs(OUT, exist_ok=True); json.dump(jobs, open(jp, "w"), indent=1)
    if not args.analyze_only:
        for j in reuse:                                   # copy, never overwrite
            dst = result_path(OUT, j)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(v5_source(j), dst)
        todo = [j for j in sims if not os.path.exists(result_path(OUT, j))]
        workers = int(os.environ.get("GEM5_WORKERS", 16))
        print(f"{len(sims)} distinct designs, {len(reuse)} copied from gem5_val5, {len(todo)} gem5 runs, {workers} workers", flush=True)
        t0 = time.time(); ok = 0
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(run_job, {k: j[k] for k in ("bench", "idx", "tag", "config", "insts")}, OUT) for j in todo]
            for f in as_completed(futs):
                r = f.result(); ok += r.get("error") is None
                print(f"  {r['bench']:16s} p{r['idx']}_{r['tag']} {'skip' if r.get('skipped') else r.get('error') or 'ok'} "
                      f"[{time.time()-t0:.0f}s] ok {ok}/{len(todo)}", flush=True)
    sp = os.path.join(OUT, "summary.txt")
    if os.path.exists(sp):
        sp = sp.replace(".txt", f".{os.getpid()}.txt")
    L = analyze(jobs); open(sp, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
