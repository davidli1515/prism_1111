#!/usr/bin/env python3
"""
gem5 round 3: ONE representative walk per benchmark, all 17 (random-start greedy walks, results/).
Selection rule (same for every benchmark): the walk whose model Dispatch-entropy curve, resampled on 21
points of walk progress, is closest (L2) to the benchmark's median curve (the blue line of fig_entropy).
States: up to 12 evenly spaced along the walk, first and last always kept (as gem5_val). Same runner and
command line as gem5_val / gem5_bounds (checkpoint, -W 20M, -I = surrogate slice range).
States already simulated in gem5_val/ (same benchmark, walk, step) are copied, not re-run.
Writes gem5_val3/ (plan.json, runs/, summary.txt); never overwrites.
  python gem5_val3.py [--analyze-only]      (GEM5_WORKERS, default 16)
"""
import os, sys, json, glob, time, shutil, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DSE_V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DSE_V2, "gem5_bounds")); sys.path.insert(0, HERE)
import traj_common as T
from run_gem5_bounds import run_job, surrogate_insts

OUT, R, OLD = os.path.join(HERE, "gem5_val3"), os.path.join(HERE, "results"), os.path.join(HERE, "gem5_val")
PER_WALK, GX = 12, np.linspace(0, 1, 21)


def ent(cpi, m):
    c = np.stack([cpi * m[:, 4 + k] for k in (1, 2, 3)], 1); p = c / np.maximum(c.sum(1, keepdims=True), 1e-12)
    return -(p * np.log(p + 1e-12)).sum(1) / np.log(3)


def plan():
    jobs = []
    for b in T.CM.SPEC17:
        fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
        d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "step", "cfg", "cpi", "mscs")}
        H = ent(d["cpi"], d["mscs"]); st = d["start"]; ids = np.unique(st); C = []
        for s in ids:
            h = H[st == s]; C.append(np.interp(GX, np.linspace(0, 1, len(h)), h) if len(h) > 1 else np.full_like(GX, h[0]))
        C = np.array(C); med = np.median(C, 0); dist = np.sqrt(((C - med) ** 2).mean(1))
        s = int(ids[np.argmin(dist)]); rows = np.where(st == s)[0]
        sel = rows[np.unique(np.linspace(0, len(rows) - 1, min(PER_WALK, len(rows))).round().astype(int))]
        for i in sel:
            c = T.S.project17(dict(zip(T.SEARCH, map(int, d["cfg"][i]))))
            jobs.append({"bench": b, "idx": s, "tag": f"s{int(d['step'][i]):02d}", "config": c, "step": int(d["step"][i]),
                         "walk_len": len(rows), "rms_to_median": float(dist.min()), "rms_median_walk": float(np.median(dist)),
                         "model_cpi": float(d["cpi"][i]), "model_mscs": d["mscs"][i].tolist(),
                         "insts": surrogate_insts(b, c["l1_size"])})
    return jobs


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
    Hm, Hg = ent(mc, mm), ent(gc, gm)
    from scipy.stats import spearmanr
    L.append(f"all: CPI MAPE {100*np.mean(abs(mc-gc)/gc):.1f}%, MSCS MAE {np.mean(abs(mm-gm)):.3f}, "
             f"Dispatch entropy MAE {np.mean(abs(Hm-Hg)):.3f}, Spearman {spearmanr(Hm, Hg).correlation:.2f}")
    L.append("per benchmark: walk, #states | entropy start->end model / gem5 | within-walk Spearman | CPI end/start model / gem5")
    for b in T.CM.SPEC17:
        ii = sorted([i for i, (j, _, _) in enumerate(rows) if j["bench"] == b], key=lambda i: rows[i][0]["step"])
        if not ii:
            continue
        j0 = rows[ii[0]][0]
        rho = spearmanr(Hm[ii], Hg[ii]).correlation if len(ii) > 2 and np.ptp(Hm[ii]) > 0 and np.ptp(Hg[ii]) > 0 else float("nan")
        L.append(f"  {b.split('.')[1]:10s} p{j0['idx']:<4d} {len(ii):2d} | {Hm[ii[0]]:.2f}->{Hm[ii[-1]]:.2f} / {Hg[ii[0]]:.2f}->{Hg[ii[-1]]:.2f}"
                 f" | {rho:5.2f} | {mc[ii[-1]]/mc[ii[0]]:.3f} / {gc[ii[-1]]/gc[ii[0]]:.3f}")
    up_m = sum(1 for b in T.CM.SPEC17 for ii in [[i for i, (j, _, _) in enumerate(rows) if j["bench"] == b]] if ii
               and Hm[max(ii, key=lambda i: rows[i][0]['step'])] > Hm[min(ii, key=lambda i: rows[i][0]['step'])])
    up_g = sum(1 for b in T.CM.SPEC17 for ii in [[i for i, (j, _, _) in enumerate(rows) if j["bench"] == b]] if ii
               and Hg[max(ii, key=lambda i: rows[i][0]['step'])] > Hg[min(ii, key=lambda i: rows[i][0]['step'])])
    L.append(f"entropy end > start: model {up_m}/17, gem5 {up_g}/17")
    return L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--analyze-only", action="store_true"); args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); jp = os.path.join(OUT, "plan.json")
    if os.path.exists(jp):
        jobs = json.load(open(jp))
    else:
        jobs = plan(); json.dump(jobs, open(jp, "w"), indent=1)
    if not args.analyze_only:
        reused = 0
        for j in jobs:                                   # reuse identical states from gem5_val (copy, never overwrite)
            src = os.path.join(OLD, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
            dst = os.path.join(OUT, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
            if os.path.exists(src) and not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst); reused += 1
        workers = int(os.environ.get("GEM5_WORKERS", 16))
        print(f"{len(jobs)} states, {reused} copied from gem5_val, {workers} workers", flush=True)
        t0 = time.time(); ok = 0
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(run_job, {k: j[k] for k in ("bench", "idx", "tag", "config", "insts")}, OUT) for j in jobs]
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
