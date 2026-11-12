#!/usr/bin/env python3
"""
gem5 check of model trajectories (random-start runs in results/). For 8 benchmarks covering the route types
(window: imagick, wrf; cache: xalancbmk, mcf; front-end/width: exchange2; migrating: deepsjeng, cactuBSSN;
anchored: gcc) take 2 walks: the one with the largest Dispatch-entropy increase and the one with the median
increase; up to 12 states per walk evenly spaced along it (first and last always kept). Every state is run in
gem5 with the same command line as gem5_bounds (stack read through a FIFO; instruction window = the 50
surrogate slices). Writes gem5_val/runs/<bench>/p<start>_s<step>/result.json (existing results skipped),
gem5_val/plan.json, and at the end gem5_val/summary.txt via analyze().
"""
import os, sys, json, glob, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DSE_V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DSE_V2, "gem5_bounds")); sys.path.insert(0, HERE)
import traj_common as T
from run_gem5_bounds import run_job, surrogate_insts
from scipy.stats import spearmanr

OUT = os.path.join(HERE, "gem5_val"); R = os.path.join(HERE, "results")
BENCHES = ["638.imagick_s", "621.wrf_s", "623.xalancbmk_s", "605.mcf_s", "648.exchange2_s",
           "631.deepsjeng_s", "607.cactuBSSN_s", "602.gcc_s"]
PER_WALK = 12


def load(b):
    fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
    return {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "step", "cfg", "cpi", "mscs")}


def ent(cpi, m):
    c = np.stack([cpi * m[:, 4 + k] for k in (1, 2, 3)], 1)
    p = c / np.maximum(c.sum(1, keepdims=True), 1e-12)
    return -(p * np.log(p + 1e-12)).sum(1) / np.log(3)


def feat6(cpi, m):
    X = np.stack([cpi * m[:, 4 * s + k] for s in (1, 3) for k in (1, 2, 3)], 1)
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def plan():
    jobs = []
    for b in BENCHES:
        d = load(b); H = ent(d["cpi"], d["mscs"]); st = d["start"]
        ids = np.unique(st); dH = np.array([H[st == s][-1] - H[st == s][0] for s in ids])
        pick = {"max_dH": int(ids[np.argmax(dH)]), "median_dH": int(ids[np.argsort(dH)[len(dH) // 2]])}
        for why, s in pick.items():
            rows = np.where(st == s)[0]
            sel = rows[np.unique(np.linspace(0, len(rows) - 1, min(PER_WALK, len(rows))).round().astype(int))]
            for i in sel:
                c = T.S.project17(dict(zip(T.SEARCH, map(int, d["cfg"][i]))))
                jobs.append({"bench": b, "idx": s, "tag": f"s{int(d['step'][i]):02d}", "config": c, "why": why,
                             "step": int(d["step"][i]), "model_cpi": float(d["cpi"][i]),
                             "model_mscs": d["mscs"][i].tolist(), "insts": surrogate_insts(b, c["l1_size"])})
    return jobs


def analyze(jobs):
    Z = np.load(os.path.join(R, "state_clusters_v1.npz")); cent = Z["centroids"]
    rows = []
    for j in jobs:
        p = os.path.join(OUT, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
        if os.path.exists(p):
            r = json.load(open(p)); rows.append((j, r["cpi"], np.array(r["mscs16"])))
    lines = [f"{len(rows)}/{len(jobs)} gem5 runs available"]
    if not rows:
        return lines
    mc = np.array([j["model_cpi"] for j, _, _ in rows]); gc = np.array([c for _, c, _ in rows])
    mm = np.array([j["model_mscs"] for j, _, _ in rows]); gm = np.array([m for _, _, m in rows])
    Hm, Hg = ent(mc, mm), ent(gc, gm)
    lm = np.argmin(((feat6(mc, mm)[:, None] - cent[None]) ** 2).sum(-1), 1)
    lg = np.argmin(((feat6(gc, gm)[:, None] - cent[None]) ** 2).sum(-1), 1)
    lines += [f"CPI MAPE {100*np.mean(abs(mc-gc)/gc):.1f}%   MSCS share MAE {np.mean(abs(mm-gm)):.3f}",
              f"Dispatch entropy: Spearman(model, gem5) {spearmanr(Hm, Hg).correlation:.2f}, MAE {np.mean(abs(Hm-Hg)):.3f}",
              f"bottleneck-state label agreement (nearest centroid) {np.mean(lm == lg):.2f}"]
    lines.append("\nper walk: bench, pick, #states | entropy start->end model / gem5 | state sequence model / gem5 | CPI end/start model / gem5")
    for b in BENCHES:
        for why in ("max_dH", "median_dH"):
            ii = [i for i, (j, _, _) in enumerate(rows) if j["bench"] == b and j["why"] == why]
            if not ii:
                continue
            ii = sorted(ii, key=lambda i: rows[i][0]["step"])
            col = lambda lab: "→".join(str(x) for k_, x in enumerate(lab) if k_ == 0 or x != lab[k_ - 1])
            lines.append(f"  {b.split('.')[1]:10s} {why:9s} {len(ii):2d} | {Hm[ii[0]]:.2f}->{Hm[ii[-1]]:.2f} / {Hg[ii[0]]:.2f}->{Hg[ii[-1]]:.2f}"
                         f" | S{col(lm[ii])} / S{col(lg[ii])} | {mc[ii[-1]]/mc[ii[0]]:.3f} / {gc[ii[-1]]/gc[ii[0]]:.3f}")
    return lines


def main():
    os.makedirs(OUT, exist_ok=True)
    jp = os.path.join(OUT, "plan.json")
    jobs = json.load(open(jp)) if os.path.exists(jp) else plan()
    if not os.path.exists(jp):
        json.dump(jobs, open(jp, "w"), indent=1)
    workers = int(os.environ.get("GEM5_WORKERS", 12))
    print(f"{len(jobs)} gem5 runs, {workers} workers", flush=True)
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
    lines = analyze(jobs); open(sp, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
