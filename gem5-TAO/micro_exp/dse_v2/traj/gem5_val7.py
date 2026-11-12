#!/usr/bin/env python3
"""
gem5 round 7: is a Dispatch stall dominated by front-end (FP) cycles removed by front-end or by back-end resources?
All changes stay inside the 17 searched parameters (L2, branch predictor, fetch buffer untouched).
  Design (one per benchmark): from the representative small-start greedy walk already simulated in gem5_val4, the
           states whose gem5 Dispatch stall is FP-dominant (FP > BP and FP > SP cycles); among them the one closest
           to the middle of the walk (if none is FP-dominant on gem5: the model's FP-dominant state, flagged).
  Arms (sweep scale, decided before running):
    FE_max  : front end maxed: fetchWidth = decodeWidth = 12, l1i_assoc = largest allowed (generous to the FE side)
    FE_best : best single front-end parameter (fetchWidth / decodeWidth / l1i_assoc, any value above the current
              one) by model CPI, area increase <= that of FE_max
    BE_best : best single back-end parameter (ROB, IQ, LQ, SQ, FUs, RdWrPort, l1d_assoc, wbWidth, commitWidth, any
              value above the current one) by model CPI, same area budget as FE_best
    (dispatchWidth and l1_size feed both sides and are not used.)
  base = the gem5_val4 result of the design (copied, never re-run).
Writes gem5_val7/ (plan.json, runs/, summary.txt); never overwrites.
  python gem5_val7.py --dry-run | (run) | --analyze-only        (GEM5_WORKERS, default 16)
"""
import os, sys, json, time, shutil, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DSE_V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DSE_V2, "gem5_bounds")); sys.path.insert(0, HERE)
import traj_common as T
from run_gem5_bounds import run_job, surrogate_insts
from area17 import FastArea

OUT, V4 = os.path.join(HERE, "gem5_val7"), os.path.join(HERE, "gem5_val4")
FE = ["fetchWidth", "decodeWidth", "l1i_assoc"]
BE = ["commitWidth", "numROBEntries", "numIQEntries", "LQEntries", "SQEntries", "IntALU", "IntMultDiv", "FP_ALU",
      "FP_MultDiv", "RdWrPort", "l1d_assoc", "wbWidth"]
ARMS = ["FE_max", "FE_best", "BE_best"]


def disp(cpi, m):
    """Dispatch stall cycles per instruction (FP, BP, SP)"""
    return float(cpi) * np.asarray(m)[[5, 6, 7]]


def values_above(p, v):
    spec = T.ALL_PARAMS[p]
    if isinstance(spec, list):
        return [x for x in spec if x > v]
    lo, hi, s = spec
    return list(range(v + s, hi + 1, s))


def v4_path(j):
    return os.path.join(V4, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")


def result_path(j, arm):
    return os.path.join(OUT, "runs", j["bench"], f"p{j['idx']}_{arm}", "result.json")


def plan():
    area = FastArea()
    v4 = json.load(open(os.path.join(V4, "plan.json"))); jobs = []
    for b in T.CM.SPEC17:
        rows = [j for j in v4 if j["bench"] == b and os.path.exists(v4_path(j))]
        if not rows:
            print(f"  {b}: no gem5_val4 state -> skipped"); continue
        g = [disp(*(lambda r: (r["cpi"], r["mscs16"]))(json.load(open(v4_path(j))))) for j in rows]
        m = [disp(j["model_cpi"], j["model_mscs"]) for j in rows]
        cand = [i for i in range(len(rows)) if g[i][0] > g[i][1:].max()]; src = "gem5"
        if not cand:
            cand = [i for i in range(len(rows)) if m[i][0] > m[i][1:].max()]; src = "model"
        if not cand:
            print(f"  {b}: no FP-dominant Dispatch state -> skipped"); continue
        i = min(cand, key=lambda i: abs(rows[i]["step"] / max(rows[i]["walk_len"] - 1, 1) - 0.5)); j0 = rows[i]
        c0 = T.S.project17(dict(j0["config"])); a0 = area(c0)
        P = T.WLPredictor(b); cpi0, _ = P(c0)
        fe = dict(c0, fetchWidth=12, decodeWidth=12, l1i_assoc=max(T.ALL_PARAMS["l1i_assoc"]))
        arms = {"FE_max": T.S.project17(fe)}
        budget = area(arms["FE_max"]) - a0
        for name, grp in (("FE_best", FE), ("BE_best", BE)):
            best = None
            for p in grp:
                for v in values_above(p, c0[p]):
                    c = T.S.project17(dict(c0, **{p: v}))
                    if T.ckey(c) == T.ckey(c0) or area(c) - a0 > budget:
                        continue
                    cp, _ = P(c)
                    if best is None or cp < best[0]:
                        best = (cp, c, p, v)
            arms[name] = best[1] if best else None
            if best:
                print(f"  {b:16s} {name}: {best[2]} {c0[best[2]]}->{best[3]}  model CPI {cpi0:.3f}->{best[0]:.3f}")
        for arm, c in arms.items():
            if c is None:
                continue
            cp, ms = P(c)
            jobs.append({"bench": b, "idx": j0["idx"], "base_tag": j0["tag"], "base_step": j0["step"],
                         "walk_len": j0["walk_len"], "select": src, "arm": arm, "tag": arm, "config": c,
                         "changed": {k: [c0[k], c[k]] for k in T.SEARCH if c[k] != c0[k]},
                         "area0": a0, "area": float(area(c)), "budget": float(budget),
                         "model_cpi0": float(cpi0), "model_mscs0": j0["model_mscs"],
                         "model_cpi": float(cp), "model_mscs": np.asarray(ms).tolist(),
                         "insts": surrogate_insts(b, c["l1_size"])})
    return jobs


def analyze(jobs):
    L = []; R = {}
    for j in jobs:
        bp, p = v4_path({"bench": j["bench"], "idx": j["idx"], "tag": j["base_tag"]}), result_path(j, j["arm"])
        if os.path.exists(bp) and os.path.exists(p):
            r0, r = json.load(open(bp)), json.load(open(p))
            R.setdefault(j["bench"], {})[j["arm"]] = (j, r0, r)
    L.append(f"{sum(len(v) for v in R.values())}/{len(jobs)} arm runs with gem5 results, {len(R)} benchmarks")
    L.append("per benchmark: CPI change % and Dispatch FP cycles removed (% of base FP), model | gem5")
    W = {"model": [], "gem5": []}
    for b, d in R.items():
        s = f"  {b:16s}"
        for arm in ARMS:
            if arm not in d:
                s += f" | {arm} n/a"; continue
            j, r0, r = d[arm]
            mc = 100 * (j["model_cpi"] / j["model_cpi0"] - 1); gc = 100 * (r["cpi"] / r0["cpi"] - 1)
            mf = disp(j["model_cpi0"], j["model_mscs0"])[0]; mf1 = disp(j["model_cpi"], j["model_mscs"])[0]
            gf = disp(r0["cpi"], r0["mscs16"])[0]; gf1 = disp(r["cpi"], r["mscs16"])[0]
            s += f" | {arm} {mc:+5.1f}/{gc:+5.1f}% FP-{100*(1-mf1/mf):3.0f}/{100*(1-gf1/gf):3.0f}%"
        if "BE_best" in d and "FE_max" in d:
            for k, (x, y) in (("model", ("model_cpi", None)), ("gem5", (None, "cpi"))):
                be = d["BE_best"][0]["model_cpi"] if k == "model" else d["BE_best"][2]["cpi"]
                fm = d["FE_max"][0]["model_cpi"] if k == "model" else d["FE_max"][2]["cpi"]
                W[k].append(be < fm)
        L.append(s + f"  [{d[ARMS[0]][0]['select'] if ARMS[0] in d else '?'}]")
    for k in W:
        L.append(f"BE_best (area <= FE_max) beats FE_max on {k}: {sum(W[k])}/{len(W[k])}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    jp = os.path.join(OUT, "plan.json")
    jobs = json.load(open(jp)) if os.path.exists(jp) else plan()
    if args.dry_run:
        print(f"{len(jobs)} gem5 runs over {len({j['bench'] for j in jobs})} benchmarks "
              f"(selected on gem5: {len({j['bench'] for j in jobs if j['select'] == 'gem5'})})")
        for j in jobs:
            print(f"  {j['bench']:16s} {j['arm']:7s} step {j['base_step']}/{j['walk_len']-1} area +{100*(j['area']/j['area0']-1):5.1f}% "
                  f"model CPI {100*(j['model_cpi']/j['model_cpi0']-1):+5.1f}%  {j['changed']}")
        return
    if not os.path.exists(jp):
        os.makedirs(OUT, exist_ok=True); json.dump(jobs, open(jp, "w"), indent=1)
    if not args.analyze_only:
        todo = [j for j in jobs if not os.path.exists(result_path(j, j["arm"]))]
        workers = int(os.environ.get("GEM5_WORKERS", 16))
        print(f"{len(todo)} gem5 runs, {workers} workers", flush=True)
        t0 = time.time(); ok = 0
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(run_job, {"bench": j["bench"], "idx": j["idx"], "tag": j["arm"], "config": j["config"],
                                        "insts": j["insts"]}, OUT) for j in todo]
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
