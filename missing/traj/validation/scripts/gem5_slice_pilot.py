#!/usr/bin/env python3
"""gem5 pilot (CPU): split the traj surrogate-vs-gem5 bias into
  (a) model error on the windows the surrogate averages, and
  (b) windowed (training-style) vs contiguous gem5.
For each benchmark in BENCHES: one validation design (median model error among its gem5_val* runs);
  slice runs  'w<k>': training-style windows, -W 20M + k*100K, -I 100K, for the surrogate's 50 slices k
                      (same build_cmd/checkpoint as the validation runs).
For benchmarks whose precompute checkpoint is in the 18k training set (results_calibrated/diag_ckpt_in_train.txt):
  repro runs 'r<i>': two 18k training rows rerun with their own config and window -> label reproducible?
CPI recorded two ways: stats (switch_cpus_1) and stack-derived 1/(Base_commit share * W_min) (= training label).
Writes results_calibrated/gem5_slice_pilot/{runs/<bench>/<tag>/result.json, summary.txt}; never overwrites.
Usage: python calib/gem5_slice_pilot.py [--workers 30] [--dry-run]"""
import os, sys, json, glob, time, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
TRAJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, TRAJ)
sys.path.insert(0, os.path.join(os.path.dirname(TRAJ), "gem5_bounds")); sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/window_ids")
import traj_common as T
from run_gem5_bounds import build_cmd, parse_cpi, parse_last, stack16, BENCH, TIMEOUT, CPU
from make_simpoint_ids import window_ids_in_order, ARCH_START, CFG_TAGS
from dse_infer import DseEngine, load_precomputed
OUT = os.path.join(TRAJ, "results_calibrated", "gem5_slice_pilot")
BENCHES = ["638.imagick_s", "600.perlbench_s", "648.exchange2_s", "654.roms_s", "602.gcc_s"]
SEEN = {"600.perlbench_s", "648.exchange2_s", "654.roms_s", "602.gcc_s"}
WARM, SLICE = 20_000_000, 100_000
KEYS = {"fw": "fetchWidth", "dw": "decodeWidth", "rw": "renameWidth", "dpw": "dispatchWidth", "iw": "issueWidth",
        "wbw": "wbWidth", "cw": "commitWidth", "rob": "numROBEntries", "iq": "numIQEntries", "lq": "LQEntries",
        "sq": "SQEntries", "fb": "fetchBufferSize", "ialu": "IntALU", "imd": "IntMultDiv", "falu": "FP_ALU",
        "fmd": "FP_MultDiv", "rdwr": "RdWrPort", "l1": "l1_size", "l2": "l2_size", "da": "l1d_assoc",
        "ia": "l1i_assoc", "la": "l2_assoc", "bp": "bp_type"}


def s50(b, l1):
    fs = sorted(glob.glob(os.path.join(T.CM.PRECOMPUTE, f"l1_{l1}", b, "slice_*", "dse_precomputed.npy")))
    return [i for i in (int(os.path.basename(os.path.dirname(f)).split("_")[1]) for f in fs) if i % 2 == 0][:50]


def pick_design(b):
    P = T.WLPredictor(b); best = []
    for f in sorted(glob.glob(os.path.join(TRAJ, "gem5_val*", "runs", b, "*", "result.json"))):
        d = json.load(open(f))
        if d.get("error") or not d.get("cpi") or d["insts"] != 18_700_000:
            continue
        c = T.S.project17({k: d["config"][k] for k in T.SEARCH})
        if any(c[k] != d["config"][k] for k in d["config"] if k in c):
            continue
        best.append(((P(c)[0] - d["cpi"]) / d["cpi"], f, d))
    best.sort(key=lambda t: t[0]); e, f, d = best[len(best) // 2]
    return {"src": f, "config": d["config"], "gem5_cpi_contig": d["cpi"], "model_err_m50": e, "model_cpi_m50": P(T.S.project17({k: d["config"][k] for k in T.SEARCH}))[0]}


def train_rows(b, n=2):
    """two training rows of b, each with the precompute slice its window matches exactly (trace-only columns)."""
    F = np.load(f"/scratch2/wl730/micro_traindata_packed/gem/{b}.npy", mmap_mode="r"); D = T.CM.GEM_INPUT_DIM
    P = np.asarray(F[:, :D], np.float64); w = window_ids_in_order(np.asarray(F, np.float32)); W = np.unique(w)
    within = np.zeros(D)
    for k in W:
        within = np.maximum(within, P[w == k].std(0))
    trace = (within < 1e-9) & (P.std(0) > 1e-9)
    e = DseEngine.__new__(DseEngine); e.max_slices = 10**9; e.workload_names = {b}
    e.data = {64: {b: [(s, load_precomputed(os.path.join(T.CM.PRECOMPUTE, "l1_64", b, f"slice_{s}", "dse_precomputed.npy"))) for s in range(400)]}}
    eng = T.FastEngine(e)
    c1 = T.S.project17({p: (v[0]) for p, v in T.ALL_PARAMS.items() if p in T.SEARCH} | {"l1_size": 64})
    c2 = T.S.project17({p: (v[-1] if isinstance(v, list) else v[1]) for p, v in T.ALL_PARAMS.items() if p in T.SEARCH} | {"l1_size": 64})
    F1 = np.stack(eng.get_features(c1, b))[:, :D].astype(np.float64); F2 = np.stack(eng.get_features(c2, b))[:, :D].astype(np.float64)
    cols = np.where(trace & (np.abs(F1 - F2).max(0) < 1e-9))[0]; sc = np.maximum(np.abs(F1[:, cols]).max(0), 1e-12)
    out = []
    for k in np.random.RandomState(0).choice(W, n, replace=False):
        i = int(np.where(w == k)[0][0]); r = np.asarray(F[i], np.float64)
        d = np.abs(r[cols] / sc - F1[:, cols] / sc).max(1); s = int(d.argmin())
        if d[s] > 1e-5:
            continue
        cfg = {KEYS[t]: int(round(v)) for t, v in zip(CFG_TAGS, r[ARCH_START:ARCH_START + len(CFG_TAGS)])}
        base_c = r[4681 + 12]; wmin = min(cfg[x] for x in ("fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth", "issueWidth", "commitWidth"))
        out.append({"slice": s, "config": cfg, "label_cpi": float(1.0 / (base_c * wmin)), "row": i})
    return out


def run(job):
    d = os.path.join(OUT, "runs", job["bench"], job["tag"]); res = os.path.join(d, "result.json")
    if os.path.exists(res):
        return json.load(open(res))
    os.makedirs(d, exist_ok=True); fifo = os.path.join(d, "stack_fifo")
    if os.path.exists(fifo + ".txt"):
        os.remove(fifo + ".txt")
    os.mkfifo(fifo + ".txt")
    argv = build_cmd(job["bench"], job["config"], d, job["warmup"], job["insts"]) + ["-P", f"{CPU}.dump_inst_trace_path='{fifo}'"]
    t0 = time.time(); last = open(os.path.join(d, "stack_last.txt"), "w")
    tail = subprocess.Popen(["tail", "-n", "1", fifo + ".txt"], stdout=last); err = None
    try:
        with open(os.path.join(d, "run.log"), "w") as lf:
            subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, timeout=TIMEOUT, check=False, cwd=BENCH[job["bench"]].get("cwd"))
    except subprocess.TimeoutExpired:
        err = "timeout"
    try:
        tail.wait(timeout=120)
    except subprocess.TimeoutExpired:
        tail.kill(); err = err or "no_trace"
    last.close(); os.remove(fifo + ".txt")
    cpi = parse_cpi(os.path.join(d, "stats.txt")); f28 = parse_last(os.path.join(d, "stack_last.txt"))
    c = job["config"]; wmin = min(int(c.get(x, c["dispatchWidth"])) for x in ("fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth", "issueWidth", "commitWidth"))
    m = stack16(f28) if f28 else None
    cpi_stack = (sum(m[12:16]) / m[12]) / wmin if m and m[12] > 0 else None
    r = dict(job, cpi=cpi, cpi_stack=cpi_stack, mscs16=m, error=err or (None if cpi and m else "parse_failed"), elapsed=time.time() - t0)
    if r["error"] is None:
        json.dump(r, open(res, "w"), indent=1)
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=30); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(os.path.join(OUT, "summary.txt")):
        sys.exit(f"exists, refusing to overwrite: {OUT}/summary.txt")
    plan_p = os.path.join(OUT, "plan.json")
    if os.path.exists(plan_p):
        plan = json.load(open(plan_p))
    else:
        plan = {}
        for b in BENCHES:
            plan[b] = {"design": pick_design(b), "repro": train_rows(b) if b in SEEN else []}
            print(b, plan[b]["design"]["model_err_m50"], len(plan[b]["repro"]), flush=True)
        os.makedirs(OUT, exist_ok=True); json.dump(plan, open(plan_p, "w"), indent=1)
    jobs = []
    for b, p in plan.items():
        c = p["design"]["config"]
        jobs += [{"bench": b, "tag": f"w{k}", "kind": "slice", "slice": k, "config": c, "warmup": WARM + k * SLICE, "insts": SLICE} for k in s50(b, c["l1_size"])]
        jobs += [{"bench": b, "tag": f"r{i}", "kind": "repro", "slice": r["slice"], "config": r["config"], "label_cpi": r["label_cpi"],
                  "warmup": WARM + r["slice"] * SLICE, "insts": SLICE} for i, r in enumerate(p["repro"])]
    print(len(jobs), "jobs;", " ".join(build_cmd(jobs[0]["bench"], jobs[0]["config"], "<d>", jobs[0]["warmup"], jobs[0]["insts"])))
    if a.dry_run:
        return
    R = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(run, j) for j in jobs])):
            R.append(f.result())
            if i % 25 == 0:
                print(f"{i + 1}/{len(jobs)} done", flush=True)
    L = ["bench | design: gem5 contiguous 0..186 | model m50 | gem5 windowed mean over the 50 slices (stats / stack) | "
         "model err vs windowed | windowed vs contiguous | n ok"]
    for b, p in plan.items():
        ok = [r for r in R if r["bench"] == b and r["kind"] == "slice" and r["error"] is None]
        g = p["design"]["gem5_cpi_contig"]; m = p["design"]["model_cpi_m50"]
        ws = np.mean([r["cpi"] for r in ok]) if ok else np.nan; wk = np.mean([r["cpi_stack"] for r in ok]) if ok else np.nan
        L.append(f"{b:16s} | {g:.3f} | {m:.3f} ({(m - g) / g:+.1%}) | {ws:.3f} / {wk:.3f} | {(m - wk) / wk:+.1%} | {(wk - g) / g:+.1%} | {len(ok)}/50")
        for r in [r for r in R if r["bench"] == b and r["kind"] == "repro"]:
            L.append(f"    repro slice {r['slice']}: label {r['label_cpi']:.4f}  rerun stack {r.get('cpi_stack') or float('nan'):.4f}  stats {r.get('cpi') or float('nan'):.4f}  err {r['error']}")
    print("\n".join(L)); open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
