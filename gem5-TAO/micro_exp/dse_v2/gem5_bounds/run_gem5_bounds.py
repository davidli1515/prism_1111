#!/usr/bin/env python3
"""
gem5 validation of multi-stage CPI-stack bounds (Eyerman et al., CAL'18, Fig. 2 style):
does the PRISM-predicted stack bound the real (gem5) what-if gain as well as the gem5-measured
stack does?  -> can the predicted MSCS replace the simulator's MSCS for what-if analysis?

Jobs: for each benchmark, P points of the pure-BO front (taken from mscs_guide/results/
bounds_<bench>.json, which already holds PRISM stacks for them) x {base, maxed config of each
direction FE / L1 / BE / SP / ALL (check_bounds.maxed)}.  Every run is one gem5 simulation
from the existing checkpoint, same command line as dse_exp/validate_joint_dse.py (imported, not
modified), plus:
  -P system.switch_cpus_1[0].dump_inst_trace_path=<run>/stack_fifo
     gem5 writes one line per committed instruction ending in the 28 cumulative stack fields;
     the stream goes into a named pipe read by `tail -n 1` -> only the last line is kept
     (no multi-GB trace on disk).
  -W 20M, -I = (max slice id used by the surrogate + 1) x 100K, i.e. the same instruction range
     as the 50 slices the surrogate averages (dse_infer: lexicographic, even ids, first 50).
Per run writes <out>/runs/<bench>/p<idx>_<tag>/{stats.txt, run.log, stack_last.txt, result.json};
runs with an existing result.json are skipped (resumable, never overwritten).

Usage:
  python run_gem5_bounds.py --dry-run                      # list jobs + one command, no gem5
  python run_gem5_bounds.py --workers 16                   # run everything
  python run_gem5_bounds.py --benches 641.leela_s --points 1 --workers 2   # small pilot
"""
import os, sys, json, glob, time, argparse, subprocess, shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DSE_V2 = os.path.dirname(HERE)
MICRO = os.path.dirname(DSE_V2)
sys.path.insert(0, os.path.join(DSE_V2, "mscs_guide"))
sys.path.insert(0, DSE_V2)
sys.path.insert(0, os.path.join(MICRO, "dse_exp"))
from validate_joint_dse import BENCH, build_cmd, parse_cpi, TIMEOUT          # reused as is
from check_bounds import DIRS, maxed                                           # reused as is
from common import config_key

BENCHES = ["638.imagick_s", "623.xalancbmk_s", "641.leela_s", "648.exchange2_s", "602.gcc_s", "605.mcf_s"]
PRECOMPUTE = os.path.join(MICRO, "dse_exp", "precompute")
WARMUP = 20_000_000
SLICE = 100_000
CPU = "system.switch_cpus_1[0]"


def surrogate_insts(bench, l1):
    """Instructions after warm-up that cover the slices the surrogate uses for this config."""
    fs = sorted(glob.glob(os.path.join(PRECOMPUTE, f"l1_{l1}", bench, "slice_*", "dse_precomputed.npy")))
    ids = [int(os.path.basename(os.path.dirname(f)).split("_")[1]) for f in fs]
    ids = [i for i in ids if i % 2 == 0][:50]
    return (max(ids) + 1) * SLICE


def stack16(fields):
    """28 cumulative gem5 fields -> 16-dim MSCS, exactly as process_test_result.py."""
    a = fields
    dec = a[0:4]
    disp = [a[4], a[5], a[6] + a[7] + a[8], a[9]]
    iss = [a[16], a[17], a[18] + a[19] + a[20], a[21]]
    com = [a[22], a[23], a[24] + a[25] + a[26], a[27]]
    return dec + disp + iss + com


def parse_last(path):
    if not os.path.isfile(path):
        return None
    line = open(path, errors="replace").read().strip()
    i = line.find("START")
    if i < 0:
        return None
    parts = [x for x in line[i:].split(",")[1:] if x.strip()]
    if len(parts) != 28:
        return None
    return [float(x) for x in parts]


def make_jobs(args):
    jobs = []
    for b in args.benches:
        z = json.load(open(os.path.join(DSE_V2, "mscs_guide", "results", f"bounds_{b}.json")))
        src = json.load(open(z["src"]))
        pts = sorted([p for p in z["points"] if p["front"] == "bo500"], key=lambda p: p["area"])
        pick = [pts[i] for i in np.unique(np.linspace(0, len(pts) - 1, min(args.points, len(pts))).round().astype(int))]
        for p in pick:
            c = src["configs"][p["idx"]]
            todo = [("base", c)]
            for d in DIRS:
                for k, cm in enumerate(maxed(c, d)):
                    todo.append((f"{d}{k}", cm))
            seen = set()
            for tag, cfg in todo:
                key = config_key(cfg)
                if key in seen:                      # e.g. ALL0 == BE0 when FE/L1 already maxed
                    continue
                seen.add(key)
                jobs.append({"bench": b, "idx": p["idx"], "tag": tag, "config": cfg,
                             "insts": surrogate_insts(b, cfg["l1_size"])})
    return jobs


def run_job(job, out_root):
    d = os.path.abspath(os.path.join(out_root, "runs", job["bench"], f"p{job['idx']}_{job['tag']}"))
    res_path = os.path.join(d, "result.json")
    if os.path.exists(res_path):
        return json.load(open(res_path)) | {"skipped": True}
    os.makedirs(d, exist_ok=True)
    fifo = os.path.join(d, "stack_fifo")
    if os.path.exists(fifo + ".txt"):
        os.remove(fifo + ".txt")
    os.mkfifo(fifo + ".txt")
    argv = build_cmd(job["bench"], job["config"], d, WARMUP, job["insts"])
    argv += ["-P", f"{CPU}.dump_inst_trace_path='{fifo}'"]
    t0 = time.time()
    last = open(os.path.join(d, "stack_last.txt"), "w")
    tail = subprocess.Popen(["tail", "-n", "1", fifo + ".txt"], stdout=last)
    err = None
    try:
        with open(os.path.join(d, "run.log"), "w") as lf:
            subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, timeout=TIMEOUT, check=False,
                           cwd=BENCH[job["bench"]].get("cwd"))
    except subprocess.TimeoutExpired:
        err = "timeout"
    try:
        tail.wait(timeout=120)
    except subprocess.TimeoutExpired:          # gem5 never opened the pipe
        tail.kill(); err = err or "no_trace"
    last.close()
    os.remove(fifo + ".txt")
    cpi = parse_cpi(os.path.join(d, "stats.txt"))
    f28 = parse_last(os.path.join(d, "stack_last.txt"))
    r = {"bench": job["bench"], "idx": job["idx"], "tag": job["tag"], "config": job["config"],
         "insts": job["insts"], "cpi": cpi, "fields28": f28, "mscs16": stack16(f28) if f28 else None,
         "error": err or (None if cpi and f28 else "parse_failed"), "elapsed": time.time() - t0}
    if r["error"] is None:                     # only successful runs are frozen
        json.dump(r, open(res_path, "w"), indent=1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", nargs="+", default=BENCHES)
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = make_jobs(args)
    by = {}
    for j in jobs:
        by.setdefault(j["bench"], []).append(j)
    print(f"{len(jobs)} gem5 runs: " + ", ".join(f"{b} {len(v)}" for b, v in by.items()))
    print(f"insts per run: {sorted({j['insts'] for j in jobs})}  (+{WARMUP:,} warm-up)")
    if args.dry_run:
        j = jobs[0]
        print("example command:\n  " + " ".join(shlex.quote(x) for x in
              build_cmd(j["bench"], j["config"], "<run_dir>", WARMUP, j["insts"])
              + ["-P", f"{CPU}.dump_inst_trace_path=<run_dir>/stack_fifo"]))
        return
    t0 = time.time(); n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j, args.out) for j in jobs]
        for f in as_completed(futs):
            r = f.result(); n_ok += r["error"] is None
            st = "skip" if r.get("skipped") else (f"CPI={r['cpi']:.4f}" if r["error"] is None else r["error"])
            print(f"  {r['bench']:16s} p{r['idx']}_{r['tag']:5s} {st} [{r['elapsed']:.0f}s] "
                  f"ok {n_ok}/{len(jobs)} wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
