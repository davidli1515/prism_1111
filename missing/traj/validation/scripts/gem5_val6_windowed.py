#!/usr/bin/env python3
"""Windowed (training-label protocol) gem5 validation of every gem5_val6 design (BO / GA / RW trajectories).
Per design: 10 slices = every 5th of the surrogate's 50 slices for its L1 size; each slice is one training-style gem5
run (-W 20M + k*100K, -I 100K, same build_cmd/checkpoint as the validation), CPI = cycles per uop from the stack.
Jobs are interleaved round-robin over benchmarks (one design at a time per benchmark) so every benchmark gets
results early.  Runs: results_calibrated/gem5_val6_windowed/runs/<bench>/<design>/w<k>/result.json (resumable).
  python calib/gem5_val6_windowed.py --workers 24        # run
  python calib/gem5_val6_windowed.py --summary           # model vs windowed gem5 on finished designs
Summary file: summary_<n designs>.txt (never overwritten)."""
import os, sys, json, glob, time, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TRAJ = os.path.dirname(HERE); sys.path.insert(0, TRAJ); sys.path.insert(0, HERE)
import gem5_slice_pilot as GP
T = GP.T
OUT = os.path.join(TRAJ, "results_calibrated", "gem5_val6_windowed"); GP.OUT = OUT
WARM, SLICE = 20_000_000, 100_000


def designs():
    D = {}
    for f in sorted(glob.glob(os.path.join(TRAJ, "gem5_val6", "runs", "*", "*", "result.json"))):
        d = json.load(open(f))
        if d.get("error") or not d.get("cpi"):
            continue
        k = (d["bench"], T.ckey(d["config"]), d["insts"])
        if k in D:
            continue
        m = d["mscs16"]; c = d["config"]
        w = min(int(c.get(x, c["dispatchWidth"])) for x in ("fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth", "issueWidth", "commitWidth"))
        D[k] = {"bench": d["bench"], "name": os.path.basename(os.path.dirname(f)), "src": f, "config": c, "insts": d["insts"],
                "gem5_cpi": d["cpi"], "gem5_cpuop": sum(m[12:16]) / m[12] / w}
    return list(D.values())


def run_nolog(job):
    """GP.run without run.log: gem5 stdout/stderr are discarded."""
    d = os.path.join(OUT, "runs", job["bench"], job["tag"]); res = os.path.join(d, "result.json")
    if os.path.exists(res):
        return json.load(open(res))
    os.makedirs(d, exist_ok=True); fifo = os.path.join(d, "stack_fifo")
    if os.path.exists(fifo + ".txt"):
        os.remove(fifo + ".txt")
    os.mkfifo(fifo + ".txt")
    argv = GP.build_cmd(job["bench"], job["config"], d, job["warmup"], job["insts"]) + ["-P", f"{GP.CPU}.dump_inst_trace_path='{fifo}'"]
    t0 = time.time(); last = open(os.path.join(d, "stack_last.txt"), "w")
    tail = subprocess.Popen(["tail", "-n", "1", fifo + ".txt"], stdout=last); err = None
    try:
        subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=GP.TIMEOUT, check=False, cwd=GP.BENCH[job["bench"]].get("cwd"))
    except subprocess.TimeoutExpired:
        err = "timeout"
    try:
        tail.wait(timeout=120)
    except subprocess.TimeoutExpired:
        tail.kill(); err = err or "no_trace"
    last.close(); os.remove(fifo + ".txt")
    cpi = GP.parse_cpi(os.path.join(d, "stats.txt")); f28 = GP.parse_last(os.path.join(d, "stack_last.txt"))
    c = job["config"]; wmin = min(int(c.get(x, c["dispatchWidth"])) for x in ("fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth", "issueWidth", "commitWidth"))
    m = GP.stack16(f28) if f28 else None
    cpi_stack = (sum(m[12:16]) / m[12]) / wmin if m and m[12] > 0 else None
    r = dict(job, cpi=cpi, cpi_stack=cpi_stack, mscs16=m, error=err or (None if cpi and m else "parse_failed"), elapsed=time.time() - t0)
    if r["error"] is None:
        json.dump(r, open(res, "w"), indent=1)
    return r


def slices(b, l1):
    return GP.s50(b, l1)[::5]


def jobs(Ds):
    per = {}
    for d in Ds:
        per.setdefault(d["bench"], []).append(d)
    J, i = [], 0
    while any(per.values()):
        for b in sorted(per):
            if per[b]:
                d = per[b].pop(0)
                J += [{"bench": b, "tag": f"{d['name']}/w{k}", "kind": "slice", "slice": k, "config": d["config"],
                       "warmup": WARM + k * SLICE, "insts": SLICE} for k in slices(b, d["config"]["l1_size"])]
    return J


def summary():
    import torch, diag_slice_coverage as DS
    Ds = designs(); done = []
    for d in Ds:
        R = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "runs", d["bench"], d["name"], "w*", "result.json"))]
        if len(R) == len(slices(d["bench"], d["config"]["l1_size"])):
            done.append((d, {r["slice"]: r["cpi_stack"] for r in R}))
    p = os.path.join(OUT, f"summary_{len(done)}.txt")
    if os.path.exists(p):
        sys.exit(f"exists, refusing to overwrite: {p}")
    model = T.GEMModel(T.CM.GEM_INPUT_DIM, [512, 256], 0.0); model.load_state_dict(torch.load(T.CM.DEFAULT_CKPT, map_location="cpu")); model.eval()
    norm = T.CM.get_normalizer("18k"); rows = []
    for b in sorted({d["bench"] for d, _ in done}):
        sub = [(d, g) for d, g in done if d["bench"] == b]
        need = {}
        for d, _ in sub:
            need.setdefault(d["config"]["l1_size"], set()).update(slices(b, d["config"]["l1_size"]))
        eng = DS.engine_multi(b, {l1: sorted(s) for l1, s in need.items()}); P = T.WLPredictor(b)
        for d, g in sub:
            c = T.S.project17({k: d["config"][k] for k in T.SEARCH}); l1 = c["l1_size"]; sids = sorted(need[l1])
            with torch.no_grad():
                X = np.stack([f[:T.CM.GEM_INPUT_DIM] for f in eng.get_features(c, b)])
                m = model(torch.from_numpy(norm.transform(X)).float())
                cs = torch.clamp(1.0 / (torch.clamp(m[:, T.CM.BASE_IDX], min=0.01) * float(T.compute_w_min(c)) + 1e-8), max=100.0).mean(1).numpy()
            mp = {s: cs[i] for i, s in enumerate(sids)}; ks = sorted(g)
            ew = np.array([(mp[k] - g[k]) / g[k] for k in ks]); m50 = P(c)[0]
            rows.append((b, np.abs(ew).mean(), ew.mean(), (np.mean([mp[k] for k in ks]) - np.mean([g[k] for k in ks])) / np.mean([g[k] for k in ks]),
                         (m50 - d["gem5_cpuop"]) / d["gem5_cpuop"], (m50 - d["gem5_cpi"]) / d["gem5_cpi"], *np.abs(ew)))
    L = [f"gem5_val6 designs with all 10 windows finished: {len(done)} / {len(Ds)}; model = surrogate checkpoint, cycles per uop",
         "bench            designs | per-window MAPE (signed) | design-level |err| | traj-style: m50 vs contiguous cycles/uop | m50 vs contiguous cycles/inst"]
    for b in sorted({r[0] for r in rows}):
        R = [r for r in rows if r[0] == b]; pw = np.concatenate([r[6:] for r in R])
        L.append(f"{b:16s} {len(R):4d} | {pw.mean():5.1%} ({np.mean([r[2] for r in R]):+6.1%}) | {np.mean([abs(r[3]) for r in R]):5.1%} | "
                 f"{np.mean([abs(r[4]) for r in R]):5.1%} ({np.mean([r[4] for r in R]):+6.1%}) | {np.mean([abs(r[5]) for r in R]):5.1%} ({np.mean([r[5] for r in R]):+6.1%})")
    unseen = {"526.blender_r", "623.xalancbmk_s", "631.deepsjeng_s", "638.imagick_s"}
    for name, sel in (("ALL", lambda r: True), ("seen checkpoint", lambda r: r[0] not in unseen), ("unseen checkpoint", lambda r: r[0] in unseen)):
        R = [r for r in rows if sel(r)]
        if not R:
            continue
        pw = np.concatenate([r[6:] for r in R])
        L.append(f"{name:18s} {len(R):4d} | per-window MAPE {pw.mean():5.1%}, p90 {np.percentile(pw, 90):5.1%} | design-level {np.mean([abs(r[3]) for r in R]):5.1%} | "
                 f"vs contiguous cycles/uop {np.mean([abs(r[4]) for r in R]):5.1%} | vs contiguous cycles/inst {np.mean([abs(r[5]) for r in R]):5.1%}")
    print("\n".join(L)); open(p, "w").write("\n".join(L) + "\n"); print("wrote", p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=24); ap.add_argument("--summary", action="store_true")
    ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    if a.summary:
        return summary()
    Ds = designs(); J = jobs(Ds)
    print(f"{len(Ds)} designs, {len(J)} windowed runs, {a.workers} workers", flush=True)
    if a.dry_run:
        return
    os.makedirs(OUT, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(run_nolog, j) for j in J])):
            r = f.result()
            if (i + 1) % 50 == 0 or r.get("error"):
                print(f"{i + 1}/{len(J)} {r['bench']} {r['tag']} err={r.get('error')}", flush=True)


if __name__ == "__main__":
    main()
