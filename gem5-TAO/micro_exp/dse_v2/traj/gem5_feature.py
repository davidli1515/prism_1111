#!/usr/bin/env python3
"""
gem5: add a feature OUTSIDE the 17 searched parameters to the end design of a greedy walk.
Question: where tuning parameters no longer helps (gcc, cactuBSSN), does a mechanism located by the stacks help?
  Base   : design already simulated in gem5_val4 (copied by reference, never re-run)
  Arms   : L1D prefetchers (all off in every earlier run) and the L1D MSHR count. Fetch buffer and the 17 parameters are unchanged.
  Control: deepsjeng, whose end design has almost no memory stalls, so little gain is predicted (written before running).
Same runner and command line as gem5_val4 (run_gem5_bounds.run_job); the only addition is the prefetcher flag.
The surrogate cannot predict these arms: results are gem5 only.
Writes feature/gem5/runs/<bench>/p<idx>_<base tag>_<arm>/result.json, feature/summary.txt; never overwrites.
  python gem5_feature.py --dry-run | (run) | --analyze-only        (GEM5_WORKERS, default 16)
"""
import os, sys, json, glob, time, argparse, shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DSE_V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DSE_V2, "gem5_bounds")); sys.path.insert(0, HERE)
import run_gem5_bounds as R

OUT = os.path.join(HERE, "feature", "gem5"); V4 = os.path.join(HERE, "gem5_val4"); V6 = os.path.join(HERE, "gem5_val6")
# The simulated system has no L2/L3 (--l2cache is never passed, see mscs_continuous_1010/inactive_l2_audit_01): L1 talks to
# DRAM directly, so an L2 prefetcher would be a no-op; every arm acts on the L1 data cache, which has 4 MSHRs by default.
MSHR16 = ["-P", "system.cpu[0].dcache.mshrs=16"]
ARMS = {"strideL1D": ["--l1d-hwp-type=StridePrefetcher"], "bopL1D": ["--l1d-hwp-type=BOPPrefetcher"],
        "impL1D": ["--l1d-hwp-type=IndirectMemoryPrefetcher"], "mshr16": MSHR16,
        "strideMshr16": ["--l1d-hwp-type=StridePrefetcher"] + MSHR16}
BASES = [("602.gcc_s", None), ("607.cactuBSSN_s", None), ("607.cactuBSSN_s", 21), ("631.deepsjeng_s", None)]   # None = end of the walk
_build = R.build_cmd


def build_with_feature(bench, config, d, warmup, insts):
    """the gem5_val4 command line plus the flag of the arm named by the run directory"""
    return _build(bench, config, d, warmup, insts) + ARMS[os.path.basename(d).rsplit("_", 1)[1]]


def jobs():
    plan = json.load(open(os.path.join(V4, "plan.json"))); out = []
    for b, step in BASES:
        rows = [j for j in plan if j["bench"] == b]
        j0 = max(rows, key=lambda j: j["step"]) if step is None else next(j for j in rows if j["step"] == step)
        for arm in ARMS:
            out.append({"bench": b, "idx": j0["idx"], "tag": f"{j0['tag']}_{arm}", "config": j0["config"], "insts": R.surrogate_insts(b, j0["config"]["l1_size"]),
                        "base": os.path.join(V4, "runs", b, f"p{j0['idx']}_{j0['tag']}"), "arm": arm, "step": j0["step"]})
    return out


def jain(c):
    c = np.maximum(c, 0); return float(c.sum() ** 2 / max(3 * (c ** 2).sum(), 1e-12))


def row(d):
    r = json.load(open(os.path.join(d, "result.json"))); x = np.array(r["mscs16"]) * r["cpi"]; st = {}
    for l in open(os.path.join(d, "stats.txt")):
        q = l.split()
        if len(q) > 1:
            st.setdefault(q[0], q[1])
    n = r["insts"] / 1000
    g = lambda k: float(st.get(k, "nan")) / n
    # stats.txt also counts the warmup CPU (switch_cpus) in cache and memory totals: read the detailed CPU's column only (switch_cpus_1)
    D = "system.cpu.dcache.overall%s::switch_cpus_1.data"; cyc = float(st["system.switch_cpus_1.numCycles"])
    return {"cpi": r["cpi"], "J": [jain(x[4 * s + 1:4 * s + 4]) for s in range(4)], "x": x, "dmiss": g(D % "Misses"), "mem": g(D % "MshrMisses"),
            "merged": g(D % "MshrHits"), "occ": float(st.get(D % "MshrMissLatency", "nan")) / 500 / cyc}   # 500 ticks per cycle; occ = demand MSHRs in use on average


def best_param_only(b):
    """lowest gem5 CPI among designs without a feature: BO/GA/random incumbents (gem5_val6) and the greedy walk (gem5_val4)"""
    best = (None, None)
    for f in glob.glob(os.path.join(V6, "runs", b, "*", "result.json")) + glob.glob(os.path.join(V4, "runs", b, "*", "result.json")):
        c = json.load(open(f)).get("cpi")
        if c and (best[0] is None or c < best[0]):
            best = (c, os.path.relpath(os.path.dirname(f), HERE))
    return best


def analyze(J):
    L = ["gem5 only. CPI, Jain per stage (Dec/Disp/Iss/Com), Issue and Commit BP cycles per inst, demand D-cache misses, misses sent to memory, misses merged into an in-flight line per kI (detailed phase only), demand MSHRs in use"]
    fmt = lambda n, r: (f"  {n:14s} CPI {r['cpi']:.3f} J {' '.join(f'{j:.2f}' for j in r['J'])} BP Iss {r['x'][10]:.2f} Com {r['x'][14]:.2f} "
                        f"FP Disp {r['x'][5]:.2f} dmiss {r['dmiss']:.1f} mem {r['mem']:.1f} merged {r['merged']:.1f} occ {r['occ']:.2f}")
    done = set()
    for j in J:
        k = (j["bench"], j["step"])
        if k not in done:
            done.add(k); b0 = row(j["base"]); bp = best_param_only(j["bench"])
            L += [f"{j['bench']} base = walk step {j['step']}; best parameter-only design on gem5: CPI {bp[0]:.3f} ({bp[1]})", fmt("no feature", b0)]
        d = os.path.join(OUT, "runs", j["bench"], f"p{j['idx']}_{j['tag']}")
        if os.path.exists(os.path.join(d, "result.json")):
            r = row(d); b0 = row(j["base"])
            L.append(fmt(j["arm"], r) + f" | {100 * (r['cpi'] / b0['cpi'] - 1):+.1f}%")
        else:
            L.append(f"  {j['arm']:14s} no result")
    return L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args(); J = jobs(); R.build_cmd = build_with_feature
    if a.dry_run:
        print(f"{len(J)} gem5 runs"); [print(f"  {j['bench']:16s} step {j['step']:2d} {j['arm']:10s} insts {j['insts']}") for j in J]
        d = os.path.join(OUT, "runs", J[0]["bench"], f"p{J[0]['idx']}_{J[0]['tag']}")
        print("example command:\n" + shlex.join(R.build_cmd(J[0]["bench"], J[0]["config"], d, R.WARMUP, J[0]["insts"]))); return
    if not a.analyze_only:
        print(f"{len(J)} gem5 runs", flush=True)
        with ThreadPoolExecutor(int(os.environ.get("GEM5_WORKERS", 16))) as ex:
            for f in as_completed([ex.submit(R.run_job, j, OUT) for j in J]):
                r = f.result(); print(f"  {r['bench']} {r['tag']} {'skip' if r.get('skipped') else r.get('error') or 'ok'} CPI {r.get('cpi')}", flush=True)
    L = analyze(J); sp = os.path.join(HERE, "feature", "summary.txt")
    if os.path.exists(sp):
        sp = sp.replace(".txt", f".{os.getpid()}.txt")
    open(sp, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
