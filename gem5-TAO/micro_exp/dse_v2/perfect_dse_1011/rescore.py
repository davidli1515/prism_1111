#!/usr/bin/env python3
"""
Step 1 of the first look: collect every configuration already evaluated by earlier DSE runs
(read only), and store its stage-resolved CPI stack per workload.  No new search, no McPAT
(areas are taken from the archives).

Sources (single benchmark): results_v4_1007/<b>, results_v3_single_1006/<b>,
                            results_v3_single_nodecouple_1006/<b>
Sources (joint 17):         results_v4_1007/joint_17, results_v4_1007_noprotect/joint_17,
                            results_v3_1006/*.json, results_v2_1005/*_s42.json
Output: stacks/<tag>[_cK-of-N].npz  (refuses to overwrite)

Usage: python rescore.py --bench 638.imagick_s
       python rescore.py --joint --chunk 0 --n-chunks 12
"""
import os, sys, json, glob, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stack_eval import StackEval, DSE_V2, config_key
from common import SPEC17


def sources(bench):
    R = lambda *p: os.path.join(DSE_V2, *p)
    if bench == "joint_17":
        fs = (glob.glob(R("results_v4_1007", "joint_17", "*.json"))
              + glob.glob(R("results_v4_1007_noprotect", "joint_17", "*.json"))
              + glob.glob(R("results_v3_1006", "*.json"))
              + [f for f in glob.glob(R("results_v2_1005", "*_s42.json")) if "pareto" not in f])
    else:
        fs = []
        for d in ("results_v4_1007", "results_v3_single_1006", "results_v3_single_nodecouple_1006"):
            fs += glob.glob(R(d, bench, "*.json"))
    return sorted(fs)


def collect(bench):
    seen, cfgs, areas, src = set(), [], [], []
    for f in sources(bench):
        d = json.load(open(f))
        if "configs" not in d:
            continue
        for c, a in zip(d["configs"], d["areas"]):
            k = config_key(c)
            if k in seen or a is None:
                continue
            seen.add(k); cfgs.append(c); areas.append(a); src.append(os.path.relpath(f, DSE_V2))
    return cfgs, areas, src


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bench", choices=SPEC17)
    g.add_argument("--joint", action="store_true")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    args = ap.parse_args()
    tag = "joint_17" if args.joint else args.bench
    wls = SPEC17 if args.joint else [args.bench]
    os.makedirs(os.path.join(HERE, "stacks"), exist_ok=True)
    out = os.path.join(HERE, "stacks", tag + (f"_c{args.chunk}-of-{args.n_chunks}" if args.n_chunks > 1 else "") + ".npz")
    if os.path.exists(out):
        sys.exit(f"exists, refusing to overwrite: {out}")
    t0 = time.time()
    cfgs, areas, src = collect(tag)
    idx = list(range(args.chunk, len(cfgs), args.n_chunks))
    print(f"[{tag}] {len(cfgs)} unique archived configs from {len(sources(tag))} files; chunk has {len(idx)}", flush=True)
    E = StackEval(wls)
    E.enable_fast([cfgs[i] for i in idx[:2]])
    cpi = np.zeros((len(idx), len(wls))); comp = np.zeros((len(idx), len(wls), 3, 4)); bse = np.zeros((len(idx), len(wls), 4))
    for j, i in enumerate(idx):
        for k, wl in enumerate(wls):
            cpi[j, k], comp[j, k], bse[j, k] = E.eval(cfgs[i], wl)
        if (j + 1) % 200 == 0:
            print(f"  {j+1}/{len(idx)} [{time.time()-t0:.0f}s]", flush=True)
    np.savez(out, idx=np.array(idx), configs=np.array([json.dumps(cfgs[i]) for i in idx]),
             area=np.array([areas[i] for i in idx]), src=np.array([src[i] for i in idx]),
             workloads=np.array(wls), cpi=cpi, comp=comp, base=bse)
    print(f"[{tag}] done {len(idx)} configs x {len(wls)} wl [{time.time()-t0:.0f}s] -> {out}", flush=True)


if __name__ == "__main__":
    main()
