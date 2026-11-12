#!/usr/bin/env python3
"""
gem5 on an expert walk written by expert_walk_at.py (same runner and command line as gem5_val4 / gem5_expert.py).
The original walk's gem5 points come from gem5_val4 (never re-run). Expert record k is the state k steps after the
decision (search step at + 1 + k).
Writes expert/gem5/runs/<bench>/p<start>_at<at>_<tag>_s<k>/result.json; never overwrites.
  python gem5_expert_at.py expert/607.cactuBSSN_s_s840_at10_port2.json --steps 0 2 4 8 12
"""
import os, sys, json, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); DSE_V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DSE_V2, "gem5_bounds")); sys.path.insert(0, HERE)
import traj_common as T
from run_gem5_bounds import run_job, surrogate_insts

OUT = os.path.join(HERE, "expert", "gem5")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("walk"); ap.add_argument("--steps", type=int, nargs="+", required=True)
    a = ap.parse_args()
    W = json.load(open(a.walk)); recs = {r["step"]: r for r in W["expert"]}
    jobs = []
    for k in a.steps:
        if k not in recs:
            print(f"  step {k} not in the expert walk ({len(recs)} records), skipped"); continue
        c = T.S.project17(dict(recs[k]["cfg"]))
        jobs.append({"bench": W["bench"], "idx": W["start"], "tag": f"at{W['at']}_{W['tag']}_s{k:02d}", "config": c,
                     "insts": surrogate_insts(W["bench"], c["l1_size"])})
    print(f"{len(jobs)} gem5 runs", flush=True)
    with ThreadPoolExecutor(max(len(jobs), 1)) as ex:
        for f in as_completed([ex.submit(run_job, j, OUT) for j in jobs]):
            r = f.result(); print(f"  {r['tag']} {'skip' if r.get('skipped') else r.get('error') or 'ok'} CPI {r.get('cpi')}", flush=True)


if __name__ == "__main__":
    main()
