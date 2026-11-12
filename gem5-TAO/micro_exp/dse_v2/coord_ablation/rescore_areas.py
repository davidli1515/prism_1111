#!/usr/bin/env python3
"""
Re-score the areas of every archived config with a race-free McPAT call, then write
results/<name>_s<seed>.rescored.json (originals untouched).

Why: the original run's AreaRO.prefetch ran McPAT in threads of one process; area_model.compute_area
writes and DELETES <work_dir>/mcpat_temp_<pid>.xml, so threads overwrote / removed each other's
XML (wrong areas and spurious failures).  Here every call gets its own fresh directory and runs in
its own process; no old area cache is read.

Checks before anything is written (any failure -> abort):
  1. serial vs parallel: 40 configs computed one by one in this process == the parallel values
  2. repeat: the same 40 configs computed again in parallel (different order) == first values
  3. every fetchBufferSize == 8 config fails (known McPAT limit), and every other failure fails
     again when retried serially in isolation (genuine McPAT failures, e.g. 16 KB L1 with 32 ways)
Usage: python rescore_areas.py [--seed 42] [--workers 16]
"""
import os, sys, json, glob, shutil, tempfile, argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coord as C                                       # sets paths to dse_exp (read-only import)
from area_model import compute_area

WORK = os.path.join(HERE, "mcpat_work_rescore")


def area_isolated(c):
    d = tempfile.mkdtemp(dir=WORK)
    try:
        return compute_area(c, work_dir=d)
    except Exception:
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def same(a, b):
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1e-9 * max(1, abs(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    names = ["bo_base"] + list(C.ARMS)
    files = {n: os.path.join(HERE, "results", f"{n}_s{args.seed}.json") for n in names}
    outs = {n: f.replace(".json", ".rescored.json") for n, f in files.items()}
    for n in names:
        if not os.path.exists(files[n]):
            sys.exit(f"missing {files[n]}")
        if os.path.exists(outs[n]):
            sys.exit(f"exists, refusing to overwrite: {outs[n]}")
    D = {n: json.load(open(files[n])) for n in names}
    uniq = {}
    for d in D.values():
        for c in d["configs"]:
            uniq.setdefault(C.ckey(c), c)
    keys = list(uniq)
    print(f"{len(keys)} unique configs", flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        vals = list(ex.map(area_isolated, [uniq[k] for k in keys], chunksize=4))
    area = dict(zip(keys, vals))

    rng = np.random.RandomState(0)
    probe = [keys[i] for i in rng.choice(len(keys), min(40, len(keys)), replace=False)]
    serial = [area_isolated(uniq[k]) for k in probe]
    with ProcessPoolExecutor(args.workers) as ex:
        again = list(ex.map(area_isolated, [uniq[k] for k in probe[::-1]]))[::-1]
    ok1 = all(same(area[k], v) for k, v in zip(probe, serial))
    ok2 = all(same(area[k], v) for k, v in zip(probe, again))
    fb8 = [uniq[k]["fetchBufferSize"] == 8 for k in keys]
    fails = [area[k] is None for k in keys]
    other = [k for k, f, b in zip(keys, fails, fb8) if f and not b]
    ok3 = all(f for f, b in zip(fails, fb8) if b) and all(area_isolated(uniq[k]) is None for k in other)
    print(f"check serial==parallel {ok1}  repeat identical {ok2}  failures genuine {ok3} "
          f"(failures {sum(fails)}: fb=8 {sum(fb8)}, other {len(other)} reproduced serially)", flush=True)
    if not (ok1 and ok2 and ok3):
        sys.exit("CHECKS FAILED, nothing written")

    for n, d in D.items():
        old = np.array([a if a is not None else np.nan for a in d["areas"]], float)
        new_all = [area[C.ckey(c)] for c in d["configs"]]
        keep = [i for i, a in enumerate(new_all) if a is not None]
        new = np.array([new_all[i] for i in keep])
        bad = ~np.isclose(old[keep], new, rtol=1e-6)
        r = dict(d)
        for k in ("configs", "cpis", "cost", "origin"):
            if k in d:
                r[k] = [d[k][i] for i in keep]
        r["areas"] = new.tolist()
        r["rescore"] = {"n_in": len(d["configs"]), "n_kept": len(keep), "n_area_changed": int(bad.sum()),
                        "median_rel_change": float(np.median(np.abs(new[bad] - old[keep][bad]) / old[keep][bad])) if bad.any() else 0.0}
        json.dump(r, open(outs[n], "w"))
        print(f"  {n:18s} kept {len(keep)}/{len(d['configs'])}  area changed {bad.sum()} "
              f"({bad.mean():.2%}, median |rel| {r['rescore']['median_rel_change']:.2f}) -> {outs[n]}", flush=True)
    shutil.rmtree(WORK, ignore_errors=True)
    print("RESCORE DONE")


if __name__ == "__main__":
    main()
