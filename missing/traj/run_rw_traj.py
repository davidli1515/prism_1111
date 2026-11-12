#!/usr/bin/env python3
"""
Random walker (ArchGym's RandomWalker = an independent uniformly random design at every step, i.e. random
search) on the same model / space / area as run_traj.py. Designs come from dse17.sample17 (the sampler of the
greedy random starts and of BO's initial points) with a fixed seed, so EVERY benchmark sees the same designs.
Two uses: (1) blocks of 512 consecutive designs = a random-walker optimiser run with the BO/GA budget;
(2) all designs = an optimiser-free reference for the Pareto front and for area-matched non-optimal designs.

  python run_rw_traj.py --bench 602.gcc_s [--n 100000]

Output: results_rw/<bench>/part_XXX.npz (10k designs each; existing parts skipped, never overwritten) with
idx, cfg (17 params, order = meta params), area, cpi, mscs (16); meta.json.
"""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import traj_common as T

RW_SEED, CHUNK = 4026, 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=T.CM.SPEC17)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--out", default=os.path.join(HERE, "results_rw"))
    args = ap.parse_args()
    from area17 import FastArea
    fa = FastArea()
    d = os.path.join(args.out, args.bench); os.makedirs(d, exist_ok=True)
    meta = os.path.join(d, "meta.json")
    if not os.path.exists(meta):
        json.dump({"bench": args.bench, "params": T.SEARCH, "fixed": T.S.FIXED, "sampler": "dse17.sample17",
                   "seed": RW_SEED, "n": args.n, "chunk": CHUNK, "ckpt": T.CM.DEFAULT_CKPT,
                   "max_slices": T.MAX_SLICES, "area": "dse17 fast area model"}, open(meta, "w"), indent=1)
    rng = np.random.RandomState(RW_SEED)
    cfgs = [T.S.sample17(rng) for _ in range(args.n)]
    P = T.WLPredictor(args.bench)
    t0 = time.time()
    for i in range(0, args.n, CHUNK):
        path = os.path.join(d, f"part_{i // CHUNK:03d}.npz")
        if os.path.exists(path):
            continue
        cs = cfgs[i:i + CHUNK]; ev = [P(c) for c in cs]
        arr = {"idx": np.arange(i, i + len(cs), dtype=np.int32),
               "cfg": np.array([[c[p] for p in T.SEARCH] for c in cs], np.int32),
               "area": fa.many(cs).astype(np.float32), "cpi": np.array([e[0] for e in ev], np.float32),
               "mscs": np.array([e[1] for e in ev], np.float32)}
        np.savez_compressed(path + ".tmp.npz", **arr); os.replace(path + ".tmp.npz", path)
        print(f"[{args.bench}] part {i // CHUNK:03d} [{time.time()-t0:.0f}s]", flush=True)
    print(f"[{args.bench}] DONE [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
