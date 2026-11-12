#!/usr/bin/env python3
"""
BO as the optimiser (companion to the greedy trajectories). Same model / space / area as run_traj.py:
traj_common.WLPredictor (gem5-validated surrogate), dse17 17-parameter space, dse17 fast area model.
Optimiser = the original dse_engine.run_bo_gem (imported, not modified; its candidate sampler is swapped
for the 17-parameter sampler at runtime via space17.patched_bo_sampler): 30 random initial points, then
GP (Matern) on CPI and area with an optimistic-HVI acquisition over 500 random candidates per iteration;
budget 500. MSCS is only recorded (from the same model, cached).

  python run_bo_traj.py --bench 602.gcc_s --seeds 0 1 2 ...

Output: results_bo/<bench>/seed_XX.npz (refuses to overwrite) with, in evaluation order: it, phase
(0 = random initial, 1 = BO), cfg (17 params, order = meta params), area, cpi, mscs (16); meta.json.
"""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import traj_common as T


class Pred:
    def __init__(self, P):
        self.P = P

    def predict_cpi(self, c):
        return self.P(T.S.project17(c))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=T.CM.SPEC17)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--n-initial", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "results_bo"))
    args = ap.parse_args()
    from area17 import FastArea
    fa = FastArea()
    area = lambda c: fa(T.S.project17(c))
    d = os.path.join(args.out, args.bench); os.makedirs(d, exist_ok=True)
    meta = os.path.join(d, "meta.json")
    if not os.path.exists(meta):
        json.dump({"bench": args.bench, "params": T.SEARCH, "fixed": T.S.FIXED, "budget": args.budget,
                   "n_initial": args.n_initial, "optimiser": "dse_engine.run_bo_gem (17-param sampler)",
                   "ckpt": T.CM.DEFAULT_CKPT, "max_slices": T.MAX_SLICES, "area": "dse17 fast area model"},
                  open(meta, "w"), indent=1)
    P = T.WLPredictor(args.bench)
    t0 = time.time()
    for s in args.seeds:
        path = os.path.join(d, f"seed_{s:02d}.npz")
        if os.path.exists(path):
            continue
        with T.S.patched_bo_sampler():
            res = T.C.DE.run_bo_gem(Pred(P), area, budget=args.budget, n_initial=args.n_initial,
                                    seed=s, verbose=False)
        cfgs = [T.S.project17(c) for c in res["configs"]]
        ev = [P(c) for c in cfgs]
        n = len(cfgs)
        arr = {"it": np.arange(n, dtype=np.int32),
               "phase": (np.arange(n) >= args.n_initial).astype(np.int8),
               "cfg": np.array([[c[p] for p in T.SEARCH] for c in cfgs], np.int32),
               "area": np.array(res["areas"], np.float32), "cpi": np.array([e[0] for e in ev], np.float32),
               "cpi_bo": np.array(res["cpis"], np.float32), "mscs": np.array([e[1] for e in ev], np.float32)}
        np.savez_compressed(path + ".tmp.npz", **arr); os.replace(path + ".tmp.npz", path)
        print(f"[{args.bench}] seed {s}: {n} evaluations, best CPI {arr['cpi'].min():.3f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"[{args.bench}] DONE [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
