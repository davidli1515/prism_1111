#!/usr/bin/env python3
"""
GA as the optimiser (companion to the greedy and BO trajectories). Same model / space / area as run_traj.py
and run_bo_traj.py: traj_common.WLPredictor, dse17 17-parameter space, dse17 fast area model.
Optimiser = ArchGym's GA (oss-arch-gym/sko/GA.py, scikit-opt; imported, not modified): binary Gray coding,
tournament selection, 2-point crossover, bit-flip mutation, no elitism. Hyper-parameters from ArchGym's grid
(configs/arch_gym_configs.py): population 16 x 32 generations = 512 evaluations, prob_mut 0.01.
Each parameter is encoded as the index into its value list (lb 0, ub len-1, precision 1); the decoded
configuration goes through dse17.project17 (validity rules) before evaluation.
GA is single-objective: minimise CPI subject to area <= A_ref (sko constraint_ueq, penalty 1e5 x violation),
A_ref = median end area of the greedy small-start walks of this benchmark (the matched area of analyze_bo.py).

  python run_ga_traj.py --bench 602.gcc_s --seeds 0 1 2 ...

Output: results_ga/<bench>/seed_XX.npz (refuses to overwrite) with, in evaluation order: it, gen, cfg (17
params, order = meta params), area, cpi, mscs (16), feasible (area <= A_ref); meta.json.
"""
import os, sys, json, glob, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import traj_common as T
from config_dse import get_param_values
sys.path.insert(0, "/scratch/gem5-TAO/oss-arch-gym")
from sko.GA import GA

POP, GENS, PROB_MUT = 16, 32, 0.01
VALS = {p: get_param_values(p) for p in T.SEARCH}


def decode(x):
    c = {p: VALS[p][int(np.clip(round(v), 0, len(VALS[p]) - 1))] for p, v in zip(T.SEARCH, x)}
    c.update(T.S.FIXED)
    return T.S.project17(c)


def a_ref(bench):
    fs = sorted(glob.glob(os.path.join(HERE, "results_small", bench, "part_*.npz")))
    st = np.concatenate([np.load(f)["start"] for f in fs]); ar = np.concatenate([np.load(f)["area"] for f in fs])
    return float(np.median(ar[np.r_[np.where(np.diff(st) != 0)[0], len(st) - 1]]))


def run_one(P, area, A, seed):
    log, acache = [], {}

    def area_of(c):
        k = T.ckey(c)
        if k not in acache:
            acache[k] = area(c)
        return acache[k]

    def f(x):
        c = decode(x); cpi, m = P(c); a = area_of(c)
        log.append((c, a, cpi, m))
        return cpi

    np.random.seed(seed)                                   # sko draws from the global numpy RNG
    ga = GA(func=f, n_dim=len(T.SEARCH), size_pop=POP, max_iter=GENS, prob_mut=PROB_MUT,
            lb=[0] * len(T.SEARCH), ub=[len(VALS[p]) - 1 for p in T.SEARCH], precision=[1] * len(T.SEARCH),
            constraint_ueq=(lambda x: area_of(decode(x)) - A,))
    ga.run()
    log = log[:POP * GENS]                                 # drop sko's final re-evaluation of best_x
    n = len(log)
    return {"it": np.arange(n, dtype=np.int32), "gen": (np.arange(n) // POP).astype(np.int16),
            "cfg": np.array([[c[p] for p in T.SEARCH] for c, *_ in log], np.int32),
            "area": np.array([a for _, a, _, _ in log], np.float32),
            "cpi": np.array([cpi for _, _, cpi, _ in log], np.float32),
            "mscs": np.array([m for *_, m in log], np.float32),
            "feasible": np.array([a <= A for _, a, _, _ in log])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=T.CM.SPEC17)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default=os.path.join(HERE, "results_ga"))
    args = ap.parse_args()
    from area17 import FastArea
    fa = FastArea()
    A = a_ref(args.bench)
    d = os.path.join(args.out, args.bench); os.makedirs(d, exist_ok=True)
    meta = os.path.join(d, "meta.json")
    if not os.path.exists(meta):
        json.dump({"bench": args.bench, "params": T.SEARCH, "fixed": T.S.FIXED, "optimiser": "ArchGym sko.GA",
                   "size_pop": POP, "max_iter": GENS, "prob_mut": PROB_MUT, "objective": "min CPI s.t. area <= A_ref",
                   "A_ref": A, "A_ref_def": "median end area of greedy small-start walks", "ckpt": T.CM.DEFAULT_CKPT,
                   "max_slices": T.MAX_SLICES, "area": "dse17 fast area model"}, open(meta, "w"), indent=1)
    P = T.WLPredictor(args.bench)
    t0 = time.time()
    for s in args.seeds:
        path = os.path.join(d, f"seed_{s:02d}.npz")
        if os.path.exists(path):
            continue
        arr = run_one(P, fa, A, s)
        np.savez_compressed(path + ".tmp.npz", **arr); os.replace(path + ".tmp.npz", path)
        ok = arr["feasible"]
        print(f"[{args.bench}] seed {s}: {len(ok)} evaluations, feasible {ok.mean():.2f}, best feasible CPI "
              f"{arr['cpi'][ok].min() if ok.any() else float('nan'):.3f} (A_ref {A:.1f}) [{time.time()-t0:.0f}s]", flush=True)
    print(f"[{args.bench}] DONE [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
