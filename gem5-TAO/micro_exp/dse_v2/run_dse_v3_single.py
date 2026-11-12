#!/usr/bin/env python3
"""
Single-benchmark DSE v3 (finite differences), one benchmark per run.

Same methods / refine as run_dse_v3.py (joint), but the surrogate covers ONE workload:
  bo             pure BO
  bo_v3_scalar   BO + v3 refine, scalar decisions
  bo_v3_mscs     BO + v3 refine, MSCS decouple + protect
BO here batches its McPAT calls (identical search to bo.run_bo; only faster).

Usage: python run_dse_v3_single.py --bench 605.mcf_s --method bo_v3_mscs --seed 42 \
           --device cuda:1 --out results_v3_single_1006
Output: <out>/<bench>/<method>_s<seed>.json   (refuses to overwrite)
"""
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Area, Archive, SPEC17, DEFAULT_CKPT, sample_random_config, pareto_mask, hypervolume
from bo import to_vec, _gp_subset
from refine_v3 import SurrogateWL, refine_front_v3, P3
from mscs_reclaim_v2 import prefetch_areas

METHODS = {"bo": None, "bo_v3_scalar": "scalar", "bo_v3_mscs": "mscs"}


def run_bo_par(S, archive, seed, query_budget, prefetch, n_initial=30, n_cand=500,
               batch=10, gp_cap=600, log_every=250):
    """bo.run_bo with the McPAT calls of each proposal batch done in parallel."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    rng = np.random.RandomState(seed)
    t0 = time.time()
    init = [sample_random_config(rng) for _ in range(3 * n_initial)]
    prefetch(init)
    for c in init:
        if len(archive.cpis) >= n_initial:
            break
        archive.add(c, "bo_init")
    last = 0
    while S.n_queries < query_budget:
        idx = _gp_subset(archive, gp_cap, rng)
        X = np.array([to_vec(archive.configs[i]) for i in idx])
        kw = dict(kernel=ConstantKernel() * Matern(nu=2.5), n_restarts_optimizer=2,
                  normalize_y=True, random_state=seed)
        gp_c = GaussianProcessRegressor(**kw).fit(X, np.array(archive.cpis)[idx])
        gp_a = GaussianProcessRegressor(**kw).fit(X, np.array(archive.areas)[idx])
        P = np.c_[archive.cpis, archive.areas]
        F = P[pareto_mask(P)]
        ref = (P[:, 0].max() * 1.1, P[:, 1].max() * 1.1)
        hv0 = hypervolume(F, ref)
        cands = [sample_random_config(rng) for _ in range(n_cand)]
        V = np.array([to_vec(c) for c in cands])
        mc, sc = gp_c.predict(V, return_std=True)
        ma, sa = gp_a.predict(V, return_std=True)
        score = np.array([hypervolume(np.vstack([F, [mc[i] - 1.5 * sc[i], ma[i] - 1.5 * sa[i]]]), ref)
                          - hv0 + 0.1 * (sc[i] + sa[i]) for i in range(n_cand)])
        pick = [cands[i] for i in np.argsort(-score)[:batch]]
        prefetch(pick)
        before = S.n_queries
        for c in pick:
            if S.n_queries >= query_budget:
                break
            archive.add(c, "bo")
        if S.n_queries == before:
            archive.add(sample_random_config(rng), "bo_rand")
        if S.n_queries - last >= log_every:
            last = S.n_queries
            print(f"    BO q={S.n_queries} pts={len(archive.cpis)} [{time.time()-t0:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=SPEC17)
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total-queries", type=int, default=5000)
    ap.add_argument("--bo-queries", type=int, default=500)
    ap.add_argument("--refine-steps", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mcpat-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    od = os.path.join(args.out, args.bench); os.makedirs(od, exist_ok=True)
    path = os.path.join(od, f"{args.method}_s{args.seed}.json")
    if os.path.exists(path):
        sys.exit(f"exists, refusing to overwrite: {path}")
    S = SurrogateWL(DEFAULT_CKPT, [args.bench], "18k", device=args.device, max_slices=50)
    A = Area(); arch = Archive(S, A)
    pre = lambda cs: prefetch_areas(A, cs, args.mcpat_workers)
    t0 = time.time()
    sel = METHODS[args.method]
    print(f"[{args.bench} {args.method} s{args.seed}] budget={args.total_queries}", flush=True)
    run_bo_par(S, arch, args.seed, args.total_queries if sel is None else args.bo_queries, pre)
    if sel is not None:
        refine_front_v3(S, arch, sel, A, args.total_queries, args.refine_steps, prefetch=pre,
                        log=lambda m: print(m, flush=True))
    A.flush()
    res = arch.to_json(method=args.method, bench=args.bench, seed=args.seed, args=vars(args),
                       params=P3, ckpt=DEFAULT_CKPT, wall_time=time.time() - t0,
                       surrogate_time=S.t_query, mcpat_failures=A.n_fail)
    json.dump(res, open(path, "w"), indent=1, default=str)
    print(f"[{args.bench} {args.method} s{args.seed}] done q={S.n_queries} pts={len(arch.cpis)} "
          f"wall={res['wall_time']:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
