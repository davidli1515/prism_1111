#!/usr/bin/env python3
"""
DSE v2 driver: one (method, seed) per invocation, results to <out>/<method>_s<seed>.json

Methods
  greedy_scalar          unified greedy, scalar decisions
  greedy_mscs            unified greedy, Algorithm 2 (decouple + protect)
  greedy_mscs_decouple   ablation: decouple only
  greedy_mscs_protect    ablation: protect only
  bo                     pure BO up to --total-queries
  bo_scalar              BO to --bo-queries, then scalar refine up to --total-queries
  bo_mscs                BO to --bo-queries, then Algorithm-2 refine up to --total-queries
  random                 random sampling up to --total-queries (sanity floor)

All methods are charged in surrogate queries (distinct configs evaluated on all
workloads); greedy methods are also capped by --total-queries so every method
gets the same budget.

Usage:
  python run_dse_v2.py --method bo_mscs --seed 42 --out results_v2_1005
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Surrogate, Area, Archive, SPEC17, DEFAULT_CKPT, sample_random_config
from search import run_greedy, DEFAULTS
from bo import run_bo, run_bo_refine

METHODS = ["greedy_scalar", "greedy_mscs", "greedy_mscs_decouple", "greedy_mscs_protect",
           "greedy_mscs_jac", "greedy_scalar_jac",
           "bo", "bo_scalar", "bo_mscs", "bo_mscs_jac", "random"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--benchmarks", nargs="+", default=SPEC17)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--train-data", default="18k", choices=["18k", "40k"],
                    help="dataset the checkpoint was trained on (for the normalizer)")
    ap.add_argument("--cpi-stage", default="mean", choices=["mean", "commit"])
    ap.add_argument("--total-queries", type=int, default=3000)
    ap.add_argument("--bo-queries", type=int, default=500)
    ap.add_argument("--n-starts", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--refine-steps", type=int, default=15)
    ap.add_argument("--cancel", default="paper", choices=["paper", "legacy"])
    ap.add_argument("--eps", type=float, default=DEFAULTS["eps"])
    ap.add_argument("--tau", type=float, default=DEFAULTS["tau"])
    ap.add_argument("--tau-m", type=float, default=DEFAULTS["tau_m"])
    ap.add_argument("--max-slices", type=int, default=50,
                    help="slices per workload used by the surrogate (dse_exp used 200)")
    ap.add_argument("--bo-batch", type=int, default=10)
    ap.add_argument("--gp-cap", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.method}_s{args.seed}.json")
    if os.path.exists(path):
        sys.exit(f"exists, refusing to overwrite: {path}")
    p = dict(DEFAULTS, eps=args.eps, tau=args.tau, tau_m=args.tau_m)

    S = Surrogate(args.ckpt, args.benchmarks, args.train_data, cpi_stage=args.cpi_stage,
                  max_slices=args.max_slices)
    bo_kw = dict(batch=args.bo_batch, gp_cap=args.gp_cap)
    A = Area()
    arch = Archive(S, A)
    t0 = time.time()
    print(f"[{args.method} s{args.seed}] {len(args.benchmarks)} workloads, budget={args.total_queries}",
          flush=True)

    extra = {}
    m = args.method
    if m.startswith("greedy_"):
        sel = m[len("greedy_"):]
        extra["trajectories"] = run_greedy(S, arch, sel, args.seed, args.n_starts, args.max_steps,
                                           p, args.cancel, query_budget=args.total_queries)
    elif m == "bo":
        run_bo(S, arch, args.seed, args.total_queries, **bo_kw)
    elif m in ("bo_scalar", "bo_mscs", "bo_mscs_jac"):
        run_bo_refine(S, arch, args.seed, m[len("bo_"):], args.bo_queries, args.total_queries,
                      args.refine_steps, p=p, cancel=args.cancel, **bo_kw)
    elif m == "random":
        rng = np.random.RandomState(args.seed)
        while S.n_queries < args.total_queries:
            arch.add(sample_random_config(rng), "random")

    A.flush()
    res = arch.to_json(method=m, seed=args.seed, args=vars(args), params=p,
                       ckpt=S.ckpt, wall_time=time.time() - t0,
                       surrogate_time=S.t_query, n_jacobians=S.n_jac,
                       mcpat_failures=A.n_fail, **extra)
    json.dump(res, open(path, "w"), indent=1, default=str)
    print(f"[{m} s{args.seed}] done: queries={S.n_queries} points={len(arch.cpis)} "
          f"pareto={len(res['pareto_indices'])} wall={res['wall_time']:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
