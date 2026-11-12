#!/usr/bin/env python3
"""
DSE v4 driver (single benchmark or joint): BO to --bo-queries, then refine_v4 to
--total-queries. Writes <out>/<bench-or-joint>/<method>_s<seed>.json (refuses overwrite),
including per-step walk logs and walk_stats.

Methods: bo | bo_v4_scalar | bo_v4_mscs | bo_v4_mscsavg
Rule parameters: --eps-floor --rel-eps --gamma --tau --tau-m --k-wl

Usage:
  python run_dse_v4.py --bench 602.gcc_s --method bo_v4_mscs --out results_v4_1007
  python run_dse_v4.py --joint --method bo_v4_mscs --out results_v4_1007
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Area, Archive, SPEC17, DEFAULT_CKPT
from refine_v3 import SurrogateWL
from refine_v4 import refine_front_v4, walk_stats, P4
from mscs_reclaim_v2 import prefetch_areas
try:
    from run_dse_v3_single import run_bo_par as _run_bo          # McPAT-batched BO (same search)
    def run_bo(S, arch, seed, budget, pre): return _run_bo(S, arch, seed, budget, pre)
except Exception:
    from bo import run_bo as _run_bo
    def run_bo(S, arch, seed, budget, pre): return _run_bo(S, arch, seed, budget)

METHODS = {"bo": None, "bo_v4_scalar": "scalar", "bo_v4_mscs": "mscs", "bo_v4_mscsavg": "mscs_avg"}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bench", choices=SPEC17)
    g.add_argument("--joint", action="store_true")
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total-queries", type=int, default=5000)
    ap.add_argument("--bo-queries", type=int, default=500)
    ap.add_argument("--refine-steps", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mcpat-workers", type=int, default=4)
    for k, v in P4.items():
        if k != "dec_scales":
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wls = SPEC17 if args.joint else [args.bench]
    tag = "joint_17" if args.joint else args.bench
    d = os.path.join(args.out, tag); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{args.method}_s{args.seed}.json")
    if os.path.exists(path):
        sys.exit(f"exists, refusing to overwrite: {path}")
    p = dict(P4, **{k: getattr(args, k) for k in P4 if k != "dec_scales"})
    S = SurrogateWL(DEFAULT_CKPT, wls, "18k", device=args.device, max_slices=50)
    A = Area(); arch = Archive(S, A)
    pre = lambda cs: prefetch_areas(A, cs, args.mcpat_workers)
    t0 = time.time(); sel = METHODS[args.method]
    print(f"[{tag} {args.method} s{args.seed}] budget={args.total_queries} params={p}", flush=True)
    run_bo(S, arch, args.seed, args.total_queries if sel is None else args.bo_queries, pre)
    walks = []
    if sel is not None:
        walks = refine_front_v4(S, arch, sel, A, args.total_queries, args.refine_steps, p=p,
                                prefetch=pre, log=lambda m: print(m, flush=True))
    A.flush()
    res = arch.to_json(method=args.method, seed=args.seed, bench=tag, args=vars(args), params=p,
                       ckpt=DEFAULT_CKPT, wall_time=time.time() - t0, surrogate_time=S.t_query,
                       mcpat_failures=A.n_fail, walks=walks, walk_stats=walk_stats(walks))
    json.dump(res, open(path, "w"), indent=1, default=str)
    print(f"[{tag} {args.method} s{args.seed}] done q={S.n_queries} pts={len(arch.cpis)} "
          f"stats={res['walk_stats']} -> {path}", flush=True)


if __name__ == "__main__":
    main()
