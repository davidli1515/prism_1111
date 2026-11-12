#!/usr/bin/env python3
"""
Joint (17-benchmark geomean) DSE v3: BO + finite-difference refine.

Methods (all to --total-queries; refine methods: BO to --bo-queries first)
  bo             pure BO
  bo_v3_scalar   BO + v3 refine, scalar decisions
  bo_v3_mscs     BO + v3 refine, per-workload MSCS (decouple + protect)
  bo_v3_mscsavg  BO + v3 refine, averaged MSCS (ablation)

Usage: python run_dse_v3.py --method bo_v3_mscs --seed 42 --device cuda:0 --out results_v3_1005
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Area, Archive, SPEC17, DEFAULT_CKPT
from bo import run_bo
from refine_v3 import SurrogateWL, refine_front_v3, P3
from mscs_reclaim_v2 import prefetch_areas

METHODS = {"bo": None, "bo_v3_scalar": "scalar", "bo_v3_mscs": "mscs", "bo_v3_mscsavg": "mscs_avg"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total-queries", type=int, default=5000)
    ap.add_argument("--bo-queries", type=int, default=500)
    ap.add_argument("--refine-steps", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mcpat-workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.method}_s{args.seed}.json")
    if os.path.exists(path):
        sys.exit(f"exists, refusing to overwrite: {path}")
    S = SurrogateWL(DEFAULT_CKPT, SPEC17, "18k", device=args.device, max_slices=50)
    A = Area(); arch = Archive(S, A)
    pre = lambda cs: prefetch_areas(A, cs, args.mcpat_workers)
    t0 = time.time()
    sel = METHODS[args.method]
    print(f"[{args.method} s{args.seed}] budget={args.total_queries} bo={args.bo_queries}", flush=True)
    if sel is None:
        run_bo(S, arch, args.seed, args.total_queries)
    else:
        run_bo(S, arch, args.seed, args.bo_queries)
        refine_front_v3(S, arch, sel, A, args.total_queries, args.refine_steps, prefetch=pre,
                        log=lambda m: print(m, flush=True))
    A.flush()
    res = arch.to_json(method=args.method, seed=args.seed, args=vars(args), params=P3,
                       ckpt=S.ckpt, wall_time=time.time() - t0, surrogate_time=S.t_query,
                       mcpat_failures=A.n_fail)
    json.dump(res, open(path, "w"), indent=1, default=str)
    print(f"[{args.method} s{args.seed}] done q={S.n_queries} pts={len(arch.cpis)} "
          f"wall={res['wall_time']:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
