#!/usr/bin/env python3
"""
Post-BO reclaim v2: rank one-stride decreases by *benefit / cost*.

  benefit = area saved by the move (McPAT; not a surrogate query, so every method may use it)
  cost    = predicted performance impact from the Jacobian screen (1 query per step)

Methods (verify the top-k by exact query, accept if CPI rises < tau):
  mscs_area    area_saved / (||ΔMSCS_pred||_1 + eps_m)     MSCS says the resource is idle
  scalar_area  area_saved / (max(ΔCPI_pred,0) + eps_c)     same, scalar prediction (isolates MSCS)
  area_only    area_saved                                   no performance information
  exhaustive   verify every decrease, keep the largest area saving among acceptable ones

v1 (mscs_reclaim.py, results in reclaim/) ranked by ||ΔMSCS|| alone and lost to random
because it picked idle-but-tiny resources. Code in mscs_reclaim.py is reused, not modified.

Usage: python mscs_reclaim_v2.py --method mscs_area --device cuda:0 --budget 1000
Output: results_v2_1005/reclaim_v2/<method>_s<seed>.json
"""
import os, sys, json, time, argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Surrogate, Area, Archive, config_key, arch_delta, pareto_mask, MCPAT_WORK
from mscs_reclaim import decrease_moves
from area_model import compute_area

METHODS = ["mscs_area", "scalar_area", "area_only", "exhaustive"]


def prefetch_areas(A, configs, workers=8):
    """Fill the area cache for many configs in parallel (McPAT is a subprocess)."""
    todo = [c for c in configs if A._k(c) not in A.cache]
    def f(c):
        try:
            return A._k(c), compute_area(c, work_dir=MCPAT_WORK)
        except Exception:
            return A._k(c), None
    with ThreadPoolExecutor(workers) as ex:
        for k, a in ex.map(f, todo):
            A.cache[k] = a
            if a is None:
                A.n_fail += 1
    A._dirty += len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--base", default="results_v2_1005/bo_s42.json")
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.003)
    ap.add_argument("--eps-m", type=float, default=1e-3)
    ap.add_argument("--eps-c", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mcpat-workers", type=int, default=8)
    ap.add_argument("--out", default="results_v2_1005/reclaim_v2")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.method}_s{args.seed}.json")
    if os.path.exists(path):
        sys.exit(f"exists, refusing to overwrite: {path}")
    base = json.load(open(args.base))
    S, A = Surrogate(device=args.device), Area()
    arch = Archive(S, A)
    for c, cpi, a in zip(base["configs"], base["cpis"], base["areas"]):
        arch._seen.add(config_key(c))
        arch.configs.append(c); arch.cpis.append(cpi); arch.areas.append(a)
        arch.queries.append(0); arch.tags.append("bo")
    P = np.c_[arch.cpis, arch.areas]
    starts = sorted([arch.configs[i] for i in np.where(pareto_mask(P))[0]], key=lambda c: -A(c))
    active = {i: c for i, c in enumerate(starts)}
    print(f"[{args.method}] {len(starts)} Pareto starts from {args.base}", flush=True)

    t0, steps, accepted, area_saved = time.time(), 0, 0, 0.0
    while active and S.n_queries < args.budget:
        for i in list(active):
            if S.n_queries >= args.budget:
                break
            c = active[i]
            cpi0, a0 = S.cpi(c), A(c)
            moves = decrease_moves(c)
            prefetch_areas(A, [cn for _, cn in moves], args.mcpat_workers)
            moves = [(n, cn, a0 - A(cn)) for n, cn in moves if A(cn) is not None and A(cn) < a0]
            if args.method == "exhaustive":
                order = moves
            else:
                if args.method == "area_only":
                    score = lambda m: m[2]
                else:
                    Jc, Jm = S.jac(c)
                    if args.method == "mscs_area":
                        score = lambda m: m[2] / (np.abs(Jm @ arch_delta(c, m[1])).sum() + args.eps_m)
                    else:
                        score = lambda m: m[2] / (max(float(Jc @ arch_delta(c, m[1])) / cpi0, 0.0)
                                                  + args.eps_c)
                order = sorted(moves, key=score, reverse=True)[:args.k]
            ok = []
            for n, cn, saved in order:
                if S.n_queries >= args.budget:
                    break
                arch.add(cn, f"reclaim_{args.method}")
                if (S.cpi(cn) - cpi0) / cpi0 < args.tau:
                    ok.append((n, cn, saved))
                    if args.method != "exhaustive":
                        break
            steps += 1
            if not ok:
                del active[i]
                continue
            n, cn, saved = max(ok, key=lambda m: m[2])
            active[i] = cn; accepted += 1; area_saved += saved
        print(f"  q={S.n_queries} steps={steps} accepted={accepted} active={len(active)} "
              f"area_saved_total={area_saved:.1f} [{time.time()-t0:.0f}s]", flush=True)
        A.flush()

    A.flush()
    res = arch.to_json(method=f"reclaim2_{args.method}", seed=args.seed, args=vars(args),
                       base=args.base, steps=steps, accepted=accepted, area_saved=area_saved,
                       n_jacobians=S.n_jac, wall_time=time.time() - t0)
    json.dump(res, open(path, "w"), indent=1, default=str)
    print(f"[{args.method}] done q={S.n_queries} -> {path}", flush=True)


if __name__ == "__main__":
    main()
