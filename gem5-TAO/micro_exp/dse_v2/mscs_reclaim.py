#!/usr/bin/env python3
"""
Post-BO reclaim: start from a finished BO run, walk its Pareto points toward
smaller area by removing resources that do not buy performance, under a fixed
extra surrogate-query budget. Compares how the next move is chosen:

  mscs        Jacobian screen (1 query): rank one-stride decreases by predicted
              ||ΔMSCS||_1 (resource the pipeline does not use -> small change),
              verify the top-k exactly
  scalar_jac  same screen, ranked by predicted ΔCPI only        (isolates MSCS)
  random      random order, verify k                            (no information)
  exhaustive  verify every decrease, take the largest area saving (no screening)

A move is accepted if the exact CPI rises by < tau (relative). All 21 parameters
are eligible (incl. l2_size, l2_assoc, fetchBufferSize, wbWidth, bp_type), since the
headroom check showed many missed improvements there. Every verified config enters
the archive, so HV counts all non-dominated designs found.

Usage:
  python mscs_reclaim.py --method mscs --device cuda:0 --budget 1000
Output: results_v2_1005/reclaim/<method>_s<seed>.json
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (Surrogate, Area, Archive, ALL_PARAMS, config_key, arch_delta,
                    pareto_mask, hypervolume)
from dse_engine import _project_to_valid, _decrease_by_stride

METHODS = ["mscs", "scalar_jac", "random", "exhaustive"]


def decrease_moves(c):
    out = []
    for n in ALL_PARAMS:
        v, _ = _decrease_by_stride(c, n)
        if v is None:
            continue
        cn = dict(c); cn[n] = v; cn = _project_to_valid(cn)
        if config_key(cn) != config_key(c):
            out.append((n, cn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--base", default="results_v2_1005/bo_s42.json")
    ap.add_argument("--budget", type=int, default=1000, help="extra surrogate queries")
    ap.add_argument("--k", type=int, default=3, help="exact verifications per step")
    ap.add_argument("--tau", type=float, default=0.003, help="max relative CPI increase")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results_v2_1005/reclaim")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.method}_s{args.seed}.json")
    if os.path.exists(path):
        sys.exit(f"exists, refusing to overwrite: {path}")
    base = json.load(open(args.base))
    S, A = Surrogate(device=args.device), Area()
    arch = Archive(S, A)
    for c, cpi, a in zip(base["configs"], base["cpis"], base["areas"]):   # BO archive, 0 extra queries
        k = config_key(c); arch._seen.add(k)
        arch.configs.append(c); arch.cpis.append(cpi); arch.areas.append(a)
        arch.queries.append(0); arch.tags.append("bo")
    rng = np.random.RandomState(args.seed)
    P = np.c_[arch.cpis, arch.areas]
    starts = [arch.configs[i] for i in np.where(pareto_mask(P))[0]]
    starts.sort(key=lambda c: -A(c))
    active = {i: c for i, c in enumerate(starts)}
    print(f"[{args.method}] base={args.base}: {len(base['cpis'])} pts, {len(starts)} Pareto starts", flush=True)

    t0, steps, accepted = time.time(), 0, 0
    while active and S.n_queries < args.budget:
        for i in list(active):                       # round-robin over front points
            if S.n_queries >= args.budget:
                break
            c = active[i]
            cpi0 = S.cpi(c)
            moves = decrease_moves(c)
            if args.method in ("mscs", "scalar_jac"):
                Jc, Jm = S.jac(c)
                if args.method == "mscs":
                    key = lambda m: np.abs(Jm @ arch_delta(c, m[1])).sum()
                else:
                    key = lambda m: (Jc @ arch_delta(c, m[1])) / cpi0
                order = sorted(moves, key=key)[:args.k]
            elif args.method == "random":
                order = [moves[j] for j in rng.permutation(len(moves))[:args.k]]
            else:
                order = moves
            ok = []
            for n, cn in order:
                if S.n_queries >= args.budget:
                    break
                r = arch.add(cn, f"reclaim_{args.method}")
                cpi = S.cpi(cn)
                if (cpi - cpi0) / cpi0 < args.tau:
                    ok.append((n, cn))
                    if args.method != "exhaustive":
                        break                        # first acceptable move
            steps += 1
            if not ok:
                del active[i]
                continue
            n, cn = min(ok, key=lambda m: A(m[1]))   # exhaustive: largest area saving
            active[i] = cn; accepted += 1
        P = np.c_[arch.cpis, arch.areas]
        print(f"  q={S.n_queries} steps={steps} accepted={accepted} active={len(active)} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    A.flush()
    res = arch.to_json(method=f"reclaim_{args.method}", seed=args.seed, args=vars(args),
                       base=args.base, steps=steps, accepted=accepted,
                       n_jacobians=S.n_jac, wall_time=time.time() - t0)
    json.dump(res, open(path, "w"), indent=1, default=str)
    print(f"[{args.method}] done q={S.n_queries} -> {path}", flush=True)


if __name__ == "__main__":
    main()
