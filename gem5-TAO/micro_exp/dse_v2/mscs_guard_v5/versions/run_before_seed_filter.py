#!/usr/bin/env python3
"""One benchmark/seed, one fresh BO start, three independent logical budgets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np

from engine import (HERE, METHODS, SliceOracle, BudgetOracle, BudgetExhausted, AtomicArchive,
                    refine, pareto_mask, hypervolume)
from area_store import AreaStore
from common import DEFAULT_CKPT, SPEC17
from run_dse_v3_single import run_bo_par


def save(path, data):
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, allow_nan=False))
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True, choices=SPEC17)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total-queries", type=int, default=5000)
    ap.add_argument("--bo-queries", type=int, default=500)
    ap.add_argument("--mcpat-workers", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not 30 <= args.bo_queries < args.total_queries:
        ap.error("Require 30 <= bo_queries < total_queries")
    target = args.out / args.bench / f"s{args.seed}"
    target.mkdir(parents=True, exist_ok=True)
    if any(target.glob("*.json")):
        raise FileExistsError(f"Refusing to overwrite/resume an existing run: {target}")
    # Never use the historically raced cache or work directory.
    area = AreaStore(HERE / "area_store_v1")
    physical = SliceOracle(args.bench)
    prefetch = lambda cs: area.prefetch(cs, workers=args.mcpat_workers)
    bo = AtomicArchive(BudgetOracle(physical, args.bo_queries), area)
    t0 = time.time()
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    print(f"START {args.bench} seed={args.seed} BO={args.bo_queries} total={args.total_queries}", flush=True)
    # The legacy initializer requests 90 areas before accepting 30 points. Only
    # prefetch the first 30; any failed-area replacements are computed on demand.
    bo_prefetch = lambda cs: prefetch(cs[:30] if not bo.cpis else cs)
    try:
        run_bo_par(bo.S, bo, args.seed, args.bo_queries, bo_prefetch)
    except BudgetExhausted:
        if bo.S.n_queries != args.bo_queries:
            raise
    if not bo.cpis:
        raise RuntimeError(f"No valid BO areas: {area.errors}")
    ref = [max(bo.cpis)*1.1, max(bo.areas)*1.1]
    bo_hv = hypervolume(bo.points(), ref)
    metadata = {
        "protocol": "v5.0_conditional_joint_actions", "bench": args.bench, "seed": args.seed,
        "budget": args.total_queries, "bo_queries": args.bo_queries, "reference": ref,
        "reference_rule": "1.1 times componentwise maxima of shared BO archive; frozen before refinement",
        "initial_hv": bo_hv, "checkpoint": DEFAULT_CKPT,
        "checkpoint_sha256": hashlib.sha256(Path(DEFAULT_CKPT).read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (HERE / "engine.py", HERE / "run.py", HERE / "area_store.py")},
        "area_sources": area.sources,
        "note": "Surrogate-only. Identical 50 slices and frozen checkpoint. Logical distinct queries "
                "include BO, all future references and all joint checks, even physical cache hits "
                "from another arm. Every area invocation has a private work directory. "
                "Guarded arms differ only in ranking (per-slice CPI P90 / pressure-composition L1). "
                "No permanent resource protection. Joint choices are evaluated configurations."
    }
    save(target / "initial.json", dict(metadata, **bo.result(), wall_seconds=time.time()-t0))
    for method in METHODS:
        start = time.time()
        arch = bo.clone(BudgetOracle(physical, args.total_queries))
        def progress(message):
            print(f"{args.bench} s{args.seed} {message}", flush=True)
            p = arch.points()
            checkpoints = sorted(set([args.bo_queries] + list(range(1000, arch.S.n_queries+1, 500))))
            data = {"bench": args.bench, "seed": args.seed, "method": method,
                    "status": "running", "queries": arch.S.n_queries, "budget": args.total_queries,
                    "curve": [{"query": q, "hv": hypervolume(p[np.asarray(arch.queries) <= q], ref)}
                              for q in checkpoints], "seconds": time.time()-start}
            path = target / f"{method}_progress.json"
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(data, allow_nan=False))
            temp.replace(path)
        detail = refine(arch, method, args.seed, prefetch,
                        log=progress)
        assert arch.S.n_queries == args.total_queries
        assert len(arch.configs) + len({tuple(sorted(x["config"].items())) for x in arch.failed_area}) >= arch.S.n_queries
        P = arch.points()
        curve = []
        for q in sorted(set([args.bo_queries, args.total_queries] + list(range(1000, args.total_queries, 500)))):
            if q < args.bo_queries:
                continue
            mask = np.asarray(arch.queries) <= q
            curve.append({"query": q, "hv": hypervolume(P[mask], ref)})
        hv = hypervolume(P, ref)
        result = dict(metadata, method=method, **arch.result(), **detail,
                      n_queries=arch.S.n_queries, final_hv=hv, curve=curve,
                      pareto_indices=np.flatnonzero(pareto_mask(P)).tolist(),
                      physical_queries_cumulative=physical.n_queries,
                      wall_seconds=time.time()-start, mcpat_failures_cumulative=area.n_fail,
                      area_errors=area.errors)
        save(target / f"{method}.json", result)
        area.flush()
        print(f"DONE {args.bench} s{args.seed} {method} q={arch.S.n_queries} "
              f"HV={hv:.6f} gain_from_BO={(hv/bo_hv-1)*100:.3f}% "
              f"seconds={time.time()-start:.1f}", flush=True)
    print(f"COMPLETE {args.bench} s{args.seed} wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
