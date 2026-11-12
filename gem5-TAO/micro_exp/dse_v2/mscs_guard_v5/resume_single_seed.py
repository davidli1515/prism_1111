#!/usr/bin/env python3
"""Recover missing arms using saved BO data and exactly the frozen search engine.

Incomplete walks are deterministically replayed, with cached McPAT values. The
old intermediate HV checkpoints must match before new progress is accepted.
Completed arms and all original progress files are preserved.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from run_control import skip_cancelled_run
if __name__ == "__main__" and skip_cancelled_run(sys.argv[1:]):
    raise SystemExit(0)

import numpy as np
from engine import (HERE, METHODS, SliceOracle, BudgetOracle, AtomicArchive,
                    config_key, refine, hypervolume, pareto_mask)
from area_store import AreaStore
from run import save


def restore_initial(initial, physical, area):
    S = BudgetOracle(physical, initial["budget"])
    arch = AtomicArchive(S, area)
    for name in ("configs", "cpis", "areas", "queries", "tags", "failed_area"):
        setattr(arch, name, list(initial[name]))
    arch.indices = {config_key(c): i for i, c in enumerate(arch.configs)}
    configs = arch.configs + [r["config"] for r in arch.failed_area]
    S.seen = {config_key(c): dict(c) for c in configs}
    if S.n_queries != initial["bo_queries"]:
        raise ValueError("Saved BO ledger does not reconstruct its original query count")
    if len({len(getattr(arch, n)) for n in ("configs", "cpis", "areas", "queries", "tags")}) != 1:
        raise ValueError("Misaligned saved BO archive")
    return arch


def check_replay(curve, previous):
    new = {r["query"]: r["hv"] for r in curve}
    checked = []
    for r in previous:
        if r["query"] in new:
            if not np.isclose(new[r["query"]], r["hv"], rtol=1e-10, atol=1e-10):
                raise ValueError(f"Replay diverged at q={r['query']}: {new[r['query']]} vs {r['hv']}")
            checked.append(r["query"])
    return checked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mcpat-workers", type=int, default=2)
    args = ap.parse_args()
    target = args.out / args.bench / f"s{args.seed}"
    lock = (target / "recovery.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = json.loads((target / "initial.json").read_text())
    for filename in ("engine.py", "area_store.py"):
        if hashlib.sha256((HERE / filename).read_bytes()).hexdigest() != initial["source_sha256"][filename]:
            raise ValueError(f"Original search source changed: {filename}")
    if hashlib.sha256(Path(initial["checkpoint"]).read_bytes()).hexdigest() != initial["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    area = AreaStore(HERE / "area_store_v1")
    if area.sources != initial["area_sources"]:
        raise ValueError("McPAT inputs changed")
    physical = SliceOracle(args.bench)
    prefetch = lambda cs: area.prefetch(cs, workers=args.mcpat_workers)
    excluded = {"configs", "cpis", "areas", "queries", "tags", "failed_area", "wall_seconds"}
    metadata = {k: v for k, v in initial.items() if k not in excluded}
    ref, budget = initial["reference"], initial["budget"]
    for method in METHODS:
        output = target / f"{method}.json"
        if output.exists():
            saved = json.loads(output.read_text())
            if saved["n_queries"] != budget:
                raise ValueError(f"Unexpected incomplete final file: {output}")
            print(f"KEEP {args.bench} {method} q={budget}", flush=True)
            continue
        arch = restore_initial(initial, physical, area)
        old = target / f"{method}_progress.json"
        old_curve = json.loads(old.read_text())["curve"] if old.exists() else []
        started = time.time()
        recovered = target / f"{method}_recovery_progress.json"
        def curve_at(q):
            P = arch.points()
            checkpoints = sorted(set([initial["bo_queries"]] + list(range(1000, q+1, 500))
                                     + ([budget] if q == budget else [])))
            return [{"query": n, "hv": hypervolume(P[np.asarray(arch.queries) <= n], ref)}
                    for n in checkpoints]
        def progress(message):
            curve = curve_at(arch.S.n_queries)
            verified = check_replay(curve, old_curve)
            state = {"bench": args.bench, "seed": args.seed, "method": method,
                     "status": "running", "queries": arch.S.n_queries, "budget": budget,
                     "curve": curve, "replay_verified_at": verified,
                     "seconds": time.time()-started, "updated_unix": time.time()}
            temp = recovered.with_suffix(".tmp")
            temp.write_text(json.dumps(state, allow_nan=False))
            os.replace(temp, recovered)
            print(f"{args.bench} {message} replay_verified={verified}", flush=True)
        print(f"RECOVER {args.bench} {method}: replay from BO q={arch.S.n_queries}", flush=True)
        progress("recovery started")
        detail = refine(arch, method, args.seed, prefetch, log=progress)
        assert arch.S.n_queries == budget
        assert len(arch.configs) + len({config_key(x["config"]) for x in arch.failed_area}) >= budget
        curve = curve_at(budget)
        verified = check_replay(curve, old_curve)
        assert len(verified) == len(old_curve)
        P = arch.points()
        result = dict(metadata, method=method, **arch.result(), **detail,
                      n_queries=arch.S.n_queries, final_hv=hypervolume(P, ref), curve=curve,
                      pareto_indices=np.flatnonzero(pareto_mask(P)).tolist(),
                      physical_queries_cumulative=physical.n_queries,
                      wall_seconds=time.time()-started, mcpat_failures_cumulative=area.n_fail,
                      area_errors=area.errors,
                      recovery={"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                "replay_verified_at": verified,
                                "note": "Saved BO restored; same frozen search replayed with exact area cache. "
                                        "Original progress and completed arms retained; logical budget unchanged."})
        save(output, result)
        area.flush()
        print(f"DONE {args.bench} {method} q={budget} HV={result['final_hv']:.9f}", flush=True)


if __name__ == "__main__":
    main()
