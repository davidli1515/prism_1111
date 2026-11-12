#!/usr/bin/env python3
"""Compute exact areas for ALREADY DETERMINED first-batch queries in parallel.

No model call, new candidate, policy change, or access to continuation CPI.
Each prefetched configuration must appear in the eventual query transcripts.
Writes only this experiment's exact area store, never original caches.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import time

from diagnose import (BENCHES, DATA, V5, LIMIT, AreaStore, config_key, get_context,
                      load, proposals, save, sha)


def main(out, workers):
    plan = load(out / "plan.json")
    for path, expected in plan["source_sha256"].items():
        assert sha(path) == expected
    data = {b: load(DATA / b / "s42/scalar_guard.json") for b in BENCHES}
    remaining = [c for c in plan["contexts"] if not (out/f"context_{c['id']}.json").exists()]
    # Leave the actively processed context alone to minimize duplicate work.
    pending = []
    for ctx in remaining[1:]:
        d = data[ctx["bench"]]
        known, f = get_context(ctx, d)
        for candidate in ctx["candidates"]:
            i = candidate["index"]
            current = {"config": d["configs"][i], "cpi": d["cpis"][i], "area": d["areas"][i]}
            pending += [{"context": ctx["id"], "bench": ctx["bench"], "config": p["config"]}
                        for p in proposals(current, ctx, d, f, known, 5)]
    existing = [load(p) for p in (out/"evaluations").glob("*.json")]
    keys = {(r["bench"], config_key(r["config"])) for r in existing+pending}
    assert len(keys) <= LIMIT
    save(out/"area_prefetch_plan.json", {"workers": workers, "candidates": pending,
                                         "note": "Exactly frozen first-batch proposals; no new policy queries."})
    area = AreaStore(out/"area_store")
    seed = sqlite3.connect(f"file:{V5/'area_store_v1/areas.sqlite'}?mode=ro", uri=True)
    todo = []
    for row in pending:
        c = row["config"]
        key = area.key(c)
        if area.lookup(key) is not None or seed.execute("SELECT value FROM areas WHERE key=?", (key,)).fetchone():
            continue
        todo.append(c)
    start = time.monotonic()
    area.prefetch(todo, workers=workers)
    area.flush()
    result = {"planned_config_records": len(pending), "unique_missing_area_keys": len({area.key(c) for c in todo}),
              "failures": area.n_fail, "seconds": time.monotonic()-start}
    save(out/"area_prefetch_result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    main(a.out, a.workers)
