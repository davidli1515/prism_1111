#!/usr/bin/env python3
"""Paired, fixed-budget HV comparison. Missing arms never count as ties."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from engine import METHODS, hypervolume, pareto_mask


def hv_attribution(points, tags, ref):
    """Descriptive marginal HV by first-query type; dependent on query order."""
    frontier, previous = np.empty((0, 2)), 0.
    gains = defaultdict(float)
    for point, tag in zip(points, tags):
        if np.any(np.all(frontier <= point, axis=1)):
            continue
        updated = np.vstack([frontier, point])
        frontier = updated[pareto_mask(updated)]
        current = hypervolume(frontier, ref)
        gains[tag] += current-previous
        previous = current
    return dict(gains)


def summarize(root):
    rows, incomplete = [], []
    for directory in sorted(root.glob("*/s*")):
        if not directory.is_dir():
            continue
        files = [directory / f"{m}.json" for m in METHODS]
        if not all(p.exists() for p in files):
            incomplete.append(str(directory))
            continue
        dd = [json.loads(p.read_text()) for p in files]
        assert len({d["checkpoint_sha256"] for d in dd}) == 1
        assert len({json.dumps(d["source_sha256"], sort_keys=True) for d in dd}) == 1
        assert len({tuple(d["reference"]) for d in dd}) == 1
        assert len({d["n_queries"] for d in dd}) == 1
        assert all(d["n_queries"] == d["budget"] for d in dd)
        hv, extra = {}, {}
        guarded_steps = {}
        for d in dd:
            p = np.c_[d["cpis"], d["areas"]]
            hv[d["method"]] = hypervolume(p, d["reference"])
            assert np.isclose(hv[d["method"]], d["final_hv"])
            steps = [s for w in d["walks"] for s in w["steps"]]
            checks = [c for s in steps for c in s.get("checks", [])]
            if d["method"] != "scalar":
                guarded_steps[d["method"]] = {
                    tuple(sorted(d["configs"][s["intermediate"]].items())): s
                    for s in steps if "intermediate" in s}
            extra[d["method"]] = {
                "points": len(p), "area_failure_queries": d["n_queries"]-len(p),
                "query_tags": dict(Counter(d["tags"])), "completed_steps": len(steps),
                "hv_increments_by_first_query_tag": hv_attribution(p, d["tags"], d["reference"]),
                "joint_checks": len(checks), "risky_checks": sum(c["blocked"] for c in checks),
                "joint_steps_taken": sum(s.get("choice") == "joint" for s in steps),
                "steps_with_more_than_three_guard_resources": sum(
                    s.get("eligible_guard_resources", 0) > 3 for s in steps),
                "min_cpi": float(p[:, 0].min()), "min_area": float(p[:, 1].min()),
            }
        a, b = guarded_steps["scalar_guard"], guarded_steps["mscs_guard"]
        shared_states = set(a) & set(b)
        signature = lambda s: {(d["param"], d["stride"]) for d in s.get("selected_reclaims", [])}
        selection_differences = sum(signature(a[k]) != signature(b[k]) for k in shared_states)
        rows.append({"bench": dd[0]["bench"], "seed": dd[0]["seed"], "hv": hv,
                     "guard_shared_intermediate_states": len(shared_states),
                     "guard_different_selected_sets_at_shared_states": selection_differences,
                     "initial_hv": dd[0]["initial_hv"], "budget": dd[0]["budget"],
                     "mscs_vs_scalar_pct": 100*(hv["mscs_guard"]/hv["scalar"]-1),
                     "mscs_vs_scalar_guard_pct": 100*(hv["mscs_guard"]/hv["scalar_guard"]-1),
                     "details": extra})
    by_bench = defaultdict(list)
    for r in rows:
        by_bench[r["bench"]].append(r)
    agg = {}
    for metric in ("mscs_vs_scalar_pct", "mscs_vs_scalar_guard_pct"):
        x = np.array([r[metric] for r in rows])
        agg[metric] = {"mean": float(x.mean()) if len(x) else None,
                       "median": float(np.median(x)) if len(x) else None,
                       "wins": int((x > 1e-9).sum()), "ties": int((abs(x) <= 1e-9).sum()),
                       "losses": int((x < -1e-9).sum())}
    scope_path = root / "queue_effective.json"
    scope = json.loads(scope_path.read_text()) if scope_path.exists() else None
    return {"effective_scope": scope,
            "completed_pairs": len(rows), "incomplete": incomplete, "aggregate": agg,
            "by_bench": {b: {"seeds": len(rr), **{metric: float(np.mean([r[metric] for r in rr]))
                         for metric in agg}} for b, rr in by_bench.items()}, "rows": rows,
            "limitations": "Surrogate only. Paired percentages use a BO-only reference fixed before "
                            "refinement and shared across arms within each seed. No gem5 validation, "
                            "no claim versus full-budget BO/random, no statistical significance claim. "
                            "Arm wall times are not comparable because physical caches are reused. "
                            "Risky-check counts concern only selected candidates, not risk recall."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    out = summarize(args.data)
    target = args.data / "summary.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2, allow_nan=False))
    tmp.replace(target)
    print(json.dumps({k: out[k] for k in ("completed_pairs", "aggregate", "by_bench")}, indent=2))


if __name__ == "__main__":
    main()
