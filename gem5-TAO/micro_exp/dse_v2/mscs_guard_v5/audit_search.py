#!/usr/bin/env python3
"""Read-only audit of where MSCS ordering can affect search and what it buys.

Uses ONLY saved CPI/area/trace data. No inference, no McPAT, no new DSE runs.
Shared-state comparisons are descriptive: histories may differ even when the
starting configuration, intermediate state and candidate-count fields match.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from engine import config_key, pareto_mask, hypervolume


def steps(d):
    return [s for w in d["walks"] for s in w["steps"] if "intermediate" in s]


def incremental_hv(d):
    points = np.c_[d["cpis"], d["areas"]]
    front, prev = np.empty((0, 2)), 0.
    increments = np.zeros(len(points))
    for i, point in enumerate(points):
        if len(front) and np.any(np.all(front <= point, axis=1)):
            continue
        p = np.vstack([front, point]); front = p[pareto_mask(p)]
        hv = hypervolume(front, d["reference"])
        increments[i], prev = max(0., hv-prev), hv
    assert np.isclose(prev, d["final_hv"])
    return increments


def describe(d):
    ss = steps(d)
    inc = incremental_hv(d)
    gains, counts = defaultdict(float), Counter()
    for tag, q, g in zip(d["tags"], d["queries"], inc):
        if q > d["bo_queries"]:
            counts[tag] += 1; gains[tag] += float(g)
    checks = [r for s in ss for r in s.get("checks", [])]
    checked = {r["index"]: r for r in checks}
    risky = {i: r for i, r in checked.items() if r["blocked"]}
    frontier = set(d["pareto_indices"])
    active = [s for s in ss if s["selected_reclaims"]]
    gain = d["final_hv"]-d["initial_hv"]
    positive = lambda i: bool(inc[i] > 1e-10*max(1., d["final_hv"]))
    return {"method": d["method"], "steps": len(ss), "active_guard_steps": len(active),
            "refine_query_tags": dict(counts), "refine_queries": d["n_queries"]-d["bo_queries"],
            "refine_hv_gain": gain, "marginal_hv_by_tag": dict(gains),
            "marginal_hv_share_by_tag": {k: v/gain if gain else None for k, v in gains.items()},
            "choice_counts": dict(Counter(s["choice"] for s in ss)),
            "guard_resource_count_histogram": dict(Counter(s["eligible_guard_resources"] for s in active)),
            "no_subset_or_stride_choice_steps": sum(s["eligible_guard_resources"] <= 3 and
                s["eligible_guard_actions"] == s["eligible_guard_resources"] for s in active),
            "unique_checked_configs": len(checked), "unique_risky_checked_configs": len(risky),
            "checked_configs_with_positive_hv_at_first_query": sum(positive(i) for i in checked),
            "risky_configs_with_positive_hv_at_first_query": sum(positive(i) for i in risky),
            "checked_configs_on_final_front": len(set(checked)&frontier),
            "risky_configs_on_final_front": len(set(risky)&frontier)}


def compare(a, b):
    assert a["reference"] == b["reference"]
    aa, bb = steps(a), steps(b)
    key = lambda d, s: (config_key(d["configs"][s["start"]]),
                        config_key(d["configs"][s["intermediate"]]), s["increase"])
    sa = {key(a, s): s for s in aa}; sb = {key(b, s): s for s in bb}
    kaa = [config_key(c) for c in a["configs"]]; kbb = [config_key(c) for c in b["configs"]]
    index_a = {k: i for i, k in enumerate(kaa)}
    P = np.c_[a["cpis"], a["areas"]]
    rows = []
    for k in sorted(set(sa)&set(sb)):
        x, y = sa[k], sb[k]
        selections = lambda s: {(r["param"], r["stride"]) for r in s["selected_reclaims"]}
        resources = lambda s: {r["param"] for r in s["selected_reclaims"]}
        futures = lambda d, s: {config_key(d["configs"][r["index"]]) for r in s.get("futures", [])}
        common_context = (futures(a, x) == futures(b, y) and
                          all(x[n] == y[n] for n in ("eligible_guard_actions", "eligible_guard_resources")))
        u, v = selections(x), selections(y)
        row = {"a_query_start": x["q_start"], "b_query_start": y["q_start"],
               "same_candidate_context_fields": common_context,
               "same_resource_set": resources(x) == resources(y), "same_action_set": u == v,
               "action_set_jaccard": len(u&v)/len(u|v) if u|v else 1.,
               "same_next_config": kaa[x["end"]] == kbb[y["end"]],
               "scalar_choice": x["choice"], "mscs_choice": y["choice"],
               "scalar_selected": sorted(u), "mscs_selected": sorted(v)}
        # Shared observed prefix before joint probes. Comparing joint batches on
        # this common reference removes differences in the pre-existing fronts.
        cut_a = x["q"]-x["joint_queries"]; cut_b = y["q"]-y["joint_queries"]
        prefix_a = {c for c, q in zip(kaa, a["queries"]) if q <= cut_a}
        prefix_b = {c for c, q in zip(kbb, b["queries"]) if q <= cut_b}
        shared = prefix_a & prefix_b
        front = P[[index_a[c] for c in shared]]
        front = front[pareto_mask(front)]
        h0 = hypervolume(front, a["reference"])
        batches = [np.array([[r["cpi"], r["area"]] for r in s["checks"]], float).reshape(-1, 2)
                   for s in (x, y)]
        hvi = lambda p: max(0., hypervolume(np.vstack([front, p]), a["reference"])-h0)
        row.update(scalar_batch_hvi=hvi(batches[0]), mscs_batch_hvi=hvi(batches[1]),
                   union_batch_hvi=hvi(np.vstack(batches)),
                   scalar_checks=len(batches[0]), mscs_checks=len(batches[1]))
        rows.append(row)
    strict = [r for r in rows if r["same_candidate_context_fields"]]
    return {"matched_start_and_intermediate_states": len(rows),
            "matching_candidate_context_fields": len(strict),
            "different_resource_sets": sum(not r["same_resource_set"] for r in strict),
            "different_action_sets": sum(not r["same_action_set"] for r in strict),
            "different_next_configurations": sum(not r["same_next_config"] for r in strict),
            "mean_action_set_jaccard": float(np.mean([r["action_set_jaccard"] for r in strict])) if strict else None,
            "states_with_no_joint_HV_opportunity_in_measured_union": sum(r["union_batch_hvi"] < 1e-9 for r in strict),
            "scalar_batch_HVI_wins": sum(r["scalar_batch_hvi"] > r["mscs_batch_hvi"]+1e-9 for r in strict),
            "mscs_batch_HVI_wins": sum(r["mscs_batch_hvi"] > r["scalar_batch_hvi"]+1e-9 for r in strict),
            "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    cases = []
    for p in sorted(args.data.glob("*/s42/scalar_guard.json")):
        a = json.loads(p.read_text())
        row = {"bench": a["bench"], "scalar_guard": describe(a)}
        other = p.with_name("mscs_guard.json")
        if other.exists():
            b = json.loads(other.read_text())
            row.update(mscs_guard=describe(b), comparison=compare(a, b),
                       mscs_vs_scalar_guard_pct=100*(b["final_hv"]/a["final_hv"]-1))
        cases.append(row)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "cases": cases,
              "limits": "Read-only, no new labels. Marginal HV attribution depends on query order. "
                        "State matches are descriptive, not randomized counterfactuals; visited histories may differ. "
                        "Zero joint-batch HVI does not rule out later path value. The measured union is not "
                        "the entire action space. All results are from a frozen surrogate, single seed."}
    serialized = json.dumps(result, indent=2, allow_nan=False)
    with args.out.open("x") as f:
        f.write(serialized)
    for c in cases:
        print(json.dumps({k: ({n: v for n, v in value.items() if n != "rows"} if k == "comparison" else value)
                          for k, value in c.items()}))


if __name__ == "__main__":
    main()
