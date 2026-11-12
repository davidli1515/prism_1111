#!/usr/bin/env python3
"""Resource-level equal-budget verification, with scalar/config/size controls.

One unit of budget checks ONE reclaim under exactly THREE future directions.
Retain only groups with three partners, so every selection costs three joint
queries for every policy. Risk = max future CPI cost above tau over the partners.
Selection scores use ONLY the original point and pre-joint observations.
Report captured risk, avoiding misleading inference from magnitude of MSCS alone.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from collect import PARAMS, get_param_values


def collect_groups(data_paths, learned_path=None):
    learned = {}
    if learned_path:
        for p in json.loads(Path(learned_path).read_text())["predictions"]:
            key = (p["bench"], p["state"], p["j"], p["j_stride"], p["k"], p["k_stride"])
            learned[key] = p["scalar_learned"]
    out = []
    for path in data_paths:
        data = json.loads(path.read_text())
        if "states" not in data:
            continue
        for state in data["states"]:
            grouped = defaultdict(list)
            for row in state["rows"]:
                if row["current_cost"] >= 0:
                    grouped[(row["j"], row["j_stride"])].append(row)
            actions = []
            for (j, stride), rows in grouped.items():
                if len(rows) != 3 or len({r["k"] for r in rows}) != 3:
                    continue
                r = rows[0]
                cfg, cj = state["config"], r["cfg_j"]
                d = np.array([abs(cj[p]-cfg[p]) for p in PARAMS], float)
                span = np.array([max(get_param_values(p))-min(get_param_values(p)) for p in PARAMS])
                scores = {
                    "scalar_cost": r["current_cost"],
                    "scalar_add": max(t["prediction"]["scalar_add"] for t in rows),
                    "mscs_l1": r["prediction"]["l1"],
                    "composition_l1": r["prediction"]["composition_l1"],
                    "config_range_l1": float((d/span).sum()),
                    "config_relative_l1": float((d/np.maximum([cfg[p] for p in PARAMS], 1)).sum()),
                    "stride": float(stride),
                    # Per-slice scalar controls: offsets follow scalar_stats in collector.
                    "scalar_slice_std": r["scalar_detail"][6],
                    "scalar_slice_p90": r["scalar_detail"][9],
                }
                if learned_path:
                    scores["scalar_learned"] = max(learned[(data["bench"], state["state"], j, stride,
                                                           t["k"], t["k_stride"])] for t in rows)
                actions.append({"j": j, "stride": stride,
                                "risk": max(0., max(t["future_cost"] for t in rows)-data["args"]["tau"]),
                                "interaction": max(t["interaction"] for t in rows), "scores": scores})
            if actions:
                out.append({"bench": data["bench"], "state": state["state"], "actions": actions})
    return out


def evaluate(groups):
    results = []
    for group in groups:
        aa = group["actions"]
        risk = np.array([a["risk"] for a in aa])
        if not np.any(risk > 0):
            continue
        row = {"bench": group["bench"], "state": group["state"], "actions": len(aa), "metrics": {}}
        for method in aa[0]["scores"]:
            scores = np.array([a["scores"][method] for a in aa])
            # Expected capture under UNIFORM tie-breaking, not parameter enumeration.
            # Average 200 pre-fixed jitter tie-breaks; never access labels to break ties.
            rng = np.random.RandomState(1008)
            orders = [np.lexsort((rng.uniform(size=len(aa)), -scores)) for _ in range(200)]
            row["metrics"][method] = {}
            for k in (1, 3):
                oracle = np.sort(risk)[-k:].sum()
                row["metrics"][method][str(k)] = float(np.mean([risk[o[:k]].sum()/oracle for o in orders]))
        results.append(row)
    summary = {method: {str(k): float(np.mean([r["metrics"][method][str(k)] for r in results]))
                        for k in (1, 3)} for method in results[0]["metrics"]} if results else {}
    return {"states": len(groups), "risky_states": len(results), "summary": summary, "rows": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--learned")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    target = Path(args.out)
    if target.exists():
        ap.error("Refusing to overwrite")
    groups = collect_groups(sorted(Path(args.data).glob("*.json")), args.learned)
    result = evaluate(groups)
    result["by_bench"] = {b: evaluate([g for g in groups if g["bench"] == b])
                          for b in sorted({g["bench"] for g in groups})}
    result["note"] = ("Resource verification budget: k=1 costs 3 joint queries; k=3 costs 9, "
                     "or fewer if all eligible resources are exhausted, identically for every policy. "
                     "Only risky states contribute capture ratios; zero-risk states are counted separately. "
                     "No area, no HV, no claim that pressure amplitude equals resource value.")
    with target.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k in ("states", "risky_states", "summary")}, indent=2))


if __name__ == "__main__":
    main()
