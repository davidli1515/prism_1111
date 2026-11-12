#!/usr/bin/env python3
"""Bounded, frozen-checkpoint test of MSCS-guided branch allocation.

prepare uses saved scalar observations only; run freezes all selectors before
revealing continuation labels. No original search/model files are modified.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
V5 = HERE.parent / "mscs_guard_v5"
sys.path.insert(0, str(V5))
from engine import (SliceOracle, _shift, config_key, PARAMS, hypervolume,
                    pareto_mask, compute_w_min, cpi_samples)
from area_store import AreaStore
from common import DEFAULT_CKPT
from config_dse import ALL_PARAMS, get_param_values

BENCHES = ("623.xalancbmk_s", "641.leela_s")
DATA = V5 / "results_full_01"
METHODS = ("scalar_quality", "scalar_slice_diversity", "config_diversity", "mscs_diversity")
LIMIT = 600
BRANCH_QUERIES = 11


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, allow_nan=False, indent=2)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(data)
    tmp.rename(path)


def load(path):
    return json.loads(Path(path).read_text())


def front(points):
    p = np.asarray(points, float).reshape(-1, 2)
    return p[pareto_mask(p)] if len(p) else p


def hvi(points, known_front, ref):
    return max(0., hypervolume(np.vstack([known_front, np.asarray(points).reshape(-1, 2)]), ref)
               - hypervolume(known_front, ref))


def point(d, i):
    return [d["cpis"][i], d["areas"][i]]


def steps(d):
    return [s for w in d["walks"] for s in w["steps"] if "intermediate" in s]


def near_quality(p):
    p = np.asarray(p)
    return bool(np.max(p[:, 0])/np.min(p[:, 0]) <= 1.01 and
                np.max(p[:, 1])/np.min(p[:, 1]) <= 1.02)


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    sources, contexts, counts = {}, [], {}
    for bench in BENCHES:
        paths = [DATA / bench / "s42" / f"{m}.json" for m in ("scalar_guard", "mscs_guard")]
        a, b = map(load, paths)
        sources.update({str(p): sha(p) for p in paths})
        assert a["checkpoint_sha256"] == b["checkpoint_sha256"] == sha(DEFAULT_CKPT)
        index = {config_key(c): i for i, c in enumerate(a["configs"])}
        key = lambda d, s: (config_key(d["configs"][s["start"]]),
                            config_key(d["configs"][s["intermediate"]]))
        shared = {key(b, s) for s in steps(b)}
        eligible = []
        for t, s in enumerate(steps(a)):
            if key(a, s) not in shared:
                continue
            c = a["configs"][s["start"]]
            neighbors = []
            for param in PARAMS:
                cn = _shift(c, param, 1)
                if cn is None:
                    continue
                i = index.get(config_key(cn))
                if i is None or a["queries"][i] > s["q"]:
                    raise ValueError("Saved increase neighborhood is incomplete")
                neighbors.append({"param": param, "index": i})
            cutoff = max([s["q_start"]] + [a["queries"][n["index"]] for n in neighbors])
            # All +1 probes precede all new reclaim/future/joint probes.
            assert cutoff <= s["q"]-s["future_queries"]-s["joint_queries"]
            prefix = [i for i, q in enumerate(a["queries"]) if q <= s["q_start"]]
            f = front([point(a, i) for i in prefix])
            base = np.array(point(a, s["start"]))
            rank = lambda n: (-hvi([point(a, n["index"])], f, a["reference"]),
                              float(np.prod(np.array(point(a, n["index"]))/base)), n["param"])
            pool = sorted([n for n in neighbors if n["param"] != "l1_size"], key=rank)
            if not pool:
                continue
            chosen = [pool[0]]
            for n in pool[1:]:
                if any(n["index"] == x["index"] for x in chosen):
                    continue
                if near_quality([point(a, x["index"]) for x in chosen+[n]]):
                    chosen.append(n)
                if len(chosen) == 4:
                    break
            if len(chosen) != 4:
                continue
            eligible.append({"bench": bench, "step": t, "q_start": s["q_start"],
                             "cutoff": cutoff, "root": s["start"], "candidates": chosen,
                             "neighbors": neighbors, "reference": a["reference"],
                             "source": str(paths[0])})
        counts[bench] = {"shared_states": sum(key(a, s) in shared for s in steps(a)),
                         "eligible_states": len(eligible)}
        for i in np.linspace(0, len(eligible)-1, min(6, len(eligible)), dtype=int):
            contexts.append(dict(eligible[int(i)], id=f"{bench}_{len(contexts):02d}"))
    sources[str(Path(DEFAULT_CKPT))] = sha(DEFAULT_CKPT)
    sources[str(Path(__file__))] = sha(__file__)
    sources[str(HERE / "PROTOCOL.txt")] = sha(HERE / "PROTOCOL.txt")
    for p in (V5 / "engine.py", V5 / "area_store.py",
              HERE.parent / "mscs_response_1008/probe.py",
              HERE.parent / "mscs_response_1008/response.py"):
        sources[str(p)] = sha(p)
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
            "limit": LIMIT, "max_requested_configs": len(contexts)*(5+4*BRANCH_QUERIES),
            "source_sha256": sources, "counts": counts, "contexts": contexts}
    assert plan["max_requested_configs"] <= LIMIT
    save(out / "plan.json", plan)
    print(json.dumps({"counts": counts, "contexts": len(contexts),
                      "max_requested_configs": plan["max_requested_configs"]}), flush=True)


def pressure_residual(m):
    """Keep signed within-stage allocation; remove the CPI-related pressure sum."""
    p = np.asarray(m, float).reshape(-1, 4, 4)[..., 1:]
    return p-p.mean(axis=-1, keepdims=True)


def config_vector(c):
    # Listed choices are ordinal bins, numeric ranges are linear. Actual projected
    # configurations, including collateral changes, determine these coordinates.
    out = []
    for name, spec in ALL_PARAMS.items():
        values = get_param_values(name)
        out.append((values.index(c[name])/(len(values)-1)) if isinstance(spec, list)
                   else (c[name]-values[0])/(values[-1]-values[0]))
    return np.array(out)


def diversity_pair(vectors):
    """Retain the strongest scalar candidate; maximize response distance to it."""
    v = np.asarray(vectors, float).reshape(4, -1)
    distances = np.sqrt(np.mean((v-v[0])**2, axis=1))
    partner = min(range(1, 4), key=lambda j: (-float(distances[j]), j))
    return [0, partner], distances.tolist()


def selectors(root_m, candidate_ms, root_c, candidate_cs):
    ids = [compute_w_min(c) for c in [root_c]+candidate_cs]
    root_cpi = cpi_samples(np.asarray(root_m), ids[0])
    cpi_response = np.array([np.log(cpi_samples(np.asarray(m), w)/root_cpi)
                             for m, w in zip(candidate_ms, ids[1:])])
    m_response = np.array([pressure_residual(m)-pressure_residual(root_m) for m in candidate_ms])
    x_response = np.array([config_vector(c)-config_vector(root_c) for c in candidate_cs])
    pairs, distances = {"scalar_quality": [0, 1]}, {}
    for name, vectors in (("scalar_slice_diversity", cpi_response),
                          ("config_diversity", x_response), ("mscs_diversity", m_response)):
        pairs[name], distances[name] = diversity_pair(vectors)
    return {"pairs": pairs, "distances_from_scalar_anchor": distances}


class Evaluator:
    """Only this object may reveal saved future labels or invoke the model/area."""
    def __init__(self, out, datasets, area_sources):
        self.out, self.datasets = out, datasets
        self.directory = out / "evaluations"
        self.directory.mkdir(exist_ok=True)
        self.n_requests = len(list(self.directory.glob("*.json")))
        self.oracles, self.index = {}, {}
        for bench, d in datasets.items():
            self.index[bench] = {config_key(c): i for i, c in enumerate(d["configs"])}
        self.area = AreaStore(out / "area_store")
        assert self.area.sources == area_sources
        self.seed_area = sqlite3.connect(f"file:{V5/'area_store_v1/areas.sqlite'}?mode=ro", uri=True)
        self.events = out / "events.jsonl"

    def query(self, bench, c, need_vector=False, tag=""):
        digest = hashlib.sha256(json.dumps([bench, c], sort_keys=True).encode()).hexdigest()
        path = self.directory / f"{digest}.json"
        if path.exists():
            record = load(path)
            if need_vector and record["mscs"] is None:
                raise ValueError("Feature extraction must precede continuation labels")
            return record
        if self.n_requests >= LIMIT:
            raise RuntimeError("Hard global evaluation limit reached")
        self.n_requests += 1
        t0 = time.monotonic()
        d = self.datasets[bench]
        old = self.index[bench].get(config_key(c))
        m, ids, cpi = None, None, None
        if need_vector or old is None:
            if bench not in self.oracles:
                self.oracles[bench] = SliceOracle(bench)
            m_, cpi, ids = self.oracles[bench].query(c)
            if need_vector:
                m = m_.tolist()
            if old is not None and not np.isclose(cpi, d["cpis"][old], rtol=3e-6):
                raise ValueError("Frozen CPI drift")
        if old is not None:
            cpi, area = d["cpis"][old], d["areas"][old]
            source = "saved_v5"
        else:
            k = self.area.key(c)
            hit = self.seed_area.execute("SELECT value FROM areas WHERE key=?", (k,)).fetchone()
            area = hit[0] if hit else self.area(c)
            source = "new_cpi_cached_area" if hit else "new_cpi_new_area_store"
        record = {"bench": bench, "config": c, "cpi": float(cpi), "area": area,
                  "mscs": m, "slice_ids": list(ids) if ids is not None else None,
                  "source": source, "tag": tag, "seconds": time.monotonic()-t0}
        save(path, record)
        with self.events.open("a") as f:
            f.write(json.dumps({k: v for k, v in record.items() if k not in ("mscs", "config")})+"\n")
        return record


def get_context(ctx, d):
    ids = [i for i, q in enumerate(d["queries"]) if q <= ctx["cutoff"]]
    known_keys = {config_key(d["configs"][i]) for i in ids}
    return known_keys, front([point(d, i) for i in ids])


def proposals(current, ctx, d, observed_front, excluded, count):
    """Identical scalar-only two-batch continuation for every candidate branch."""
    root_c = d["configs"][ctx["root"]]
    root_p = np.array(point(d, ctx["root"]))
    pool, unique = [], set()
    for n in ctx["neighbors"]:
        param, i = n["param"], n["index"]
        if param == "l1_size":
            continue
        dx = d["configs"][i][param]-root_c[param]
        if dx == 0:
            continue
        slope = (np.array(point(d, i))-root_p)/dx
        for stride in (1, -1, 4, -4):
            cn = _shift(current["config"], param, stride)
            if cn is None or config_key(cn) in excluded or config_key(cn) in unique:
                continue
            unique.add(config_key(cn))
            pred = np.array([current["cpi"], current["area"]])+slope*(cn[param]-current["config"][param])
            pred = np.maximum(pred, root_p*.01)
            rank = (-hvi([pred], observed_front, ctx["reference"]), float(np.prod(pred/root_p)),
                    PARAMS.index(param), (1, -1, 4, -4).index(stride))
            pool.append({"config": cn, "param": param, "stride": stride,
                         "rank": rank, "predicted_point": pred.tolist()})
    pool.sort(key=lambda r: r["rank"])
    chosen, params = [], set()
    for r in pool:
        if r["param"] not in params:
            chosen.append(r); params.add(r["param"])
        if len(chosen) == count:
            return chosen
    selected = {config_key(r["config"]) for r in chosen}
    return (chosen+[r for r in pool if config_key(r["config"]) not in selected])[:count]


def rollout(ctx, d, candidate, evaluator):
    known, initial_front = get_context(ctx, d)
    i = candidate["index"]
    current = {"config": d["configs"][i], "cpi": d["cpis"][i], "area": d["areas"][i]}
    seen, records, options = set(known), [], [current]
    observed = initial_front
    for round_i, count in enumerate((5, 6)):
        batch = proposals(current, ctx, d, observed, seen, count)
        for p in batch:
            c = p["config"]
            assert config_key(c) not in seen
            # This call is the only point where a continuation's true values are revealed.
            r = evaluator.query(ctx["bench"], c, tag="continuation")
            seen.add(config_key(c))
            records.append({"config": c, "cpi": r["cpi"], "area": r["area"],
                            "param": p["param"], "stride": p["stride"], "round": round_i,
                            "source": r["source"]})
            if r["area"] is not None:
                options.append(records[-1])
        observed = front(np.vstack([initial_front, [[r["cpi"], r["area"]] for r in options]]))
        base = np.array(point(d, ctx["root"]))
        current = min(options, key=lambda r: (-hvi([[r["cpi"], r["area"]]], initial_front, ctx["reference"]),
                                              float(np.prod(np.array([r["cpi"], r["area"]])/base))))
    return records


def merged_stream(a, b):
    """Round-robin independent branch transcripts, charging distinct configs once."""
    result, seen = [], set()
    for i in range(max(len(a), len(b))):
        for branch in (a, b):
            if i < len(branch):
                row = branch[i]
                key = config_key(row["config"])
                if key not in seen:
                    result.append(row); seen.add(key)
    return result


def assess(ctx, d, branches, decision):
    _, f = get_context(ctx, d)
    streams = {f"{i},{j}": merged_stream(branches[i], branches[j]) for i, j in combinations(range(4), 2)}
    budget = min(20, min(map(len, streams.values())))
    scores = {}
    for pair, rows in streams.items():
        p = [[r["cpi"], r["area"]] for r in rows[:budget] if r["area"] is not None]
        scores[pair] = hvi(p, f, ctx["reference"])
    methods = {name: scores[",".join(map(str, pair))] for name, pair in decision["pairs"].items()}
    anchored = [scores[f"0,{j}"] for j in range(1, 4)]
    methods.update(random_anchor=float(np.mean(anchored)), random_all=float(np.mean(list(scores.values()))))
    return {"context": ctx["id"], "bench": ctx["bench"], "query_budget": budget,
            "known_hv": hypervolume(f, ctx["reference"]), "pair_hvi": scores,
            "method_hvi": methods, "oracle_hvi": max(scores.values()),
            "anchor_oracle_hvi": max(anchored), "branches": branches,
            "area_failures": sum(r["area"] is None for b in branches for r in b)}


def summarize(out):
    results = [load(p) for p in sorted(out.glob("context_*.json"))]
    decisions = {d["id"]: d for d in load(out / "decisions.json")["contexts"]}
    summary = {}
    for bench in [*BENCHES, "all"]:
        rs = [r for r in results if (bench == "all" or r["bench"] == bench) and not r["area_failures"]]
        active = [r for r in rs if r["oracle_hvi"] > 1e-9]
        names = [*METHODS, "random_anchor", "random_all"]
        methods = {}
        for name in names:
            methods[name] = {"sum_hvi": sum(r["method_hvi"][name] for r in rs),
                             "mean_oracle_capture": float(np.mean([r["method_hvi"][name]/r["oracle_hvi"]
                                                                    for r in active])) if active else None}
        comparisons = {}
        for name in names:
            if name == "mscs_diversity":
                continue
            diff = [r["method_hvi"]["mscs_diversity"]-r["method_hvi"][name] for r in rs]
            comparisons[name] = {"wins": sum(x > 1e-9 for x in diff),
                                 "ties": sum(abs(x) <= 1e-9 for x in diff),
                                 "losses": sum(x < -1e-9 for x in diff),
                                 "mean_normalized_hvi_difference": float(np.mean([
                                     (r["method_hvi"]["mscs_diversity"]-r["method_hvi"][name])/r["oracle_hvi"]
                                     for r in active])) if active else None}
        summary[bench] = {"contexts": len(rs), "contexts_with_opportunity": len(active),
                          "budgets": [r["query_budget"] for r in rs], "methods": methods,
                          "mscs_vs": comparisons,
                          "mscs_different_from_scalar_choices": sum(
                              decisions[r["context"]]["pairs"]["mscs_diversity"] != [0, 1] for r in rs)}
    evaluations = [load(p) for p in (out / "evaluations").glob("*.json")]
    report = {"summary": summary, "requested_distinct_configs": len(evaluations),
              "physical_model_queries": sum(r["mscs"] is not None or r["source"] != "saved_v5" for r in evaluations),
              "feature_queries": sum(r["mscs"] is not None for r in evaluations),
              "source_counts": dict(Counter(r["source"] for r in evaluations)),
              "evaluation_seconds_sum": sum(r["seconds"] for r in evaluations),
              "feature_seconds_sum": sum(r["seconds"] for r in evaluations if r["mscs"] is not None),
              "limits": "12 selected contexts, 2 workloads, fixed surrogate. Short scalar continuation only. "
                        "All selectors frozen before labels. Common unique-query budget per context. "
                        "Global raw HV sums mix workload scales; use per-workload and normalized capture. "
                        "No claim of full-search improvement or gem5 validation."}
    save(out / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def run(out):
    started = time.monotonic()
    plan = load(out / "plan.json")
    for p, expected in plan["source_sha256"].items():
        if sha(p) != expected:
            raise ValueError(f"Frozen input changed: {p}")
    data = {b: load(DATA / b / "s42/scalar_guard.json") for b in BENCHES}
    evaluator = Evaluator(out, data, data[BENCHES[0]]["area_sources"])
    dp = out / "decisions.json"
    if not dp.exists():
        decisions = []
        for ctx in plan["contexts"]:
            d = data[ctx["bench"]]
            ids = [ctx["root"]]+[c["index"] for c in ctx["candidates"]]
            cs = [d["configs"][i] for i in ids]
            observations = [evaluator.query(ctx["bench"], c, need_vector=True, tag="selection_feature") for c in cs]
            assert all(r["slice_ids"] == observations[0]["slice_ids"] for r in observations)
            scores = selectors(observations[0]["mscs"], [r["mscs"] for r in observations[1:]], cs[0], cs[1:])
            decisions.append(dict(id=ctx["id"], **scores))
            print(f"FEATURES {ctx['id']} requests={evaluator.n_requests} pairs={scores['pairs']}", flush=True)
        save(dp, {"frozen_before_continuation": True, "plan_sha256": sha(out / "plan.json"),
                  "contexts": decisions})
    decisions = {d["id"]: d for d in load(dp)["contexts"]}
    for ctx in plan["contexts"]:
        target = out / f"context_{ctx['id']}.json"
        if target.exists():
            continue
        branches = []
        for candidate in ctx["candidates"]:
            branches.append(rollout(ctx, data[ctx["bench"]], candidate, evaluator))
            print(f"BRANCH {ctx['id']} {candidate['param']} requests={evaluator.n_requests} "
                  f"seconds={time.monotonic()-started:.1f}", flush=True)
        result = assess(ctx, data[ctx["bench"]], branches, decisions[ctx["id"]])
        save(target, result)
        print(f"DONE {ctx['id']} q={result['query_budget']} scores={result['method_hvi']}", flush=True)
    evaluator.area.flush()
    if not (out / "summary.json").exists():
        summarize(out)
    print(f"COMPLETE requests={evaluator.n_requests} seconds={time.monotonic()-started:.1f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "run"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (prepare if args.action == "prepare" else run)(args.out.resolve())
