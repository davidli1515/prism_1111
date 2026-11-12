#!/usr/bin/env python3
"""Paper V-C Protect: measure resource value BEFORE and AFTER another improvement.

Frozen checkpoint; all output is new. No McPAT, no old-area based selection.
Each state observes base, +/- resource probes, then chooses three DISTINCT
future directions using scalar gain alone. Only then are joint interventions
queried for labels. Config effects must commute and add, preventing validity
projection from masquerading as performance interaction. Mean MSCS, pure
pressure-composition features and rich per-slice CPI controls are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "mscs_response_1008"))
from probe import SliceOracle, _shift, ALL_PARAMS, config_key, compute_w_min, DEFAULT_CKPT
from response import compose, cpi_samples, config_additive
from config_dse import DEFAULT_CONFIG, sample_random_config, get_param_values
from dse_engine import _project_to_valid

PARAMS = list(ALL_PARAMS)


def describe(m):
    """Strip Base/magnitude: conditional FP/BP/SP proportions per stage."""
    a = np.asarray(m, float).reshape(-1, 4, 4)
    pressure = a[..., 1:]
    return pressure / pressure.sum(axis=-1, keepdims=True)


def scalar_stats(x):
    x = np.asarray(x)
    return [float(x.mean()), float(x.std()), *np.quantile(x, [.1, .5, .9]).tolist()]


def get_states(bench, seed, state_set="mixed", random_states=2):
    saved = json.loads((ROOT / "mscs_guide/results" / f"masking_{bench}.json").read_text())
    result = ([(f"legacy_{s['state']}", s["cfg"]) for s in saved["rows"][:3]]
              if state_set == "mixed" else [])
    # Same new configurations across workloads: selection independent of labels/MSCS.
    rng = np.random.RandomState(seed)
    if state_set == "mixed":
        result.append(("fresh_default", _project_to_valid(DEFAULT_CONFIG)))
    result.extend((f"fresh_random_{i}", _project_to_valid(sample_random_config(rng)))
                  for i in range(random_states))
    return result


def observe(oracle, c, base_ids):
    m, v, ids = oracle.query(c)
    if ids != base_ids:
        raise ValueError("Unaligned slice IDs")
    return m, v


def collect(bench, args):
    oracle = SliceOracle(bench)
    states, t0 = [], time.time()
    for state, c in get_states(bench, args.seed, args.state_set, args.random_states):
        m0, c0, ids0 = oracle.query(c)
        w0 = compute_w_min(c)
        v0 = cpi_samples(m0, w0)
        q0 = describe(m0)
        up, down, seen = [], [], {config_key(c)}
        for n in PARAMS:
            # Preserve trace identity: L1 trace switches are a separate problem.
            if n == "l1_size":
                continue
            for direction, strides in ((+1, (1, 4, 8)), (-1, (1, 4))):
                for stride in strides:
                    cn = _shift(c, n, direction*stride)
                    if cn is None or config_key(cn) in seen:
                        continue
                    seen.add(config_key(cn))
                    mn, vn = observe(oracle, cn, ids0)
                    record = {"param": n, "stride": stride, "config": cn,
                              "mscs": mn, "cpi": vn, "cost": (vn-c0)/c0}
                    if direction > 0:
                        up.append(record)
                    elif abs(record["cost"]) <= args.tau:
                        down.append(record)
        # Distinct future directions, all strategies share these already observed probes.
        future, used = [], set()
        for u in sorted(up, key=lambda r: (r["cost"], r["param"], r["stride"])):
            if u["cost"] >= 0 or u["param"] in used:
                continue
            future.append(u)
            used.add(u["param"])
            if len(future) == args.partners:
                break
        observed_queries = oracle.n_queries
        rows, rejected = [], 0
        for dec in down:
            j, cj, mj, vj = dec["param"], dec["config"], dec["mscs"], dec["cpi"]
            for inc in future:
                k, ck, mk, vk = inc["param"], inc["config"], inc["mscs"], inc["cpi"]
                if j == k:
                    continue
                cjk = _shift(cj, k, inc["stride"])
                ckj = _shift(ck, j, -dec["stride"])
                if (cjk is None or ckj is None or config_key(cjk) != config_key(ckj)
                        or not config_additive(c, cj, ck, cjk)):
                    rejected += 1
                    continue
                mjk, vjk = observe(oracle, cjk, ids0)
                qj, qk = describe(mj), describe(mk)
                dqj, dqk = qj-q0, qk-q0
                wj, wk, wjk = [compute_w_min(x) for x in (cj, ck, cjk)]
                pred_cpi = float(cpi_samples(compose(m0, mj, mk), wjk).mean())
                dj = cpi_samples(mj, wj)/v0-1
                gk = 1-cpi_samples(mk, wk)/v0
                delta_j = mj-m0
                pressure = delta_j.reshape(-1, 4, 4)[..., 1:]
                record = {
                    "j": j, "j_stride": dec["stride"], "k": k, "k_stride": inc["stride"],
                    "cfg_j": cj, "cfg_k": ck, "cfg_jk": cjk,
                    "current_cost": dec["cost"], "future_gain": (c0-vk)/c0,
                    "future_cost": (vjk-vk)/vk,
                    "interaction": (vjk-vk-vj+c0)/c0,
                    "masked": bool(dec["cost"] >= 0 and (vjk-vk)/vk > args.tau),
                    "material_interaction": bool((vjk-vk-vj+c0)/c0 > args.effect),
                    "prediction": {
                        "current_cost": dec["cost"],
                        "scalar_add": (vj-c0)/vk,
                        "mscs_compose": (pred_cpi-vk)/vk,
                        "l1": float(np.abs(delta_j).sum(axis=1).mean()),
                        "cancel": float((np.abs(pressure).sum(axis=(1, 2))-
                                         np.abs(pressure.sum(axis=(1, 2)))).mean()),
                        "composition_l1": float(np.abs(dqj).sum(axis=(1, 2)).mean()),
                        "signed_alignment": float((dqj*dqk).sum(axis=(1, 2)).mean()),
                    },
                    "scalar_detail": scalar_stats(v0) + scalar_stats(dj) + scalar_stats(gk),
                    "q0": q0.mean(axis=0).ravel().tolist(),
                    "dqj": dqj.mean(axis=0).ravel().tolist(),
                    "dqk": dqk.mean(axis=0).ravel().tolist(),
                    "q_interaction": (dqj*dqk).mean(axis=0).ravel().tolist(),
                    "widths": [w0, wj, wk, wjk],
                    "cpi": [c0, vj, vk, vjk],
                }
                rows.append(record)
        states.append({"state": state, "config": c, "mscs": m0.mean(0).tolist(), "cpi": c0,
                       "eligible_reclaims": len(down), "future_directions": len(future),
                       "rejected_projection": rejected, "rows": rows,
                       "label_queries": oracle.n_queries-observed_queries})
        active = [r for r in rows if r["current_cost"] >= 0]
        print(f"{bench} {state}: rows={len(rows)} protect={sum(r['masked'] for r in active)}/{len(active)} "
              f"queries={oracle.n_queries} elapsed={time.time()-t0:.0f}s", flush=True)
    return {"bench": bench, "states": states, "queries": oracle.n_queries, "seconds": time.time()-t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", default="*")
    ap.add_argument("--seed", type=int, default=1008)
    ap.add_argument("--tau", type=float, default=.003)
    ap.add_argument("--effect", type=float, default=.001)
    ap.add_argument("--partners", type=int, default=3)
    ap.add_argument("--state-set", choices=("mixed", "fresh"), default="mixed")
    ap.add_argument("--random-states", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / "mscs_guide/results").glob(f"masking_{args.bench}.json"))
    if not paths:
        ap.error("No benchmarks matched")
    for p in paths:
        bench = p.stem[len("masking_"):]
        target = out / f"{bench}.json"
        if target.exists():
            raise FileExistsError(target)
        data = collect(bench, args)
        data.update(args=vars(args), checkpoint=DEFAULT_CKPT,
                    checkpoint_sha256=hashlib.sha256(Path(DEFAULT_CKPT).read_bytes()).hexdigest(),
                    note="Frozen surrogate only. No area/HV claim. Labels use extra offline queries. "
                         "All current/future observations precede joint labels; no target labels in features.")
        with target.open("x") as f:
            json.dump(data, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
