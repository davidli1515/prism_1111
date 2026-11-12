#!/usr/bin/env python3
"""Larger-step response test: one local probe predicts 2/4/8-stride actions.

Area-independent performance diagnostic, not a DSE/HV result. Every policy sees
the same base + all one-stride probes, then verifies k larger-step candidates.
Exhaustive larger-step queries are offline oracle construction costs, logged
separately from the hypothetical online query budget. No model is retrained.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from probe import ROOT, SliceOracle, _shift, ALL_PARAMS, compute_w_min, config_key, ranking_metrics
from response import extrapolate, cpi_samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--states", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        ap.error("Refusing to overwrite output")
    start, results = time.time(), []
    for file in sorted((ROOT/"mscs_guide/results").glob("masking_*.json")):
        old = json.loads(file.read_text())
        oracle = SliceOracle(old["bench"])
        rows = []
        for s in old["rows"][:args.states]:
            c = s["cfg"]
            m0, v0, ids0 = oracle.query(c)
            if not np.isclose(v0, s["cpi"], rtol=3e-6):
                raise ValueError("Saved baseline does not match current checkpoint")
            singles = {}
            for n in ALL_PARAMS:
                cn = _shift(c, n, 1)
                if cn is not None:
                    singles[n] = (cn, oracle.query(cn))
            known = {config_key(c)} | {config_key(v[0]) for v in singles.values()}
            baseline = max([0.] + [1-v[1][1]/v0 for v in singles.values()])
            candidates = []
            for n, (c1, (m1, v1, ids1)) in singles.items():
                # L1 switches the trace precompute: it cannot use this local continuation.
                if n == "l1_size" or ids0 != ids1:
                    continue
                d1 = np.array([c1[k]-c[k] for k in ALL_PARAMS], float)
                for stride in (2, 4, 8):
                    cn = _shift(c, n, stride)
                    if cn is None or config_key(cn) in known:
                        continue
                    dn = np.array([cn[k]-c[k] for k in ALL_PARAMS], float)
                    distance = float(dn@d1/(d1@d1))
                    if not np.allclose(dn, distance*d1):
                        continue  # a projection changed the action direction
                    known.add(config_key(cn))
                    w0, w1, wn = map(compute_w_min, (c, c1, cn))
                    clip = lambda v: float(np.clip(v, 1/wn, 100/wn))
                    pred = {
                        "scalar_linear": clip(v0+distance*(v1-v0)),
                        "scalar_log": clip(np.exp(np.log(v0)+distance*np.log(v1/v0))),
                        "scalar_width_linear": clip((v0*w0+distance*(v1*w1-v0*w0))/wn),
                        "scalar_width_log": clip(np.exp(np.log(v0*w0)+distance*np.log(v1*w1/(v0*w0)))/wn),
                        "mscs_response": float(cpi_samples(extrapolate(m0, m1, distance), wn).mean()),
                    }
                    # Same sample/stage detail, with pressure composition removed.
                    b0, b1 = np.maximum(m0[..., ::4], .01), np.maximum(m1[..., ::4], .01)
                    slots = np.exp(-np.log(b0)+distance*np.log(b0/b1))
                    pred["base_detail"] = float(np.clip(slots/wn, 1/wn, 100/wn).mean())
                    _, actual, _ = oracle.query(cn)
                    candidates.append({"param": n, "stride": stride, "distance": distance,
                                       "gain": 1-actual/v0, "config": cn,
                                       "pred_gain": {k: 1-v/v0 for k, v in pred.items()}})
            if candidates:
                predictions = {k: [p["pred_gain"][k] for p in candidates] for k in candidates[0]["pred_gain"]}
                predictions["random"] = np.zeros(len(candidates))
                rows.append({"state": s["state"], "best_single_gain": baseline,
                             "known_configurations": 1+len(singles), "candidates": candidates,
                             "metrics": ranking_metrics([p["gain"] for p in candidates], predictions,
                                                        baseline, np.random.RandomState(s["state"]))})
            print(f"{old['bench']} state={s['state']} candidates={len(candidates)} q={oracle.n_queries}", flush=True)
        results.append({"bench": old["bench"], "rows": rows, "offline_queries": oracle.n_queries})
    rows = [r for b in results for r in b["rows"] if r["metrics"]]
    summary = {name: {k: float(np.mean([r["metrics"][name][k]["retained_gain"] for r in rows]))
                      for k in ("1", "3", "5")} for name in rows[0]["metrics"]}
    result = {"args": vars(args), "summary": summary, "results": results,
              "wall_seconds": time.time()-start,
              "note": "60 previously selected plateau states, current frozen surrogate; no gem5/area claim. "
                      "Ranking budgets include all observed singles identically for all methods."}
    out.parent.mkdir(exist_ok=True)
    with out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
