#!/usr/bin/env python3
"""Held-out-workload test of MSCS incremental information for conditional value.

The auxiliary regressor is a search policy, not a retrained PRISM checkpoint.
Fixed ExtraTrees settings, identical splits/labels/weights for all variants.
Predict the mixed finite difference from PRE-joint observations. Evaluate
conditional cost and within-state rankings; never split interventions from
one workload across train/test. Workloads remain in PRISM's original training
distribution; 'held out' refers only to this auxiliary search regressor.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from collect import PARAMS, get_param_values


def scalar_features(state, row):
    cfg = state["config"]
    span = np.array([max(get_param_values(p))-min(get_param_values(p)) for p in PARAMS])
    x0 = np.array([cfg[p] for p in PARAMS], float)
    xj = np.array([row["cfg_j"][p] for p in PARAMS], float)
    xk = np.array([row["cfg_k"][p] for p in PARAMS], float)
    x = [*(x0/span), *((xj-x0)/span), *((xk-x0)/span),
         *[float(row["j"] == p) for p in PARAMS],
         *[float(row["k"] == p) for p in PARAMS],
         row["current_cost"], row["future_gain"], state["cpi"],
         row["j_stride"], row["k_stride"], *row["widths"], *row["scalar_detail"]]
    return np.asarray(x, float)


def metrics(rows, score, tau, calibrated=True):
    actual = np.array([r["future_cost"] for r in rows])
    lab = actual > tau
    out = {"n": len(rows), "positives": int(lab.sum()),
           "mae": float(np.abs(actual-score).mean()),
           "auc": float(roc_auc_score(lab, score)) if 0 < lab.sum() < len(lab) else None,
           "ap": float(average_precision_score(lab, score)) if lab.any() else None}
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["bench"], r["state"])].append(i)
    aucs, capture, regret = [], [], []
    for ids in groups.values():
        ids = np.asarray(ids)
        y, s = actual[ids], np.asarray(score)[ids]
        labels = y > tau
        if 0 < labels.sum() < len(ids):
            aucs.append(roc_auc_score(labels, s))
        chosen = np.argsort(-s, kind="stable")[:3]
        positive_risk = np.maximum(y-tau, 0)
        oracle_risk = np.sort(positive_risk)[-3:].sum()
        if oracle_risk > 0:
            capture.append(positive_risk[chosen].sum()/oracle_risk)
        regret.append(max(y)-max(y[chosen]))
    out.update(macro_state_auc=float(np.mean(aucs)) if aucs else None,
               risk_capture_at3=float(np.mean(capture)) if capture else None,
               risk_states=len(capture), rank_regret_at3=float(np.mean(regret)))
    # Distinguish a true mixed effect from changing the denominator: even a
    # perfectly additive CPI response can cross the relative-cost threshold.
    interaction = np.array([r["interaction"] for r in rows])
    material = interaction > np.array([r["effect"] for r in rows])
    predicted_interaction = np.asarray(score)
    if calibrated:
        predicted_interaction = (predicted_interaction *
                                 (1-np.array([r["future_gain"] for r in rows])) -
                                 np.array([r["current_cost"] for r in rows]))
    out["material_interactions"] = int(material.sum())
    out["interaction_auc"] = (float(roc_auc_score(material, predicted_interaction))
                              if 0 < material.sum() < len(rows) else None)
    if not calibrated:
        out.pop("mae")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    target = Path(args.out)
    if target.exists():
        ap.error("Refusing to overwrite")
    rows, x_scalar, x_level, x_response = [], [], [], []
    tau = None
    for p in sorted(Path(args.data).glob("*.json")):
        data = json.loads(p.read_text())
        if "states" not in data:
            continue
        if tau is not None and tau != data["args"]["tau"]:
            raise ValueError("Mixed labeling thresholds")
        tau = data["args"]["tau"]
        for state in data["states"]:
            for row in state["rows"]:
                if row["current_cost"] < 0:
                    continue  # paper's actual Protect decision regime
                rows.append(dict(row, bench=data["bench"], state=state["state"],
                                 effect=data["args"]["effect"]))
                x_scalar.append(scalar_features(state, row))
                x_level.append(row["q0"])
                x_response.append(row["dqj"]+row["dqk"]+row["q_interaction"])
    x, q, dq = map(np.asarray, (x_scalar, x_level, x_response))
    if not rows:
        ap.error("No nonnegative near-flat reclaim observations")
    benches = np.array([r["bench"] for r in rows])
    labels = np.array([r["interaction"] for r in rows])
    costs = np.array([r["current_cost"] for r in rows])
    gains = np.array([r["future_gain"] for r in rows])
    # Shuffle complete MSCS records WITHIN workload; preserve workload fingerprint
    # while breaking their match to the particular state and action pair.
    order = np.arange(len(rows))
    rng = np.random.RandomState(1008)
    for bench in np.unique(benches):
        ix = np.flatnonzero(benches == bench)
        order[ix] = rng.permutation(ix)
    features = {"scalar_learned": x,
                "plus_mscs_level": np.c_[x, q],
                "plus_mscs_response": np.c_[x, q, dq],
                "plus_shuffled_mscs": np.c_[x, q[order], dq[order]]}
    predictions = {n: np.zeros(len(rows)) for n in features}
    weights = np.empty(len(rows))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row["bench"], row["state"])].append(i)
    for ix in groups.values():
        weights[ix] = 1/len(ix)
    params = dict(n_estimators=160, min_samples_leaf=8, max_depth=12,
                  max_features=1.0, n_jobs=2, random_state=1008)
    for bench in np.unique(benches):
        train, test = benches != bench, benches == bench
        for name, feat in features.items():
            model = ExtraTreesRegressor(**params)
            model.fit(feat[train], labels[train], sample_weight=weights[train])
            interaction = model.predict(feat[test])
            predictions[name][test] = (costs[test]+interaction)/(1-gains[test])
        print(f"held out {bench}: train={train.sum()} test={test.sum()}", flush=True)
    for name in ("current_cost", "scalar_add", "mscs_compose", "l1", "cancel", "composition_l1", "signed_alignment"):
        predictions[name] = np.array([r["prediction"][name] for r in rows])
    summary, by_bench, by_source = {}, {}, {}
    for name, pred in predictions.items():
        summary[name] = metrics(rows, pred, tau, name not in (
            "l1", "cancel", "composition_l1", "signed_alignment"))
    for bench in np.unique(benches):
        ix = np.flatnonzero(benches == bench)
        by_bench[bench] = {n: metrics([rows[i] for i in ix], p[ix], tau, n not in (
            "l1", "cancel", "composition_l1", "signed_alignment")) for n, p in predictions.items()}
    for source in ("legacy", "fresh"):
        ix = np.array([i for i, row in enumerate(rows) if row["state"].startswith(source)])
        if len(ix):
            by_source[source] = {n: metrics([rows[i] for i in ix], p[ix], tau, n not in (
                "l1", "cancel", "composition_l1", "signed_alignment")) for n, p in predictions.items()}
    result = {"protocol": "Leave one workload out, fixed model settings; all six workloads belong to the "
                           "frozen PRISM training distribution. Auxiliary regressor only; no checkpoint edits. "
                           "Scalar features include full configs, action deltas/IDs, scalar costs/gains and slice statistics. "
                           "Equal total training weight per state. Exploratory fixed-probe diagnostic, not DSE/HV.",
              "forest_params": params, "threshold": tau, "summary": summary,
              "by_bench": by_bench, "by_source": by_source,
              "predictions": [{"bench": r["bench"], "state": r["state"], "j": r["j"], "k": r["k"],
                               "j_stride": r["j_stride"], "k_stride": r["k_stride"],
                               "actual_cost": r["future_cost"], "interaction": r["interaction"],
                               **{n: float(p[i]) for n, p in predictions.items()}} for i, r in enumerate(rows)]}
    with target.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
