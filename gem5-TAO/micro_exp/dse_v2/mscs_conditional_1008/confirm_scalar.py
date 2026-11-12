#!/usr/bin/env python3
"""Freeze the auxiliary scalar baseline on discovery data; predict new configs.
No confirmation labels enter training. Exclude the target workload from fitting.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from analyze import scalar_features


def load(path):
    rows, features = [], []
    for p in sorted(Path(path).glob("*.json")):
        d = json.loads(p.read_text())
        if "states" not in d:
            continue
        for state in d["states"]:
            for row in state["rows"]:
                if row["current_cost"] < 0:
                    continue
                rows.append(dict(row, bench=d["bench"], state=state["state"]))
                features.append(scalar_features(state, row))
    return rows, np.asarray(features)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        ap.error("Refusing to overwrite")
    train, x = load(args.train)
    test, xt = load(args.test)
    weights = np.array([1/sum(t["bench"] == r["bench"] and t["state"] == r["state"] for t in train)
                        for r in train])
    y = np.array([r["interaction"] for r in train])
    prediction = np.zeros(len(test))
    for bench in sorted({r["bench"] for r in test}):
        tr = np.array([r["bench"] != bench for r in train])
        te = np.array([r["bench"] == bench for r in test])
        reg = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=8, max_depth=12,
                                 max_features=1.0, n_jobs=2, random_state=1008)
        reg.fit(x[tr], y[tr], sample_weight=weights[tr])
        prediction[te] = reg.predict(xt[te])
    result = {"protocol": "Auxiliary scalar regressor fit ONLY on earlier discovery data and excluding target "
                          "workload. Forest settings identical to discovery. Predicts independent seed configs.",
              "args": vars(args), "predictions": []}
    for r, interaction in zip(test, prediction):
        result["predictions"].append({"bench": r["bench"], "state": r["state"],
                                      "j": r["j"], "j_stride": r["j_stride"],
                                      "k": r["k"], "k_stride": r["k_stride"],
                                      "scalar_learned": float((r["current_cost"]+interaction)/(1-r["future_gain"]))})
    with out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
