#!/usr/bin/env python3
"""
Fit the fast area model used only inside iso-area refine steps.
Data: N random 17-param configs plus single-parameter +/-1 neighbours of a subset (the moves a
refine step compares), all with race-free McPAT.  Model: gradient-boosted trees on log(area).
Reports held-out error on (a) random configs and (b) RATIOS area(neighbour)/area(base), which is
what the iso-area band actually uses.  Writes area_model.pkl + area_model_report.json
(refuses to overwrite).
Usage: python fit_area_model.py [--n 1200] [--n-base 60]
"""
import os, sys, json, pickle, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import space17 as S
from area17 import AreaStore, featurize, MODEL
from sklearn.ensemble import HistGradientBoostingRegressor


def neighbours(c):
    out = []
    for n in S.C.GRADIENT_PARAMS:
        for fn in (S.C._increase_by_stride, S.C._decrease_by_stride):
            v, _ = fn(c, n)
            if v is not None:
                out.append(S.project17(dict(c, **{n: v})))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--n-base", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    rep_path = MODEL.replace(".pkl", "_report.json")
    if os.path.exists(MODEL) or os.path.exists(rep_path):
        sys.exit(f"exists, refusing to overwrite: {MODEL}")
    rng = np.random.RandomState(123)
    rand = [S.sample17(rng) for _ in range(args.n)]
    bases = rand[:args.n_base]
    nb = [(i, x) for i, b in enumerate(bases) for x in neighbours(b)]
    A = AreaStore(args.workers)
    A.prefetch(rand + [x for _, x in nb]); A.flush()
    ra = np.array([A(c) if A(c) is not None else np.nan for c in rand])
    ok = ~np.isnan(ra)
    idx = np.where(ok)[0]; rng.shuffle(idx)
    cut = int(0.8 * len(idx)); tr, te = idx[:cut], idx[cut:]
    Xr = featurize(rand)
    # neighbours of the held-out bases are test data for the ratio check; others go to training
    test_bases = set(i for i in range(args.n_base) if i in set(te))
    nb_tr = [(i, x) for i, x in nb if i not in test_bases and A(x) is not None]
    Xtr = np.vstack([Xr[tr], featurize([x for _, x in nb_tr])])
    ytr = np.log(np.r_[ra[tr], [A(x) for _, x in nb_tr]])
    m = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.05, max_leaf_nodes=63,
                                      l2_regularization=1e-3, random_state=0).fit(Xtr, ytr)
    pred = np.exp(m.predict(Xr[te]))
    rel = np.abs(pred - ra[te]) / ra[te]
    rat_err = []
    for i, x in nb:
        if i in test_bases and A(x) is not None and not np.isnan(ra[i]):
            true_r = A(x) / ra[i]; p = np.exp(m.predict(featurize([x, bases[i]])))
            rat_err.append(abs(p[0] / p[1] - true_r))
    rep = {"n_train": int(len(ytr)), "n_test": int(len(te)), "mcpat_fail_rate": float(1 - ok.mean()),
           "abs_rel_err_median": float(np.median(rel)), "abs_rel_err_p90": float(np.percentile(rel, 90)),
           "ratio_abs_err_median": float(np.median(rat_err)) if rat_err else None,
           "ratio_abs_err_p90": float(np.percentile(rat_err, 90)) if rat_err else None,
           "n_ratio_tests": len(rat_err)}
    print(json.dumps(rep, indent=1))
    pickle.dump(m, open(MODEL, "wb")); json.dump(rep, open(rep_path, "w"), indent=1)


if __name__ == "__main__":
    main()
