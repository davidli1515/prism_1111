#!/usr/bin/env python3
"""Pooled validation for Section V-A: every existing gem5 run of the trajectory study that has a stored PRISM
prediction (model_cpi, model_mscs in the round's plan.json), deduplicated by (benchmark, design).
Rounds: gem5_val, gem5_val3 (random-start greedy walks), gem5_val4 (+plan_peak; small-start greedy walks),
gem5_val2 (random search designs), gem5_val5 / gem5_val6 (BO, GA, random-search incumbents and trajectories).
gem5_val7 is skipped (no stored MSCS prediction). No simulation, no model call; read-only on all inputs.
Per method and overall: CPI MAPE, MSCS MAE, stall-share MAE, Spearman of Dispatch BP stall cycles, state
agreement (exact / exact-or-adjacent along S0-S2-S1-S3 / coarse frontend {S0,S2} vs backend {S1,S3,S4}),
with 95% bootstrap intervals resampling benchmarks. Writes gem5_val4/settings_validation_pooled.txt
(never overwrites)."""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gem5_val4", "settings_validation_pooled.txt")
ORDER = {0: 0, 2: 1, 1: 2, 3: 3}
COARSE = {0: 0, 2: 0, 1: 1, 3: 1, 4: 1, 5: 2}
ROUNDS = [("gem5_val", ["plan.json"], "Greedy"), ("gem5_val3", ["plan.json"], "Greedy"),
          ("gem5_val4", ["plan.json", "plan_peak.json"], "Greedy"), ("gem5_val2", ["plan.json"], "Random"),
          ("gem5_val5", ["plan.json"], None), ("gem5_val6", ["plan.json"], None)]


def feat(cpi, m):
    X = np.stack([cpi * m[:, 4 * s + k] for s in (1, 3) for k in (1, 2, 3)], 1)
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def load():
    rows, seen = [], set()
    for d, plans, meth in ROUNDS:
        for pf in plans:
            for j in json.load(open(os.path.join(HERE, d, pf))):
                if "model_mscs" not in j:
                    continue
                p = os.path.join(HERE, d, "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
                key = (j["bench"], tuple(sorted(j["config"].items())))
                if not os.path.exists(p) or key in seen:
                    continue
                seen.add(key); r = json.load(open(p))
                rows.append(dict(bench=j["bench"], method=meth or j["opt"], round=d, mc=j["model_cpi"], gc=r["cpi"],
                                 mm=np.array(j["model_mscs"]), gm=np.array(r["mscs16"])))
    return rows


def metrics(R, cent):
    mc = np.array([r["mc"] for r in R]); gc = np.array([r["gc"] for r in R])
    mm = np.stack([r["mm"] for r in R]); gm = np.stack([r["gm"] for r in R])
    fm, fg = feat(mc, mm), feat(gc, gm)
    lab = lambda F: np.argmin(((F[:, None] - cent[None]) ** 2).sum(2), 1)
    lm, lg = lab(fm), lab(fg)
    adj = np.mean([(a == b) or (a in ORDER and b in ORDER and abs(ORDER[a] - ORDER[b]) == 1) for a, b in zip(lm, lg)])
    return dict(n=len(R), mape=100 * np.mean(abs(mc - gc) / gc), mae=np.mean(abs(mm - gm)), share=np.mean(abs(fm - fg)),
                rho=spearmanr(mc * mm[:, 5], gc * gm[:, 5]).correlation, exact=100 * np.mean(lm == lg),
                adj=100 * adj, coarse=100 * np.mean([COARSE[a] == COARSE[b] for a, b in zip(lm, lg)]))


def boot(R, cent, keys, nb=1000):
    rng = np.random.RandomState(0); B = sorted({r["bench"] for r in R}); by = {b: [r for r in R if r["bench"] == b] for b in B}
    S = [metrics([r for b in rng.choice(B, len(B)) for r in by[b]], cent) for _ in range(nb)]
    return {k: np.percentile([s[k] for s in S], [2.5, 97.5]) for k in keys}


def main():
    if os.path.exists(OUT):
        sys.exit(f"exists, refusing to overwrite: {OUT}")
    cent = np.load(os.path.join(HERE, "results_small", "state_clusters_v1.npz"))["centroids"]
    rows = load(); keys = ["mape", "mae", "share", "rho", "exact", "adj", "coarse"]
    L = [f"{len(rows)} unique (benchmark, design) pairs with gem5 results; {len({r['bench'] for r in rows})} benchmarks",
         "rounds: " + ", ".join(f"{d} {sum(r['round'] == d for r in rows)}" for d, _, _ in ROUNDS),
         f"{'method':8s}{'n':>6s}{'CPI MAPE':>10s}{'MSCS MAE':>10s}{'share MAE':>11s}{'rho D-BP cyc':>14s}"
         f"{'exact':>8s}{'adj':>7s}{'coarse':>8s}"]
    for name in ["Greedy", "BO", "GA", "Random", "All"]:
        R = rows if name == "All" else [r for r in rows if r["method"] == name]
        if not R:
            continue
        m = metrics(R, cent)
        L.append(f"{name:8s}{m['n']:6d}{m['mape']:9.1f}%{m['mae']:10.3f}{m['share']:11.3f}{m['rho']:14.2f}"
                 f"{m['exact']:7.0f}%{m['adj']:6.0f}%{m['coarse']:7.0f}%")
    ci = boot(rows, cent, keys)
    L.append("All, 95% bootstrap over benchmarks: " + ", ".join(f"{k} [{ci[k][0]:.3g}, {ci[k][1]:.3g}]" for k in keys))
    print("\n".join(L)); open(OUT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
