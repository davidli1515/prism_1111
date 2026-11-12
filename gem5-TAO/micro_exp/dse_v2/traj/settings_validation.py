#!/usr/bin/env python3
"""Validation numbers for Section V-A (Settings) on the full small-start gem5 set: gem5_val4 plan.json (201 states,
one representative walk per benchmark, 12 evenly spaced states) + plan_peak.json (4 peak states) = 205 designs.
Reports CPI MAPE, MSCS MAE (16 entries), stall-share MAE (six-component feature of Section V-B), and
state-assignment agreement: each design is assigned to the nearest of the six reference centroids
(results_small/state_clusters_v1.npz) once from the PRISM MSCS and once from the gem5 MSCS.
Read-only on all inputs. Writes gem5_val4/settings_validation.txt (never overwrites)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gem5_val4", "settings_validation.txt")
ORDER = [0, 2, 1, 3]                       # frontend -> backend order of the states along PC1 (Section V-B)


def feat(cpi, m):                          # as cluster_opt_independent.feat
    m = np.atleast_2d(m); cpi = np.atleast_1d(cpi)
    X = np.stack([cpi * m[:, 4 * s + k] for s in (1, 3) for k in (1, 2, 3)], 1)
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def main():
    if os.path.exists(OUT):
        sys.exit(f"exists, refusing to overwrite: {OUT}")
    jobs = json.load(open(os.path.join(HERE, "gem5_val4", "plan.json"))) + \
        json.load(open(os.path.join(HERE, "gem5_val4", "plan_peak.json")))
    rows = []
    for j in jobs:
        p = os.path.join(HERE, "gem5_val4", "runs", j["bench"], f"p{j['idx']}_{j['tag']}", "result.json")
        if os.path.exists(p):
            r = json.load(open(p)); rows.append((j, r["cpi"], np.array(r["mscs16"])))
    mc = np.array([j["model_cpi"] for j, _, _ in rows]); gc = np.array([c for _, c, _ in rows])
    mm = np.array([j["model_mscs"] for j, _, _ in rows]); gm = np.array([m for _, _, m in rows])
    fm, fg = feat(mc, mm), feat(gc, gm)
    cent = np.load(os.path.join(HERE, "results_small", "state_clusters_v1.npz"))["centroids"]
    lab = lambda F: np.argmin(((F[:, None, :] - cent[None]) ** 2).sum(2), 1)
    lm, lg = lab(fm), lab(fg)
    k = len(cent); conf = np.zeros((k, k), int)
    for a, b in zip(lm, lg):
        conf[a, b] += 1
    po = np.mean(lm == lg); pe = sum(np.mean(lm == s) * np.mean(lg == s) for s in range(k)); kappa = (po - pe) / (1 - pe)
    pos = {s: i for i, s in enumerate(ORDER)}
    adj = np.mean([(a == b) or (a in pos and b in pos and abs(pos[a] - pos[b]) == 1) for a, b in zip(lm, lg)])
    L = [f"{len(rows)}/{len(jobs)} designs with gem5 results; {len({j['bench'] for j, _, _ in rows})} benchmarks",
         f"CPI MAPE {100*np.mean(abs(mc-gc)/gc):.1f}%   MSCS MAE (16 entries) {np.mean(abs(mm-gm)):.3f}   "
         f"stall-share MAE (6 components) {np.mean(abs(fm-fg)):.3f}",
         f"state agreement (nearest of {k} reference centroids): exact {100*po:.1f}% ({int(po*len(rows))}/{len(rows)}), "
         f"Cohen kappa {kappa:.2f}, exact-or-adjacent along S0-S2-S1-S3 {100*adj:.1f}%",
         "confusion (rows: PRISM state, cols: gem5 state):",
         "      " + "".join(f"S{s:<4d}" for s in range(k))]
    L += [f"  S{s}  " + "".join(f"{conf[s, t]:<5d}" for t in range(k)) for s in range(k)]
    L.append("states per source: PRISM " + str(np.bincount(lm, minlength=k).tolist()) + ", gem5 " + str(np.bincount(lg, minlength=k).tolist()))
    print("\n".join(L)); open(OUT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
