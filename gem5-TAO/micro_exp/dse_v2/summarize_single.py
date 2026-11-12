#!/usr/bin/env python3
"""
Per-benchmark summary for single-benchmark DSE (results_v3_single_*/<bench>/<method>_s<seed>.json).

For each benchmark: common reference point over all its runs (max CPI, max area) x 1.1;
HV at query checkpoints; relative gain of each method vs bo (same seed).
Aggregates: median / geomean gain, wins/ties/losses (|gain| < 0.1% = tie).

Usage: python summarize_single.py results_v3_single_1006 [--checkpoints 500 1000 2000 5000]
Output: <dir>/summary_single.json, <dir>/summary_single.txt
"""
import os, sys, json, glob, argparse
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import hypervolume

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--checkpoints", type=int, nargs="+", default=[500, 1000, 2000, 3000, 5000])
args = ap.parse_args()

runs = defaultdict(lambda: defaultdict(dict))          # bench -> method -> seed -> run
for f in glob.glob(os.path.join(args.dir, "*", "*_s*.json")):
    r = json.load(open(f))
    runs[r["bench"]][r["method"]][r["seed"]] = r

def hv_at(r, ref, q):
    C, A, Q = map(np.array, (r["cpis"], r["areas"], r["queries"]))
    k = Q <= q
    return hypervolume(np.c_[C[k], A[k]], ref) if k.any() else 0.0

lines, out = [], {}
methods = sorted({m for b in runs.values() for m in b if m != "bo"})
Qf = max(args.checkpoints)
hdr = f"{'benchmark':18s}" + "".join(f"{m+' vs bo':>20s}" for m in methods) + f"{'HV bo':>10s}"
lines += [f"HV gain (%) vs pure BO at {Qf} queries (mean over seeds; per-benchmark ref point)", hdr]
gains = defaultdict(list)
for b in sorted(runs):
    allr = [r for m in runs[b].values() for r in m.values()]
    P = np.vstack([np.c_[r["cpis"], r["areas"]] for r in allr])
    ref = (P[:, 0].max() * 1.1, P[:, 1].max() * 1.1)
    out[b] = {"ref": ref, "methods": {}}
    for m in runs[b]:
        out[b]["methods"][m] = {str(s): {str(q): hv_at(r, ref, q) for q in args.checkpoints}
                                for s, r in runs[b][m].items()}
    row = f"{b:18s}"
    for m in methods:
        g = []
        for s, r in runs[b].get(m, {}).items():
            if s in runs[b].get("bo", {}):
                hb = hv_at(runs[b]["bo"][s], ref, Qf)
                g.append((hv_at(r, ref, Qf) - hb) / hb * 100)
        if g:
            gains[m].append(np.mean(g)); row += f"{np.mean(g):19.2f}%"
        else:
            row += f"{'-':>20s}"
    hb = [hv_at(r, ref, Qf) for r in runs[b].get("bo", {}).values()]
    row += f"{np.mean(hb):10.1f}" if hb else f"{'-':>10s}"
    lines.append(row)
lines.append("")
for m in methods:
    g = np.array(gains[m])
    if len(g):
        w, t, l = (g > 0.1).sum(), (abs(g) <= 0.1).sum(), (g < -0.1).sum()
        gm = (np.prod(1 + g / 100) ** (1 / len(g)) - 1) * 100
        lines.append(f"{m:16s} n={len(g):2d}  median {np.median(g):+.2f}%  geomean {gm:+.2f}%  "
                     f"mean {g.mean():+.2f}%  wins/ties/losses {w}/{t}/{l}")
        out.setdefault("_aggregate", {})[m] = {"n": int(len(g)), "median": float(np.median(g)),
                                               "geomean": float(gm), "wins": int(w), "ties": int(t),
                                               "losses": int(l)}
txt = "\n".join(lines)
print(txt)
open(os.path.join(args.dir, "summary_single.txt"), "w").write(txt + "\n")
json.dump(out, open(os.path.join(args.dir, "summary_single.json"), "w"), indent=1)
