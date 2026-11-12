#!/usr/bin/env python3
"""Per-benchmark comparison (common ref per bench): bo, v3 scalar, v3 mscs, v3 mscs no-decouple,
v4 mscs. Reports HV gain vs bo, share of area grid with lowest CPI, min CPI at area budgets.
Output: results_v4_1007/summary_v4_single.txt"""
import json,sys,os,numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import pareto_mask, hypervolume
SRC={"bo":"results_v3_single_1006/{b}/bo_s42.json","scalar":"results_v3_single_1006/{b}/bo_v3_scalar_s42.json",
     "v3_mscs":"results_v3_single_1006/{b}/bo_v3_mscs_s42.json","v3_nodecouple":"results_v3_single_nodecouple_1006/{b}/bo_v3_mscs_s42.json",
     "v4_mscs":"results_v4_1007/{b}/bo_v4_mscs_s42.json"}
benches=sorted({d for d in os.listdir("results_v3_single_1006") if os.path.isdir(f"results_v3_single_1006/{d}") and d!="logs"})
lines=[]; agg={m:[] for m in SRC}
for b in benches:
    R={m:json.load(open(p.format(b=b))) for m,p in SRC.items() if os.path.exists(p.format(b=b))}
    if "bo" not in R or "scalar" not in R: continue
    allP=np.vstack([np.c_[r['cpis'],r['areas']] for r in R.values()]); ref=(allP[:,0].max()*1.05,allP[:,1].max()*1.05)
    F={m:(lambda P:P[pareto_mask(P)])(np.c_[r['cpis'],r['areas']]) for m,r in R.items()}
    hb=hypervolume(F['bo'],ref); best=lambda f,A: f[f[:,1]<=A][:,0].min() if (f[:,1]<=A).any() else np.nan
    grid=np.arange(15,130,1.0); W={m:0 for m in R}
    for A in grid:
        v={m:best(F[m],A) for m in R}; lo=min(x for x in v.values() if x==x)
        for m in R:
            if v[m]==v[m] and abs(v[m]-lo)<1e-9: W[m]+=1
    lines.append(f"\n== {b}\n{'method':16s}{'HV':>8s}{'vs bo':>8s}{'grid%':>7s} | minCPI@ "+"".join(f"{A:>7d}" for A in (20,30,40,60,100)))
    for m in SRC:
        if m not in R: continue
        h=hypervolume(F[m],ref); g=(h/hb-1)*100; agg[m].append(g)
        lines.append(f"{m:16s}{h:8.1f}{g:+7.2f}%{W[m]/len(grid)*100:6.0f}% | "+"".join(f"{best(F[m],A):7.3f}" for A in (20,30,40,60,100)))
lines.append("\n== mean HV gain vs bo over benches with that method: "+"  ".join(f"{m}={np.mean(v):+.2f}%(n={len(v)})" for m,v in agg.items() if v))
out="\n".join(lines); print(out); open("results_v4_1007/summary_v4_single.txt","w").write(out)
