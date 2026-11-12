#!/usr/bin/env python3
"""
Headroom check around the best-known Pareto front (surrogate only, CPU).

1. Union front U of all runs in <dir> (best known so far).
2. For every point on U, evaluate every single-stride move (+1 / -1 stride on each of
   the 21 parameters, projected to a valid config).
3. Report: how many neighbours are NOT dominated by U (= improvements BO and the others
   missed), new union HV, and *which kind of move* produced them
   (increase vs decrease, parameter, and the MSCS state of the parent point).
Iterate --rounds times from the newly found points (a cheap local-search ceiling).

Usage: python headroom_check.py results_v2_1005 --rounds 3
Output: <dir>/headroom.json
"""
import os, sys, json, glob, argparse, time
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Surrogate, Area, pareto_mask, hypervolume, ALL_PARAMS, config_key
from dse_engine import _project_to_valid, _increase_by_stride, _decrease_by_stride

FP = [1, 5, 9, 13]; BP = [2, 6, 10, 14]; SP = [3, 7, 11, 15]


def dominated_by(p, F):
    return np.any((F[:, 0] <= p[0]) & (F[:, 1] <= p[1]) & ((F[:, 0] < p[0]) | (F[:, 1] < p[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    runs = [json.load(open(f)) for f in glob.glob(os.path.join(args.dir, "*_s*.json"))
            if not f.endswith("_pareto.json")]
    ref = json.load(open(os.path.join(args.dir, "summary.json")))["ref_point"]
    configs = [c for r in runs for c in r["configs"]]
    P = np.vstack([np.c_[r["cpis"], r["areas"]] for r in runs])
    m = pareto_mask(P)
    front = [(configs[i], P[i]) for i in np.where(m)[0]]
    hv0 = hypervolume(P[m], ref)
    print(f"start: union front {len(front)} pts, HV={hv0:.2f}", flush=True)

    S, A = Surrogate(device=args.device), Area()
    allc, allP = [c for c, _ in front], [p for _, p in front]
    seen = {config_key(c) for c in configs}
    log = {"start_hv": hv0, "rounds": []}
    parents = front
    for rnd in range(args.rounds):
        t0 = time.time()
        found = []
        for pc, pp in parents:
            ms = S.mscs(pc)
            press = {"FP": ms[FP].mean(), "BP": ms[BP].mean(), "SP": ms[SP].mean()}
            dom = max(press, key=press.get)
            for name in ALL_PARAMS:
                for kind, fn in (("inc", _increase_by_stride), ("dec", _decrease_by_stride)):
                    v, _ = fn(pc, name)
                    if v is None:
                        continue
                    c = dict(pc); c[name] = v; c = _project_to_valid(c)
                    k = config_key(c)
                    if k in seen:
                        continue
                    seen.add(k)
                    a = A(c)
                    if a is None:
                        continue
                    p = np.array([S.cpi(c), a])
                    allc.append(c); allP.append(p)
                    found.append({"config": c, "p": p, "param": name, "kind": kind,
                                  "parent_dominant": dom, "parent_cpi": float(pp[0]),
                                  "parent_area": float(pp[1])})
        PP = np.array(allP)
        mk = pareto_mask(PP)
        F = PP[mk]
        new_front = [f for f in found if not dominated_by(f["p"], F) and
                     any(np.allclose(f["p"], q) for q in F)]
        hv = hypervolume(F, ref)
        stats = {"round": rnd, "evaluated": len(found), "new_front_pts": len(new_front),
                 "hv": hv, "hv_gain_vs_start_pct": (hv - hv0) / hv0 * 100,
                 "queries": S.n_queries, "time_s": time.time() - t0,
                 "by_move": Counter(f"{f['kind']}:{f['param']}" for f in new_front).most_common(),
                 "by_parent_dominant": Counter(f"{f['parent_dominant']}->{f['kind']}"
                                               for f in new_front).most_common(),
                 "cpi_gain_pct": [float((f['parent_cpi'] - f['p'][0]) / f['parent_cpi'] * 100)
                                  for f in new_front],
                 "area_gain_pct": [float((f['parent_area'] - f['p'][1]) / f['parent_area'] * 100)
                                   for f in new_front]}
        log["rounds"].append(stats)
        print(f"round {rnd}: evaluated {len(found)}, new front pts {len(new_front)}, "
              f"HV {hv:.2f} (+{stats['hv_gain_vs_start_pct']:.2f}%), {stats['time_s']:.0f}s", flush=True)
        print("   moves:", stats["by_move"][:10], flush=True)
        print("   parent dominant bottleneck -> move:", stats["by_parent_dominant"], flush=True)
        A.flush()
        if not new_front:
            break
        parents = [(f["config"], f["p"]) for f in new_front]

    PP = np.array(allP); F = PP[pareto_mask(PP)]
    log["final_front"] = F.tolist()
    json.dump(log, open(os.path.join(args.dir, "headroom.json"), "w"), indent=1, default=str)
    print(f"saved {os.path.join(args.dir, 'headroom.json')}")


if __name__ == "__main__":
    main()
