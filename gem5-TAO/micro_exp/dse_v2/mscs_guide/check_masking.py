#!/usr/bin/env python3
"""
Check M (masking): when a +1 move has ~zero gain, can the MSCS pressure LEVEL tell
"idle" (resource really unused) from "masked" (its group is still binding, a partner
resource holds it down, so only a pair move helps)?  Scalar sees ΔCPI≈0 in both cases.

This script only COLLECTS data (analysis is analyze_masking.py, offline):
  states = the --max-states lowest best-single-gain states of check_prune (states_<bench>.json,
           same states as check_pairs)
  per state: config, CPI, area, MSCS (16); every +1 single (gain, Δarea, MSCS after, number of
  params actually changed by the validity projection); every +1/+1 pair (~210) exactly
  evaluated (gain, Δarea, synergy, params changed, whether the q-single was a no-op from c).
Writes results/masking_<bench>.json (refuses to overwrite).

Usage: python mscs_guide/check_masking.py --bench 641.leela_s
"""
import os, sys, json, time, argparse, itertools
from guide_common import HERE, PARAMS, _shift
import numpy as np
from common import Area, DEFAULT_CKPT, config_key
from refine_v3 import SurrogateWL
from mscs_reclaim_v2 import prefetch_areas


def n_diff(a, b):
    return int(sum(a[k] != b[k] for k in PARAMS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--states", default=None, help="default: results/states_<bench>.json")
    ap.add_argument("--max-states", type=int, default=10)
    ap.add_argument("--mcpat-workers", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    path = os.path.join(args.out, f"masking_{args.bench}.json")
    if os.path.exists(path):
        sys.exit(f"exists, refusing to overwrite: {path}")
    states = json.load(open(args.states or os.path.join(args.out, f"states_{args.bench}.json")))
    states = sorted(states, key=lambda s: s["best_gain"])[:args.max_states]

    S = SurrogateWL(DEFAULT_CKPT, [args.bench], "18k", device="cpu", max_slices=50)
    A = Area()
    pre = lambda cs: prefetch_areas(A, cs, args.mcpat_workers)
    t0 = time.time()
    rows = []
    for si, s in enumerate(states):
        c = s["cfg"]; cpi0, a0, m0 = S.cpi(c), A(c), S.mscs(c)
        sing, noop = {}, []
        nb = {}
        for n in PARAMS:
            cn = _shift(c, n, +1)
            if cn is None:
                noop.append(n)
            else:
                nb[n] = cn
        pre(list(nb.values()))
        for n, cn in nb.items():
            a = A(cn)
            if a is None:
                continue
            g = (cpi0 - S.cpi(cn)) / cpi0; dr = max(a - a0, 1e-6) / a0
            sing[n] = {"gain": g, "darea": dr, "eff": g / dr, "n_diff": n_diff(c, cn),
                       "mscs": S.mscs(cn).tolist()}
        cand = []
        for p, q in itertools.combinations(PARAMS, 2):
            c1 = _shift(c, p, +1)
            c2 = _shift(c1, q, +1) if c1 is not None else None
            if c2 is None or config_key(c2) == config_key(c1):
                c1 = _shift(c, q, +1)                      # try the other order
                c2 = _shift(c1, p, +1) if c1 is not None else None
                if c2 is None or config_key(c2) == config_key(c1):
                    continue
            cand.append((p, q, c2))
        pre([c2 for _, _, c2 in cand])
        pairs = []
        for p, q, c2 in cand:
            a = A(c2)
            if a is None:
                continue
            g = (cpi0 - S.cpi(c2)) / cpi0; dr = max(a - a0, 1e-6) / a0
            gp = sing[p]["gain"] if p in sing else 0.0
            gq = sing[q]["gain"] if q in sing else 0.0
            pairs.append({"p": p, "q": q, "gain": g, "darea": dr, "eff": g / dr, "syn": g - gp - gq,
                          "n_diff": n_diff(c, c2), "noop_single": [x for x in (p, q) if x in noop]})
        rows.append({"state": si, "cfg": c, "cpi": cpi0, "area": a0, "mscs": m0.tolist(),
                     "best_single_gain": max([v["gain"] for v in sing.values()] + [0.0]),
                     "singles": sing, "noop": noop, "pairs": pairs})
        bp = max(pairs, key=lambda x: x["gain"]) if pairs else None
        print(f"  state {si}: best_single={rows[-1]['best_single_gain']:.4f} pairs={len(pairs)} "
              f"best_pair={bp and (bp['p'], bp['q'], round(bp['gain'], 4))} q={S.n_queries} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        A.flush()
    json.dump({"bench": args.bench, "args": vars(args), "rows": rows, "queries": S.n_queries,
               "wall": time.time() - t0}, open(path, "w"), default=str)
    print(f"[{args.bench}] done -> {path} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
