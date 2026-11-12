#!/usr/bin/env python3
"""
Check B: on low-gain (plateau-like) states, can MSCS propose two-parameter moves that a
coordinate-wise search misses?

States: the --max-states lowest best-single-gain states visited by check_prune.py.
At each state all valid pairs (+1 on p and +1 on q, ~210) are evaluated exactly (oracle).
Proposals of k pairs, judged by the best of the k after evaluation:
  mscs_cover      top-k by covered pressure  sum_i m_i (1-(1-R_pi)(1-R_qi))
  mscs_2b         two bottlenecks: c1 = top pressure, c2 = top pressure of another
                  category; p from top-3 relievers of c1, q from top-3 of c2,
                  ranked by m_c1 R[p,c1] + m_c2 R[q,c2]
  scalar_eff      pairs of the top-4 singles (already evaluated by the scalar step),
                  ranked by (g_p+g_q)/(da_p+da_q)  (additive prediction, no MSCS)
  scalar_gain     same, ranked by g_p+g_q
  random          k random pairs (mean over 50 draws)
Reported per state: oracle pair vs best single (does a pair move exist that is worth it?),
synergy (g_pq - g_p - g_q) of the oracle, and for every proposal method the best eff / gain
relative to the oracle pair and whether it beats the best single move.

Usage: python mscs_guide/check_pairs.py --bench 638.imagick_s --map mscs_guide/maps/map_n40_s0.npz
"""
import os, sys, json, time, argparse, itertools
from guide_common import (HERE, PARAMS, load_map, pair_cover, top_pressures, eval_inc, _shift,
                          comp_name)
import numpy as np
from common import Area, DEFAULT_CKPT, config_key
from refine_v3 import SurrogateWL
from mscs_reclaim_v2 import prefetch_areas


def judge(pairs, sel, ref):
    """sel: chosen pair dicts; ref: dict of oracle / single references."""
    if not sel:
        return {"eff_ratio": 0.0, "gain_ratio": 0.0, "beats_single_eff": False, "beats_single_gain": False}
    be = max(x["eff"] for x in sel); bg = max(x["gain"] for x in sel)
    return {"eff_ratio": be / ref["pair_eff"] if ref["pair_eff"] > 0 else None,
            "gain_ratio": bg / ref["pair_gain"] if ref["pair_gain"] > 0 else None,
            "beats_single_eff": bool(be > ref["single_eff"]), "beats_single_gain": bool(bg > ref["single_gain"]),
            "synergy_pos": int(sum(x["syn"] > 0 for x in sel))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--map-mode", default="loo", choices=["loo", "own", "all"])
    ap.add_argument("--states", default=None, help="default: results/states_<bench>.json")
    ap.add_argument("--max-states", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mcpat-workers", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    path = os.path.join(args.out, f"pairs_{args.bench}.json")
    if os.path.exists(path):
        sys.exit(f"exists, refusing to overwrite: {path}")
    states = json.load(open(args.states or os.path.join(args.out, f"states_{args.bench}.json")))
    states = sorted(states, key=lambda s: s["best_gain"])[:args.max_states]

    R, _ = load_map(args.map, args.bench, args.map_mode)
    pidx = {n: i for i, n in enumerate(PARAMS)}
    S = SurrogateWL(DEFAULT_CKPT, [args.bench], "18k", device="cpu", max_slices=50)
    A = Area()
    pre = lambda cs: prefetch_areas(A, cs, args.mcpat_workers)
    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    rows = []
    for si, s in enumerate(states):
        c = s["cfg"]; cpi0, a0, m = S.cpi(c), A(c), S.mscs(c)
        singles = {x["name"]: x for x in eval_inc(c, S, A, pre)}
        cand = []
        for p, q in itertools.combinations(PARAMS, 2):
            c1 = _shift(c, p, +1)
            c2 = _shift(c1, q, +1) if c1 is not None else None
            if c2 is None or config_key(c2) == config_key(c1):
                continue
            cand.append((p, q, c2))
        pre([c2 for _, _, c2 in cand])
        pairs = []
        for p, q, c2 in cand:
            a = A(c2)
            if a is None:
                continue
            g = (cpi0 - S.cpi(c2)) / cpi0
            dr = max(a - a0, 1e-6) / a0
            gp = singles[p]["gain"] if p in singles else np.nan
            gq = singles[q]["gain"] if q in singles else np.nan
            pairs.append({"p": p, "q": q, "gain": g, "darea": dr, "eff": g / dr, "syn": g - gp - gq})
        pos_s = [x for x in singles.values() if x["gain"] > 0]
        pos_p = [x for x in pairs if x["gain"] > 0]
        if not pos_p or not pos_s:
            continue
        op = max(pos_p, key=lambda x: x["eff"])
        ref = {"single_eff": max(x["eff"] for x in pos_s), "single_gain": max(x["gain"] for x in pos_s),
               "pair_eff": op["eff"], "pair_gain": max(x["gain"] for x in pos_p)}

        k = args.k
        props = {}
        props["mscs_cover"] = sorted(pairs, key=lambda x: -pair_cover(m, R, pidx[x["p"]], pidx[x["q"]]))[:k]
        c1, c2 = top_pressures(m)
        r1 = [PARAMS[i] for i in np.argsort(-R[:, c1])[:3]]
        r2 = [PARAMS[i] for i in np.argsort(-R[:, c2])[:3]]
        pool = [x for x in pairs if (x["p"] in r1 and x["q"] in r2) or (x["q"] in r1 and x["p"] in r2)]
        sc2 = lambda x: max(m[c1] * R[pidx[a], c1] + m[c2] * R[pidx[b], c2]
                            for a, b in [(x["p"], x["q"]), (x["q"], x["p"])])
        props["mscs_2b"] = sorted(pool, key=lambda x: -sc2(x))[:k]
        top4 = [x["name"] for x in sorted(pos_s, key=lambda x: -x["eff"])[:4]]
        tp = [x for x in pairs if x["p"] in top4 and x["q"] in top4]
        props["scalar_eff"] = sorted(tp, key=lambda x: -(singles[x["p"]]["gain"] + singles[x["q"]]["gain"])
                                     / (singles[x["p"]]["darea"] + singles[x["q"]]["darea"]))[:k]
        props["scalar_gain"] = sorted(tp, key=lambda x: -(singles[x["p"]]["gain"] + singles[x["q"]]["gain"]))[:k]
        res = {name: judge(pairs, sel, ref) for name, sel in props.items()}
        rnd = [judge(pairs, [pairs[i] for i in rng.choice(len(pairs), k, replace=False)], ref) for _ in range(50)]
        res["random"] = {key: float(np.mean([r[key] for r in rnd if r[key] is not None]))
                         for key in ("eff_ratio", "gain_ratio", "beats_single_eff", "beats_single_gain")}
        top_syn = sorted(pairs, key=lambda x: -x["syn"] if np.isfinite(x["syn"]) else np.inf)[:10]
        rows.append({"state": si, "best_single_gain": s["best_gain"], "n_pairs": len(pairs),
                     "c1": comp_name(c1), "c2": comp_name(c2), "relievers_c1": r1, "relievers_c2": r2,
                     "ref": ref, "oracle_pair": op, "max_syn": top_syn[0],
                     "frac_syn_pos": float(np.mean([x["syn"] > 0 for x in pairs])),
                     "pair_vs_single_eff": op["eff"] / ref["single_eff"],
                     "pair_vs_single_gain": ref["pair_gain"] / ref["single_gain"],
                     "props": {n: [(x["p"], x["q"]) for x in v] for n, v in props.items()},
                     "top_syn_hit": {n: sum((x["p"], x["q"]) in {(y["p"], y["q"]) for y in top_syn} for x in v)
                                     for n, v in props.items()},
                     "res": res})
        print(f"  state {si}: best_single={s['best_gain']:.4f} pairs={len(pairs)} c1={comp_name(c1)} "
              f"c2={comp_name(c2)} eff(mscs_2b)={res['mscs_2b']['eff_ratio']} "
              f"eff(scalar_eff)={res['scalar_eff']['eff_ratio']} [{time.time()-t0:.0f}s]", flush=True)
        A.flush()
    json.dump({"bench": args.bench, "args": vars(args), "rows": rows, "wall": time.time() - t0},
              open(path, "w"), indent=1, default=str)
    print(f"[{args.bench}] done -> {path} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
