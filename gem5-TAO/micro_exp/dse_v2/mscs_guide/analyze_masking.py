#!/usr/bin/env python3
"""
Offline analysis of results/masking_<bench>.json (check_masking.py). No queries.

Physical prior: MSCS category -> resource group (CPI = 1/(Base*W_min), so Base ~ width-bound)
  Base -> WID : fetch/decode/dispatch/commit/wb widths
  FP   -> FE  : fetchWidth, fetchBufferSize, l1i_assoc, l1_size
  BP   -> BE  : ROB, IQ, LQ, SQ, FUs, RdWrPort, l1_size, l1d_assoc, l2_size, l2_assoc, wbWidth
  SP   -> SPEC: bp_type
Group pressure P_G = sum over the 4 stages of that category's MSCS share.

A pair is "constrained" if the validity projection changed other params (n_diff > 2) or one
single was a no-op on its own (e.g. l1d_assoc capped by l1_size): that is projection masking,
reported separately from performance masking.

Per state, b = best single gain.
  H1  within-group synergy vs group pressure: is the highest-pressure group the one with the
      most synergistic within-group pairs?  Spearman(P_G, within-G synergy) over states x groups
  H2  dead params (0 < ... gain < 0.25 b, not no-op): label = has a partner (unconstrained pair
      with synergy > 0.5 b and gain > b).  AUC of: P_G of its group (MSCS level), ||dMSCS||_1 of
      its single (the old change-based signal), its single gain, -its area cost
  H3  pair-search policies: queries spent, best pair gain / oracle pair gain, beats best single
"""
import os, sys, json, glob
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = {
    "WID": (0, ["fetchWidth", "decodeWidth", "dispatchWidth", "commitWidth", "wbWidth"]),
    "FE": (1, ["fetchWidth", "fetchBufferSize", "l1i_assoc", "l1_size"]),
    "BE": (2, ["numROBEntries", "numIQEntries", "LQEntries", "SQEntries", "IntALU", "IntMultDiv",
               "FP_ALU", "FP_MultDiv", "RdWrPort", "l1_size", "l1d_assoc", "l2_size", "l2_assoc", "wbWidth"]),
    "SPEC": (3, ["bp_type"]),
}


def gp(m, g):
    cat = GROUPS[g][0]
    return float(sum(m[4 * s + cat] for s in range(4)))


def groups_of(n):
    return [g for g, (_, ps) in GROUPS.items() if n in ps]


def within(x, g):
    ps = GROUPS[g][1]
    if len(ps) == 1:
        return ps[0] in (x["p"], x["q"])
    return x["p"] in ps and x["q"] in ps


def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or (~y).all():
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def constrained(x):
    return x["n_diff"] > 2 or bool(x["noop_single"])


def main():
    files = sorted(glob.glob(os.path.join(HERE, "results", "masking_*.json")))
    H1, H2, H3 = [], [], []
    rng = np.random.RandomState(0)
    for f in files:
        z = json.load(open(f)); bench = z["bench"]
        for r in z["rows"]:
            m = np.array(r["mscs"]); b = r["best_single_gain"]; sing = r["singles"]
            pairs = r["pairs"]
            if not pairs or b <= 0:
                continue
            free = [x for x in pairs if not constrained(x)]
            P = {g: gp(m, g) for g in GROUPS}
            top = max(P, key=P.get)
            # ---- H1
            for g in GROUPS:
                w = [x["syn"] / b for x in free if within(x, g)]
                if w:
                    H1.append({"bench": bench, "g": g, "P": P[g], "top": g == top,
                               "syn_mean": float(np.mean(w)), "syn_max": float(np.max(w)),
                               "frac_pos": float(np.mean([s > 0.5 for s in w]))})
            cross = [x["syn"] / b for x in free if not any(within(x, g) for g in GROUPS)]
            H1.append({"bench": bench, "g": "cross", "P": np.nan, "top": False,
                       "syn_mean": float(np.mean(cross)) if cross else np.nan,
                       "syn_max": float(np.max(cross)) if cross else np.nan,
                       "frac_pos": float(np.mean([s > 0.5 for s in cross])) if cross else np.nan})
            # ---- H2
            for n, v in sing.items():
                if not (v["gain"] < 0.25 * b):
                    continue
                inv = [x for x in free if n in (x["p"], x["q"])]
                lab = any(x["syn"] > 0.5 * b and x["gain"] > b for x in inv)
                inv_all = [x for x in pairs if n in (x["p"], x["q"])]
                lab_all = any(x["syn"] > 0.5 * b and x["gain"] > b for x in inv_all)
                H2.append({"bench": bench, "n": n, "y": lab, "y_all": lab_all,
                           "P": max([P[g] for g in groups_of(n)] + [0.0]),
                           "dm": float(np.abs(np.array(v["mscs"]) - m).sum()),
                           "gain": v["gain"], "neg_area": -v["darea"]})
            # ---- H3
            og = max(x["gain"] for x in pairs)
            def res(sel):
                if not sel:
                    return 0, 0.0, False
                bg = max(x["gain"] for x in sel)
                return len(sel), bg / og if og > 0 else np.nan, bg > b
            pol = {}
            pol["mscs_top"] = res([x for x in pairs if within(x, top)])
            top2 = sorted(P, key=P.get)[-2:]
            pol["mscs_top2"] = res([x for x in pairs if any(within(x, g) for g in top2)])
            rg = [res([x for x in pairs if within(x, g)]) for g in GROUPS]
            pol["rand_group"] = tuple(np.mean([t[i] for t in rg]) for i in range(3))
            top4 = sorted([n for n in sing if sing[n]["gain"] > 0], key=lambda n: -sing[n]["gain"])[:4]
            pol["scalar_top4"] = res([x for x in pairs if x["p"] in top4 and x["q"] in top4])
            k = max(pol["mscs_top"][0], 1)
            rr = [res([pairs[i] for i in rng.choice(len(pairs), min(k, len(pairs)), replace=False)])
                  for _ in range(200)]
            pol["random_same_n"] = tuple(np.mean([t[i] for t in rr]) for i in range(3))
            pol["all_pairs"] = (len(pairs), 1.0, og > b)
            H3.append({"bench": bench, "top": top, "pol": pol,
                       "oracle_constrained": constrained(max(pairs, key=lambda x: x["gain"])),
                       "pair_over_single": og / b})

    print("=== data:", len(files), "benches,", len({(h['bench']) for h in H3}), "with states;",
          len(H3), "states ===\n")
    print("Oracle pair gain / best single gain: median %.2f   oracle pair is constrained (projection) in %.0f%% of states"
          % (np.median([h["pair_over_single"] for h in H3]), 100 * np.mean([h["oracle_constrained"] for h in H3])))
    print("Top-pressure group counts:", {g: sum(h["top"] == g for h in H3) for g in GROUPS})

    print("\n=== H1  within-group synergy (unconstrained pairs, synergy in units of best single gain) ===")
    for g in list(GROUPS) + ["cross"]:
        rows = [h for h in H1 if h["g"] == g]
        if not rows:
            continue
        t = [h for h in rows if h["top"]]; o = [h for h in rows if not h["top"]]
        fmt = lambda rr: "%.2f/%.2f (n=%d)" % (np.nanmean([h["syn_mean"] for h in rr]), np.nanmean([h["frac_pos"] for h in rr]), len(rr)) if rr else "-"
        print(f"  {g:5s} when top-pressure: mean syn/frac>0.5b {fmt(t):22s} otherwise {fmt(o)}")
    gi = [h for h in H1 if h["g"] != "cross"]
    rho = spearmanr([h["P"] for h in gi], [h["syn_max"] for h in gi])
    print(f"  Spearman(P_G, max within-G synergy) over states x groups: {rho.correlation:.2f} (p={rho.pvalue:.3f}, n={len(gi)})")
    hit = []
    for h3 in H3:
        hs = [h for h in H1 if h["bench"] == h3["bench"] and h["g"] != "cross"]
    # per state: is the group with max within synergy the top-pressure one
    by_state = {}
    for h in gi:
        by_state.setdefault((h["bench"], id(h) // 1), None)

    print("\n=== H2  dead single (gain < 0.25 b): idle vs masked ===")
    for key in ["y", "y_all"]:
        y = [h[key] for h in H2]
        print(f"  label={'unconstrained partner' if key == 'y' else 'any partner (incl. projection)'}: "
              f"n={len(H2)} masked={np.mean(y):.2f}  AUC: "
              + "  ".join(f"{s}={auc(y, [h[s] for h in H2]):.2f}" for s in ["P", "dm", "gain", "neg_area"]))
    print("  per bench (unconstrained label, AUC of P / dm):")
    for bch in sorted({h["bench"] for h in H2}):
        hh = [h for h in H2 if h["bench"] == bch]; y = [h["y"] for h in hh]
        print(f"    {bch:16s} n={len(hh):3d} masked={np.mean(y):.2f}  P={auc(y, [h['P'] for h in hh]):.2f}  dm={auc(y, [h['dm'] for h in hh]):.2f}")
    from collections import Counter
    print("  most frequent masked params:", Counter(h["n"] for h in H2 if h["y"]).most_common(8))

    print("\n=== H3  pair-search policies: queries / best-of-set gain vs oracle pair / beats best single ===")
    for name in ["mscs_top", "mscs_top2", "rand_group", "scalar_top4", "random_same_n", "all_pairs"]:
        v = np.array([h["pol"][name] for h in H3], float)
        print(f"  {name:14s} queries {v[:,0].mean():6.1f}   gain/oracle {np.nanmean(v[:,1]):.2f}   beats single {v[:,2].mean():.2f}")
    print("  per bench gain/oracle (mscs_top | rand_group | scalar_top4 | random_same_n):")
    for bch in sorted({h["bench"] for h in H3}):
        hh = [h for h in H3 if h["bench"] == bch]
        print(f"    {bch:16s} " + " | ".join("%.2f" % np.nanmean([h["pol"][k][1] for h in hh])
                                            for k in ["mscs_top", "rand_group", "scalar_top4", "random_same_n"]))


if __name__ == "__main__":
    main()
