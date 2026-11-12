#!/usr/bin/env python3
"""
End-to-end re-test of MSCS-grouped pruned Shapley (dse_exp/shapley_pruning.py) with the known
implementation problems fixed.  Ground truth = Monte-Carlo Shapley, 200 permutations (Concorde
protocol).  Nothing outside sweep/ is written; shapley_pruning.py is imported read-only and its
compute_cosine_matrix / group_params are used unchanged.

Kept from shapley_pruning.py
  value v(S): players in S at their TARGET (= base config value), others at MIN; Shapley marginal of
              p = v(before) - v(after) in CPI (positive = p improves CPI)
  grouping vector of p: 12-dim MSCS share change, base -> p at min (increase direction if already
              at min), cells FP/BP/SP x 4 stages; |cos| matrix; group_params(threshold=0.3)
  pruned:     per group, context = base with the group's players at min (others at target);
              exact within groups of <= 8 players, MC 10 permutations otherwise
  baselines:  matched-budget MC, one-at-a-time from the all-min config
Changed (each listed in PROTOCOL_shapley.md)
  1. every evaluated config goes through space17.project17 (= _project_to_valid + fixed L2/BP/fetch buf)
  2. W = fetch/decode/dispatch/commit width as ONE player (target = base widths, min = all 4)
  3. players not modelled by the features dropped (IntMultDiv, FP_MultDiv, wbWidth, l1i_assoc)
  4. 17-param space; bases = DEFAULT_CONFIG (projected) + 5 random; 6 benchmarks
  5. budget = number of DISTINCT model evaluations (cache); matched MC adds permutations until it has
     used at least as many distinct evaluations as pruned (incl. the grouping-vector evaluations)
  6. l1_size min = 16 (the original overrode it to 32 because 16 KB was not precomputed then)
Added controls
  scalar grouping: SAME group sizes as the MSCS grouping, filled with players in order of |main effect|
           (e from the same grouping evaluations, no MSCS): large-main-effect players share a group
  random grouping: same group sizes as MSCS grouping, random members, 10 draws (metrics averaged)
Truth check: a second 200-permutation MC (different seed) vs the truth.
Metrics vs truth: Spearman, top-3 overlap, weighted error sum|a-t| / sum|t|.
Outputs (refuse to overwrite): results/shapley_raw.json, results/shapley_check.txt
"""
import os, sys, json, time, itertools, math, argparse
from multiprocessing import get_context
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sweep_common as SC
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import shapley_pruning as SPR                          # read-only import (dse_exp)
from config_dse import DEFAULT_CONFIG

WIDTHS = ["fetchWidth", "decodeWidth", "dispatchWidth", "commitWidth"]
PLAYERS = ["W", "numROBEntries", "numIQEntries", "LQEntries", "SQEntries", "IntALU", "FP_ALU",
           "RdWrPort", "l1_size", "l1d_assoc"]
N_TRUTH, THR = 200, 0.3


def pmin(p):
    return 4 if p == "W" else min(SC.get_param_values(p))


class Game:
    """Value function on one base config; counts distinct model evaluations per method."""
    def __init__(self, M, base):
        self.M, self.base = M, base
        self.v_cache, self.n_coupled, self.n_cfg = {}, 0, 0

    def config(self, S, ctx_min=()):
        c = dict(self.base)
        for p in PLAYERS:
            if p not in S or p in ctx_min:
                if p == "W":
                    c.update({w: 4 for w in WIDTHS})
                else:
                    c[p] = pmin(p)
        return SC.S.project17(c)

    def v(self, S, used, ctx_min=()):
        cfg = self.config(S, ctx_min)
        k = SC.C.ckey(cfg)
        used.add(k)
        if k not in self.v_cache:
            self.v_cache[k] = self.M(cfg)["cpi_p"]
        return self.v_cache[k]


def mc_shapley(G, players, perms, rng, used, ctx_min=()):
    sv = {p: 0.0 for p in players}
    for _ in range(perms):
        order = rng.permutation(len(players)); S = set()
        prev = G.v(frozenset(S), used, ctx_min)
        for i in order:
            S.add(players[i]); cur = G.v(frozenset(S), used, ctx_min)
            sv[players[i]] += (prev - cur) / perms; prev = cur
    return sv


def exact_shapley(G, group, used):
    """Exact Shapley within `group`; players outside the group stay at target (base);
    group players not in the coalition are at min (shapley_pruning.shapley_exact_small)."""
    d = len(group); sv = {p: 0.0 for p in group}
    outside = [p for p in PLAYERS if p not in group]
    for p in group:
        rest = [q for q in group if q != p]
        for k in range(d):
            w = math.factorial(k) * math.factorial(d - k - 1) / math.factorial(d)
            for T in itertools.combinations(rest, k):
                S0 = frozenset(outside) | frozenset(T)
                sv[p] += w * (G.v(S0, used) - G.v(S0 | {p}, used))
    return sv


def pruned(G, groups, used):
    sv = {}
    for gi, g in enumerate(groups):
        if len(g) <= 8:
            sv.update(exact_shapley(G, g, used))
        else:
            outside = frozenset(p for p in PLAYERS if p not in g)
            rng = np.random.RandomState(42 + gi); part = {p: 0.0 for p in g}
            for _ in range(10):
                order = rng.permutation(len(g)); S = set(outside)
                prev = G.v(frozenset(S), used)
                for i in order:
                    S.add(g[i]); cur = G.v(frozenset(S), used)
                    part[g[i]] += (prev - cur) / 10; prev = cur
            sv.update(part)
    return sv


def matched_mc(G, budget, seed):
    used, rng = set(), np.random.RandomState(seed)
    sv_sum, n = {p: 0.0 for p in PLAYERS}, 0
    while len(used) < budget and n < 2000:
        one = mc_shapley(G, PLAYERS, 1, rng, used)
        for p in PLAYERS:
            sv_sum[p] += one[p]
        n += 1
    return {p: sv_sum[p] / n for p in PLAYERS}, len(used), n


def grouping_vectors(G, used):
    """shapley_pruning.compute_mscs_vectors on the projected game: base -> p at min."""
    allS = frozenset(PLAYERS); G.v(allS, used)
    m0 = G.M(G.config(allS))["share"]
    vec, main = {}, {}
    for p in PLAYERS:
        lowS = allS - {p}
        cl = G.config(lowS)
        if SC.C.ckey(cl) != SC.C.ckey(G.config(allS)):
            G.v(lowS, used)
            vec[p] = (G.M(cl)["share"] - m0)[SPR.MSCS_12_INDICES]
            main[p] = abs(G.v(lowS, used) - G.v(allS, used))
        else:                                              # already at min: increase direction
            ci = dict(G.base)
            if p == "W":
                ci.update({w: min(12, G.base[w] + 1) for w in WIDTHS})
            else:
                ci, _ = SPR.increase_param(G.base, p)
            ci = SC.S.project17(ci); used.add(SC.C.ckey(ci))
            mi = G.M(ci)
            vec[p] = (m0 - mi["share"])[SPR.MSCS_12_INDICES]
            main[p] = abs(G.M(G.config(allS))["cpi_p"] - mi["cpi_p"])
    return vec, main


def metrics(est, truth):
    a = np.array([est[p] for p in PLAYERS]); t = np.array([truth[p] for p in PLAYERS])
    top = lambda x: set(np.argsort(-x)[:3])
    return {"rho": float(spearmanr(a, t).correlation), "top3": len(top(a) & top(t)),
            "werr": float(np.abs(a - t).sum() / max(np.abs(t).sum(), 1e-12))}


def group_by_sim(sim, size_target):
    for thr in np.linspace(0.99, 0.0, 100):
        g = SPR.group_params(sim, PLAYERS, threshold=thr)
        if max(len(x) for x in g) >= size_target:
            return g, float(thr)
    return SPR.group_params(sim, PLAYERS, threshold=0.0), 0.0


def run_base(G, bi):
    truth = mc_shapley(G, PLAYERS, N_TRUTH, np.random.RandomState(1), set())
    truth2 = mc_shapley(G, PLAYERS, N_TRUTH, np.random.RandomState(2), set())
    res = {"truth": truth, "truth_check": metrics(truth2, truth)}
    used = set()
    vec, main = grouping_vectors(G, used)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SPR.compute_cosine_matrix(vec, PLAYERS)
        g_mscs = SPR.group_params(sim, PLAYERS, threshold=THR)
    sv = pruned(G, g_mscs, used)
    budget = len(used)
    res["pruned_mscs"] = {**metrics(sv, truth), "evals": budget, "groups": g_mscs}
    sizes = sorted((len(x) for x in g_mscs), reverse=True)
    order = sorted(PLAYERS, key=lambda p: -main[p]); g_sc, i = [], 0      # same sizes, filled by |main effect|
    for sz in sizes:
        g_sc.append(order[i:i + sz]); i += sz
    u2 = set(); grouping_vectors(G, u2); sv2 = pruned(G, g_sc, u2)
    res["pruned_scalar"] = {**metrics(sv2, truth), "evals": len(u2), "groups": g_sc}
    rnd = []
    for k in range(10):
        perm = list(np.random.RandomState(100 + k).permutation(PLAYERS)); gr, i = [], 0
        for sz in sizes:
            gr.append(perm[i:i + sz]); i += sz
        u3 = set(); grouping_vectors(G, u3); rnd.append({**metrics(pruned(G, gr, u3), truth), "evals": len(u3)})
    res["pruned_random"] = {k: float(np.mean([r[k] for r in rnd])) for k in ("rho", "top3", "werr", "evals")}
    mm, n_used, n_perm = matched_mc(G, budget, seed=7)
    res["matched_mc"] = {**metrics(mm, truth), "evals": n_used, "perms": n_perm}
    u4 = set(); v0 = G.v(frozenset(), u4)
    oaat = {p: v0 - G.v(frozenset({p}), u4) for p in PLAYERS}
    res["oaat"] = {**metrics(oaat, truth), "evals": len(u4)}
    res["main_effect_rank"] = metrics(main, truth)          # |base -> p at min|, ranking only
    return res


def run_bench(args):
    bench, n_rand, seed = args
    M = SC.Models(bench)
    bases = [SC.S.project17(dict(DEFAULT_CONFIG))] + SC.bases(n_rand, seed)
    out = []
    for bi, c in enumerate(bases):
        G = Game(M, c)
        r = run_base(G, bi); r.update({"bench": bench, "base": bi, "n_distinct_configs": len(G.v_cache)})
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rand", type=int, default=5)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    out = os.path.join(HERE, "results"); os.makedirs(out, exist_ok=True)
    raw_p, txt_p = os.path.join(out, "shapley_raw.json"), os.path.join(out, "shapley_check.txt")
    for p in (raw_p, txt_p):
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    t0 = time.time()
    with get_context("spawn").Pool(len(SC.BENCHES)) as pool:
        rows = [r for rs in pool.map(run_bench, [(b, args.n_rand, args.seed) for b in SC.BENCHES]) for r in rs]
    json.dump(rows, open(raw_p, "w"), default=float)
    L = [f"Shapley check: {len(SC.BENCHES)} benchmarks x {1+args.n_rand} bases, players {PLAYERS}, "
         f"truth = MC {N_TRUTH} perms [{time.time()-t0:.0f}s]"]
    tc = [r["truth_check"] for r in rows]
    L.append(f"truth convergence (2nd MC-{N_TRUTH} vs truth): rho {np.mean([x['rho'] for x in tc]):.3f} "
             f"(min {np.min([x['rho'] for x in tc]):.3f}), top3 {np.mean([x['top3'] for x in tc]):.2f}/3, "
             f"werr {np.mean([x['werr'] for x in tc]):.3f}")
    L.append(f"MSCS group sizes: " + str([sorted((len(g) for g in r['pruned_mscs']['groups']), reverse=True) for r in rows]))
    L.append(f"\n{'method':16s} {'rho':>6s} {'top3':>5s} {'werr':>6s} {'evals':>7s}   (mean over {len(rows)} bench x base)")
    for m in ["pruned_mscs", "pruned_scalar", "pruned_random", "matched_mc", "oaat", "main_effect_rank"]:
        x = [r[m] for r in rows]
        ev = f"{np.mean([y['evals'] for y in x]):7.0f}" if "evals" in x[0] else "      -"
        L.append(f"{m:16s} {np.mean([y['rho'] for y in x]):6.3f} {np.mean([y['top3'] for y in x]):5.2f} "
                 f"{np.mean([y['werr'] for y in x]):6.3f} {ev}")
    w = sum(r["pruned_mscs"]["rho"] > r["matched_mc"]["rho"] for r in rows)
    w2 = sum(r["pruned_mscs"]["rho"] > r["pruned_scalar"]["rho"] for r in rows)
    w3 = sum(r["pruned_mscs"]["werr"] < r["matched_mc"]["werr"] for r in rows)
    L.append(f"\npruned_mscs vs matched_mc: higher rho in {w}/{len(rows)}, lower werr in {w3}/{len(rows)}; "
             f"vs pruned_scalar: higher rho in {w2}/{len(rows)}")
    L.append("\nper benchmark (rho / top3):")
    for b in SC.BENCHES:
        x = [r for r in rows if r["bench"] == b]
        L.append(f"  {b:16s} " + "  ".join(f"{m}={np.mean([r[m]['rho'] for r in x]):.2f}/{np.mean([r[m]['top3'] for r in x]):.1f}"
                                         for m in ["pruned_mscs", "pruned_scalar", "matched_mc", "oaat"]))
    open(txt_p, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
