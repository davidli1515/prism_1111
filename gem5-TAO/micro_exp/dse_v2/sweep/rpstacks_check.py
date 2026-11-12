#!/usr/bin/env python3
"""
RpStacks-inspired check: does the MSCS "relief vector" similarity of two parameters predict their
INTERACTION, and in which direction?

  H_group   (dse_exp/shapley_pruning.py): similar relief vectors (high |cos|)  -> parameters interact
  H_rpstack (RpStacks max-of-linear view): parameters relieving the SAME bottleneck are additive;
            interaction (masking/complements) arises between parameters relieving DIFFERENT
            competing bottlenecks -> LOW cos / different dominant component -> interaction

PROTOCOL (fixed before running; see PROTOCOL_rpstacks.md for the plain-language version)
Model / data
  teacher  PRISM 100% labels (exp1_v2 gem_full.pt), original DSE pipeline (sweep_common.Models):
           CPI = commit-stage CPI averaged over GEMPredictor's 200 slices; bit-exact fast features.
  reference Concorde 100% (exp1_v2 concorde.pt) on the same configs, only to measure how model-
           dependent the "truth" is.
  space    17-param space (L2 1MB/8-way, TAGE, fetch buffer 32 fixed), 6 benchmarks x 6 random bases.
Players (10)
  W        fetchWidth = decodeWidth = dispatchWidth = commitWidth moved TOGETHER (4 <-> 12):
           W_min is a min() of these, so they are one physical player (and moving one alone makes
           illegal or W_min-neutral configs).
  numROBEntries, numIQEntries, LQEntries, SQEntries, IntALU, FP_ALU, RdWrPort, l1_size, l1d_assoc
  EXCLUDED (not modelled by the features, see resolution screen / analytical_models.cpp):
           IntMultDiv, FP_MultDiv, wbWidth (never read by the analytical model),
           l1i_assoc (its distribution is frozen in the DSE precompute).
  Each player has a lo / hi value = parameter min / max (W: 4 / 12).
Legality
  Every evaluated config goes through space17.project17 (= dse_engine._project_to_valid + fixed
  params). If projection changes a player's own value or any OTHER free parameter (e.g. ROB raised
  to 2*dispatch+IQ+LQ+SQ, L1d assoc capped by L1 size), that corner is "coupled"; a pair with any
  coupled corner is flagged. Results are reported for ALL pairs and for CLEAN pairs only.
Truth (per base c, pair p,q; others stay at c)
  I_pq = [CPI(p_lo,q_lo) - CPI(p_hi,q_lo) - CPI(p_lo,q_hi) + CPI(p_hi,q_hi)] / CPI(c)
       = (benefit of p when q is hi) - (benefit of p when q is lo), relative to CPI(c)
  I > 0  complements / masking (p pays off only once q is also raised)
  I < 0  substitutes / overlap (raising one makes the other less useful)
  strong |I| >= 1%  ;  strong complement I >= +1%          (thresholds fixed)
Cheap signals (2 evaluations per player: p_lo and p_hi with others at c)
  dCPI_p   = (CPI(p_lo) - CPI(p_hi)) / CPI(c)                         main effect
  v_p      = MSCS(p_lo) - MSCS(p_hi) on the 12 pressure cells (FP,BP,SP x 4 stages)
             (a) share version (exactly the shapley_pruning vector, but lo/hi endpoints)
             (b) cycle version = CPI-weighted cells (share alone hides CPI changes)
  MSCS signals : cos(v_p,v_q) signed, |cos|, same_dom = [argmax cell of v_p == argmax cell of v_q]
  scalar-only baselines : |dCPI_p|*|dCPI_q| , min(|dCPI_p|,|dCPI_q|)
Metrics (pooled over benchmarks x bases, and per benchmark)
  Spearman(signal, I) and Spearman(signal, |I|)
  AUC(signal -> strong |I|), AUC(signal -> strong complement); AUC < 0.5 means the OPPOSITE direction
  H_group predicts  |cos| up  -> |I| up       (AUC > 0.5)
  H_rpstack predicts cos down -> complement up (AUC of cos for complements < 0.5; same_dom < 0.5)
  An MSCS signal is useful only if it beats BOTH scalar baselines.
  Model dependence: Spearman(I_PRISM, I_Concorde), sign agreement on pairs with |I_PRISM| >= 1%.
Outputs (refuse to overwrite): results/rpstacks_raw.json, results/rpstacks.txt
"""
import os, sys, json, time, itertools, argparse
from multiprocessing import get_context
import numpy as np
from scipy.stats import spearmanr, rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sweep_common as SC

WIDTHS = ["fetchWidth", "decodeWidth", "dispatchWidth", "commitWidth"]
PLAYERS = ["W", "numROBEntries", "numIQEntries", "LQEntries", "SQEntries", "IntALU", "FP_ALU",
           "RdWrPort", "l1_size", "l1d_assoc"]
STRONG = 0.01


def _range(p):
    if p == "W":
        return 4, 12
    v = SC.get_param_values(p)
    return min(v), max(v)


def _set(c, p, v):
    return {**c, **({w: v for w in WIDTHS} if p == "W" else {p: v})}


def _members(p):
    return WIDTHS if p == "W" else [p]


def legal(c, assign):
    """assign: {player: value}. Returns (projected config, coupled flag)."""
    raw = dict(c)
    for p, v in assign.items():
        raw = _set(raw, p, v)
    out = SC.S.project17(raw)
    own = [m for p in assign for m in _members(p)]
    coupled = any(out[m] != raw[m] for m in own) or \
        any(out[k] != c[k] for k in SC.PARAMS if k not in own and k != "wbWidth")   # wb unmodelled
    return out, coupled


def run_bench(args):
    bench, n_base, seed = args
    M = SC.Models(bench)
    rows = []
    for bi, c in enumerate(SC.bases(n_base, seed)):
        f0 = M(c)["cpi_p"]; f0c = M(c)["cpi_c"]
        single = {}
        for p in PLAYERS:
            lo, hi = _range(p)
            (cl, kl), (ch, kh) = legal(c, {p: lo}), legal(c, {p: hi})
            L, H = M(cl), M(ch)
            single[p] = {"dCPI": (L["cpi_p"] - H["cpi_p"]) / f0, "coupled": bool(kl or kh),
                         "v_share": (L["share"] - H["share"])[SC.PRESSURE].tolist(),
                         "v_cyc": (L["cyc"] - H["cyc"])[SC.PRESSURE].tolist()}
        for p, q in itertools.combinations(PLAYERS, 2):
            (pl, ph), (ql, qh) = _range(p), _range(q)
            F, flag = {}, False
            for a, va in (("l", pl), ("h", ph)):
                for b, vb in (("l", ql), ("h", qh)):
                    cc, k = legal(c, {p: va, q: vb}); flag |= k
                    F[a + b] = M(cc)
            I = (F["ll"]["cpi_p"] - F["hl"]["cpi_p"] - F["lh"]["cpi_p"] + F["hh"]["cpi_p"]) / f0
            Ic = (F["ll"]["cpi_c"] - F["hl"]["cpi_c"] - F["lh"]["cpi_c"] + F["hh"]["cpi_c"]) / f0c
            rows.append({"bench": bench, "base": bi, "p": p, "q": q, "I": I, "I_concorde": Ic,
                         "coupled": bool(flag or single[p]["coupled"] or single[q]["coupled"]),
                         "sp": single[p], "sq": single[q]})
    return rows


def cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or not y.any():
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def signals(r):
    out = {}
    for kind in ("share", "cyc"):
        vp, vq = r["sp"]["v_" + kind], r["sq"]["v_" + kind]
        c = cos(vp, vq)
        out[f"cos_{kind}"] = c
        out[f"abscos_{kind}"] = abs(c)
        out[f"samedom_{kind}"] = float(int(np.argmax(np.abs(vp))) == int(np.argmax(np.abs(vq))))
    a, b = abs(r["sp"]["dCPI"]), abs(r["sq"]["dCPI"])
    out["scalar_prod"] = a * b
    out["scalar_min"] = min(a, b)
    return out


def report(rows, title):
    L = [f"\n== {title}: {len(rows)} pairs  (strong |I|>=1%: {np.mean([abs(r['I'])>=STRONG for r in rows]):.2f}, "
         f"strong complement I>=+1%: {np.mean([r['I']>=STRONG for r in rows]):.2f}, "
         f"strong substitute I<=-1%: {np.mean([r['I']<=-STRONG for r in rows]):.2f})"]
    I = np.array([r["I"] for r in rows]); S = [signals(r) for r in rows]
    L.append(f"  {'signal':16s} {'rho(s,I)':>9s} {'rho(s,|I|)':>10s} {'AUC |I|':>8s} {'AUC compl':>9s} {'AUC subst':>9s}")
    for k in S[0]:
        s = np.array([x[k] for x in S])
        L.append(f"  {k:16s} {spearmanr(s, I).correlation:+9.2f} {spearmanr(s, np.abs(I)).correlation:+10.2f} "
                 f"{auc(np.abs(I) >= STRONG, s):8.2f} {auc(I >= STRONG, s):9.2f} {auc(I <= -STRONG, s):9.2f}")
    Ic = np.array([r["I_concorde"] for r in rows]); big = np.abs(I) >= STRONG
    L.append(f"  model dependence: Spearman(I_PRISM, I_Concorde) {spearmanr(I, Ic).correlation:+.2f}; "
             f"sign agreement on |I_PRISM|>=1%: {np.mean(np.sign(I[big]) == np.sign(Ic[big])) if big.any() else float('nan'):.2f}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=6)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    out = os.path.join(HERE, "results"); os.makedirs(out, exist_ok=True)
    raw_p, txt_p = os.path.join(out, "rpstacks_raw.json"), os.path.join(out, "rpstacks.txt")
    for p in (raw_p, txt_p):
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    t0 = time.time()
    with get_context("spawn").Pool(len(SC.BENCHES)) as pool:
        rows = [r for rs in pool.map(run_bench, [(b, args.n_base, args.seed) for b in SC.BENCHES]) for r in rs]
    json.dump(rows, open(raw_p, "w"))
    L = [f"RpStacks check: {len(SC.BENCHES)} benchmarks x {args.n_base} bases, players {PLAYERS} [{time.time()-t0:.0f}s]"]
    clean = [r for r in rows if not r["coupled"]]
    L += report(rows, "ALL pairs")
    L += report(clean, "CLEAN pairs (no projection coupling)")
    for b in SC.BENCHES:
        L += report([r for r in clean if r["bench"] == b], f"CLEAN, {b}")
    # which player pairs carry the strong interactions
    from collections import Counter
    cnt = Counter((r["p"], r["q"]) for r in clean if abs(r["I"]) >= STRONG)
    L.append(f"\nmost frequent strong-|I| pairs (clean): {cnt.most_common(10)}")
    L.append(f"pairs flagged as coupled: {np.mean([r['coupled'] for r in rows]):.2f}; by player: " +
             str(Counter(x for r in rows if r["coupled"] for x in (r["p"], r["q"])).most_common()))
    open(txt_p, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
