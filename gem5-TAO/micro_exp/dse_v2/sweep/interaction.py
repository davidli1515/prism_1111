#!/usr/bin/env python3
"""
Items 1 + 3 - masking / interaction at SWEEP scale (replaces the +1-stride masking, pair-move and
decouple analyses).  Needs results/measurable.json (resolution_screen.py).

Truth (teacher = PRISM 100%, gem_full.pt): for base c and measurable parameters p, q (others at c):
  G_p|q=v  = CPI(p=min, q=v) - CPI(p=max, q=v)               (benefit of p, q held at v)
  I_pq     = (G_p|q=max - G_p|q=min) / CPI(c)                 (2nd-order sweep difference, symmetric)
  I > 0: complements / masking (p pays off only once q is large) ; I < 0: substitutes / overlap
  strong  iff |I| >= 1% (fixed before running)
Cheap predictors (only the two single-parameter endpoints per parameter, i.e. 2 evals/param):
  cos_share  |cos(v_p, v_q)|, v = 12-dim pressure-share change between the endpoints
             (dse_exp/shapley_pruning.py grouping signal, here on min/max endpoints)
  cos_cyc    same on CPI-weighted cycle changes
  prod_cpi   |dCPI_p| * |dCPI_q|        (scalar-only baseline: big main effects interact more)
  min_cpi    min(|dCPI_p|, |dCPI_q|)    (scalar-only baseline)
Per-parameter (item 3, decouple): cancel = sum|v| - |sum v| on the sweep vector vs the parameter's
  strongest interaction max_q |I_pq| and strongest complement max_q I_pq; scalar baseline |dCPI_p|.
Also: Concorde's own I_pq vs the teacher's (model-agreement reference).
Raw -> results/interaction_raw.json ; report -> results/interaction.txt (refuse to overwrite).
"""
import os, sys, json, time, argparse, itertools
from multiprocessing import get_context
import numpy as np
from scipy.stats import spearmanr, rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sweep_common as SC

STRONG = 0.01


def corner(c, p, vp, q, vq):
    return SC.S.project17(dict(c, **{p: vp, q: vq}))


def run_bench(args):
    bench, n_base, seed, params = args
    M = SC.Models(bench)
    vals = {p: (min(SC.get_param_values(p)), max(SC.get_param_values(p))) for p in params}
    out = []
    for bi, c in enumerate(SC.bases(n_base, seed)):
        f0 = M(c)
        single = {}
        for p in params:
            e = SC.endpoints(c, p)
            if e is None:
                continue
            L, H = M(e[0]), M(e[1])
            single[p] = {"dP": (L["cpi_p"] - H["cpi_p"]) / f0["cpi_p"], "dC": (L["cpi_c"] - H["cpi_c"]) / f0["cpi_c"],
                         "v_share": (L["share"] - H["share"])[SC.PRESSURE].tolist(),
                         "v_cyc": (L["cyc"] - H["cyc"])[SC.PRESSURE].tolist()}
        pairs = []
        for p, q in itertools.combinations(sorted(single), 2):
            F = {}
            for a, va in zip("lh", vals[p]):
                for b, vb in zip("lh", vals[q]):
                    F[a + b] = M(corner(c, p, va, q, vb))
            Ip = ((F["lh"]["cpi_p"] - F["hh"]["cpi_p"]) - (F["ll"]["cpi_p"] - F["hl"]["cpi_p"])) / f0["cpi_p"]
            Ic = ((F["lh"]["cpi_c"] - F["hh"]["cpi_c"]) - (F["ll"]["cpi_c"] - F["hl"]["cpi_c"])) / f0["cpi_c"]
            pairs.append({"p": p, "q": q, "I": Ip, "I_concorde": Ic})
        out.append({"bench": bench, "base": bi, "config": c, "single": single, "pairs": pairs})
    return out


def cos(a, b):
    a, b = np.asarray(a), np.asarray(b); na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else abs(a @ b) / (na * nb)


def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def analyse(recs):
    L = []
    P = []
    for r in recs:
        S = r["single"]
        for x in r["pairs"]:
            a, b = S[x["p"]], S[x["q"]]
            P.append({"key": (r["bench"], r["base"]), "p": x["p"], "q": x["q"], "I": x["I"], "Ic": x["I_concorde"],
                      "cos_share": cos(a["v_share"], b["v_share"]), "cos_cyc": cos(a["v_cyc"], b["v_cyc"]),
                      "prod_cpi": abs(a["dP"]) * abs(b["dP"]), "min_cpi": min(abs(a["dP"]), abs(b["dP"]))})
    absI = np.array([x["I"] for x in P]); strong = np.abs(absI) >= STRONG
    L.append(f"pairs {len(P)}  strong (|I|>={STRONG:.0%}) {strong.mean():.2f}  "
             f"complements {np.mean(absI >= STRONG):.2f}  substitutes {np.mean(absI <= -STRONG):.2f}  "
             f"median |I| {100*np.median(np.abs(absI)):.2f}%")
    L.append(f"Concorde vs teacher interaction: Spearman {spearmanr([x['Ic'] for x in P], absI).correlation:.2f}  "
             f"sign agreement on strong pairs {np.mean(np.sign([x['Ic'] for x in P])[strong] == np.sign(absI[strong])):.2f}")
    L.append("\n[1] which pairs interact?  score vs |I| (pooled):  Spearman | AUC(strong)")
    for s in ["cos_share", "cos_cyc", "prod_cpi", "min_cpi"]:
        sc = np.array([x[s] for x in P])
        L.append(f"  {s:10s} {spearmanr(sc, np.abs(absI)).correlation:+.2f} | {auc(strong, sc):.2f}")
    L.append("    within-base Spearman (mean):  " + "  ".join(
        f"{s}={np.nanmean([spearmanr([x[s] for x in P if x['key']==k], [abs(x['I']) for x in P if x['key']==k]).correlation for k in {x['key'] for x in P}]):+.2f}"
        for s in ["cos_share", "cos_cyc", "prod_cpi", "min_cpi"]))
    L.append("  partner hit: for each parameter, is its highest-score partner its strongest-|I| partner?")
    for s in ["cos_share", "cos_cyc", "prod_cpi", "min_cpi"]:
        hits, chance = [], []
        for k in {x["key"] for x in P}:
            px = [x for x in P if x["key"] == k]
            for prm in {x["p"] for x in px} | {x["q"] for x in px}:
                cand = [x for x in px if prm in (x["p"], x["q"])]
                if len(cand) < 2 or max(abs(x["I"]) for x in cand) < STRONG:
                    continue
                hits.append(max(cand, key=lambda x: x[s]) is max(cand, key=lambda x: abs(x["I"])))
                chance.append(1 / len(cand))
        L.append(f"  {s:10s} hit {np.mean(hits):.2f}  (chance {np.mean(chance):.2f}, n={len(hits)})")

    L.append("\n[3] decouple: per parameter, cancel score of its sweep vector vs its interactions")
    rows = []
    for r in recs:
        for prm, a in r["single"].items():
            I = [x["I"] for x in r["pairs"] if prm in (x["p"], x["q"])]
            if not I:
                continue
            v = np.array(a["v_cyc"]); vs = np.array(a["v_share"])
            rows.append({"cancel_cyc": np.abs(v).sum() - abs(v.sum()), "cancel_share": np.abs(vs).sum() - abs(vs.sum()),
                         "absdP": abs(a["dP"]), "maxabsI": max(abs(i) for i in I), "maxposI": max(I)})
    for tgt in ["maxabsI", "maxposI"]:
        y = np.array([x[tgt] for x in rows])
        L.append(f"  target {tgt}: " + "  ".join(f"{s} rho={spearmanr([x[s] for x in rows], y).correlation:+.2f}"
                                                for s in ["cancel_cyc", "cancel_share", "absdP"]))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=12)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    out = os.path.join(HERE, "results")
    raw_p, txt_p = os.path.join(out, "interaction_raw.json"), os.path.join(out, "interaction.txt")
    for p in (raw_p, txt_p):
        if os.path.exists(p):
            sys.exit(f"exists, refusing to overwrite: {p}")
    params = json.load(open(os.path.join(out, "measurable.json")))["measurable"]
    if len(params) < 3:
        sys.exit(f"too few measurable parameters: {params}")
    t0 = time.time()
    with get_context("spawn").Pool(len(SC.BENCHES)) as pool:
        recs = [r for rs in pool.map(run_bench, [(b, args.n_base, args.seed, params) for b in SC.BENCHES]) for r in rs]
    json.dump(recs, open(raw_p, "w"))
    rep = f"interaction at sweep scale, params {params}, {len(recs)} bases [{time.time()-t0:.0f}s]\n" + analyse(recs)
    open(txt_p, "w").write(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
