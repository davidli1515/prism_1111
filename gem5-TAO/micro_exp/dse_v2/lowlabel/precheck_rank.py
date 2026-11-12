#!/usr/bin/env python3
"""
Pre-check (no simulation): when CPI accuracy is poor (few labels), is the RANKING of single-parameter
moves still right?  Teacher = PRISM trained with 100% labels (exp4 results_77 full_100pct);
the gem5 version of this check (true per-parameter ranking) is a separate script.

Base configs: random points of the 17-parameter space (dse17/space17.py), 6 benchmarks.
Moves: +1 stride of each of the 16 refine parameters (exactly what a DSE step compares).
Models (exp4/results_77, 18k): PRISM full_{5,10,20,100}, supervised_only_{5,10,20}, Concorde {5,10,20,100}.
Per model and ranking method, against the teacher's Delta-CPI ranking of the 16 moves:
  Spearman, top-1 hit, teacher-best within top-3, pairwise order accuracy (pairs the teacher separates
  by > 0.1% CPI), plus the model's own |CPI - teacher CPI|.
Ranking methods: scalar = predicted Delta-CPI (any model);
  mscs_dom  = reduction of the dominant pressure type (4-stage mean; dse_engine._coordinated_step);
  mscs_eyer = reduction of the dominant type at its own stage (FP@Decode, BP@Commit, SP@Decode).
Writes lowlabel/precheck_rank.json (refuses to overwrite).
"""
import os, sys, json, time, itertools
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "dse17"))
import space17 as S                                   # sets dse_exp / exp1 / coord_ablation paths
C = S.C
from fast_infer import FastEngine
from model import GEMModel, ConcordeModel, compute_cpi_mean
from dataset import load_all_benchmarks, create_exp1_dataloaders
from config import GEM_INPUT_DIM, DROPOUT
from scipy.stats import spearmanr

EXP4 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "exp4", "results_77")
BENCHES = ["602.gcc_s", "605.mcf_s", "623.xalancbmk_s", "638.imagick_s", "641.leela_s", "648.exchange2_s"]
N_BASE = 8
CON_DIM = 3873
FP, BP, SP = [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]
MODELS = ([("PRISM", r, f"exp4_18k/checkpoints/full_{r}pct.pt") for r in (5, 10, 20, 100)]
          + [("V2sup", r, f"exp4_18k_supervised_only/checkpoints/supervised_only_{r}pct.pt") for r in (5, 10, 20)]
          + [("Concorde", r, f"exp4_18k_concorde/checkpoints/concorde_{r}pct.pt") for r in (5, 10, 20, 100)])


def load(kind, path):
    if kind == "Concorde":
        m = ConcordeModel(CON_DIM, [512, 256], DROPOUT)
    else:
        m = GEMModel(GEM_INPUT_DIM, [512, 256], DROPOUT)
    m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
    return m


@torch.no_grad()
def predict(kind, m, Xg, Xc, w):
    if kind == "Concorde":
        return float(m(Xc).mean()), None
    o = m(Xg)
    cpi = compute_cpi_mean(o, torch.full((len(o),), float(w))).mean().item()
    return cpi, o.mean(0).numpy()


def rank_scores(kind, base, moves):
    """scores: higher = better move."""
    cb, mb = base
    out = {"scalar": np.array([(cb - c) / cb for c, _ in moves])}
    if kind != "Concorde":
        pres = {"FP": mb[FP].mean(), "BP": mb[BP].mean(), "SP": mb[SP].mean()}
        dom = max(pres, key=pres.get); I = {"FP": FP, "BP": BP, "SP": SP}[dom]
        out["mscs_dom"] = np.array([np.mean(mb[I] - m[I]) for _, m in moves])
        j = {"FP": 1, "BP": 14, "SP": 3}[dom]
        out["mscs_eyer"] = np.array([mb[j] - m[j] for _, m in moves])
    return out


def compare(score, teacher):
    rho = spearmanr(score, teacher).correlation
    best = int(np.argmax(teacher))
    top1 = float(np.argmax(score) == best)
    top3 = float(best in np.argsort(-score)[:3])
    pairs = [(i, j) for i, j in itertools.combinations(range(len(teacher)), 2) if abs(teacher[i] - teacher[j]) > 1e-3]
    pw = float(np.mean([(score[i] > score[j]) == (teacher[i] > teacher[j]) for i, j in pairs])) if pairs else np.nan
    return rho, top1, top3, pw


def main():
    out_path = os.path.join(HERE, "precheck_rank.json")
    if os.path.exists(out_path):
        sys.exit(f"exists, refusing to overwrite: {out_path}")
    t0 = time.time()
    data = load_all_benchmarks("/scratch2/wl730/micro_traindata_packed/gem",
                               "/scratch2/wl730/micro_traindata_packed/concord",
                               __import__("config").BENCHMARK_NAMES)
    _, _, gnorm, cnorm, *_ = create_exp1_dataloaders(data, seed=42)
    del data
    print(f"normalizers ready [{time.time()-t0:.0f}s]", flush=True)
    models = {(k, r): load(k, os.path.join(EXP4, p)) for k, r, p in MODELS}
    _, _, R = C.load_original_model()
    rng = np.random.RandomState(7)
    rows = []
    for b in BENCHES:
        eng = FastEngine(C.single_workload_engine(R.PRECOMPUTE_BASE, b))
        for bi in range(N_BASE):
            base = S.sample17(rng)
            moves = []
            for n in C.GRADIENT_PARAMS:
                v, _ = C._increase_by_stride(base, n)
                if v is not None:
                    cn = S.project17(dict(base, **{n: v}))
                    if C.ckey(cn) != C.ckey(base):
                        moves.append((n, cn))
            preds = {}
            for cfg in [base] + [c for _, c in moves]:
                X = np.stack([f[:GEM_INPUT_DIM] for f in eng.get_features(cfg, b)])
                Xg = torch.from_numpy(gnorm.transform(X)).float()
                Xc = torch.from_numpy(cnorm.transform(X[:, :CON_DIM])).float()
                w = float(C.compute_w_min(cfg))
                for key, m in models.items():
                    preds.setdefault(key, []).append(predict(key[0], m, Xg, Xc, w))
            T = preds[("PRISM", 100)]
            t_scores = rank_scores("PRISM", T[0], T[1:])["scalar"]
            for key in models:
                p = preds[key]
                cpi_err = float(np.mean([abs(a[0] - t[0]) / t[0] for a, t in zip(p, T)]))
                for meth, sc in rank_scores(key[0], p[0], p[1:]).items():
                    rows.append({"bench": b, "base": bi, "model": key[0], "ratio": key[1], "method": meth,
                                 "cpi_err_vs_teacher": cpi_err, **dict(zip(["rho", "top1", "top3", "pairwise"],
                                                                           compare(sc, t_scores)))})
        print(f"  {b} done [{time.time()-t0:.0f}s]", flush=True)
    json.dump(rows, open(out_path, "w"), indent=1)
    print("\nagreement with the PRISM-100% Delta-CPI ranking of the 16 single-parameter moves "
          f"({len(BENCHES)} benchmarks x {N_BASE} bases)")
    print(f"  {'model':9s} {'lbl%':>4s} {'method':10s} {'CPI err':>8s} {'Spearman':>8s} {'top1':>5s} {'top3':>5s} {'pairwise':>8s}")
    for k, r, _ in MODELS:
        for meth in ["scalar", "mscs_dom", "mscs_eyer"]:
            x = [q for q in rows if q["model"] == k and q["ratio"] == r and q["method"] == meth]
            if not x:
                continue
            g = lambda f: np.nanmean([q[f] for q in x])
            print(f"  {k:9s} {r:4d} {meth:10s} {100*g('cpi_err_vs_teacher'):7.2f}% {g('rho'):8.2f} "
                  f"{g('top1'):5.2f} {g('top3'):5.2f} {g('pairwise'):8.2f}")


if __name__ == "__main__":
    main()
