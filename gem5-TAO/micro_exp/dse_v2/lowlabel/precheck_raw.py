#!/usr/bin/env python3
"""
Same setting as precheck_rank.py, but stores RAW predictions so every ranking definition can be
analysed offline (analyze_precheck.py).  Per model, base config and +1 move:
  cpi            mean over slices of the 4-stage-mean CPI (compute_cpi_mean, as in training eval)
  cpi_stage (4)  mean over slices of each stage's CPI 1/(Base_s W)
  share (16)     unweighted slice mean of the MSCS shares
  share_w (16)   CPI-weighted shares: sum_slices cs_s * m / sum_slices cs_s per stage
  cyc (16)       cycle components: cpi_stage_s * share_w (CPI per instruction spent in each cell)
Concorde: cpi only.  Writes lowlabel/precheck_raw.json (refuses to overwrite).
"""
import os, sys, json, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from precheck_rank import S, C, FastEngine, GEMModel, ConcordeModel, load, load_all_benchmarks, \
    create_exp1_dataloaders, GEM_INPUT_DIM, CON_DIM, MODELS, EXP4, BENCHES, N_BASE
from model import compute_cpi_mean

BASE_IDX = [0, 4, 8, 12]


@torch.no_grad()
def raw(kind, m, Xg, Xc, w):
    if kind == "Concorde":
        return {"cpi": float(m(Xc).mean())}
    o = m(Xg)
    cpi = compute_cpi_mean(o, torch.full((len(o),), float(w))).mean().item()
    cs = 1.0 / (o[:, BASE_IDX] * w + 1e-8)                     # (n,4) per-stage CPI per slice
    share = o.mean(0).numpy()
    share_w = np.zeros(16); cyc = np.zeros(16)
    for s in range(4):
        idx = [4 * s + k for k in range(4)]
        share_w[idx] = ((cs[:, s:s + 1] * o[:, idx]).sum(0) / cs[:, s].sum()).numpy()
        cyc[idx] = cs[:, s].mean().item() * share_w[idx]
    return {"cpi": cpi, "cpi_stage": cs.mean(0).numpy().tolist(), "share": share.tolist(),
            "share_w": share_w.tolist(), "cyc": cyc.tolist()}


def main():
    out = os.path.join(HERE, "precheck_raw.json")
    if os.path.exists(out):
        sys.exit(f"exists, refusing to overwrite: {out}")
    t0 = time.time()
    data = load_all_benchmarks("/scratch2/wl730/micro_traindata_packed/gem",
                               "/scratch2/wl730/micro_traindata_packed/concord", __import__("config").BENCHMARK_NAMES)
    _, _, gnorm, cnorm, *_ = create_exp1_dataloaders(data, seed=42); del data
    models = {f"{k}_{r}": (k, load(k, os.path.join(EXP4, p))) for k, r, p in MODELS}
    _, _, R = C.load_original_model()
    rng = np.random.RandomState(7)                              # same bases as precheck_rank.py
    recs = []
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
            rec = {"bench": b, "base_i": bi, "base": base, "moves": [n for n, _ in moves], "pred": {}}
            for cfg in [base] + [c for _, c in moves]:
                X = np.stack([f[:GEM_INPUT_DIM] for f in eng.get_features(cfg, b)])
                Xg = torch.from_numpy(gnorm.transform(X)).float()
                Xc = torch.from_numpy(cnorm.transform(X[:, :CON_DIM])).float()
                w = float(C.compute_w_min(cfg))
                for key, (k, m) in models.items():
                    rec["pred"].setdefault(key, []).append(raw(k, m, Xg, Xc, w))
            recs.append(rec)
        print(f"  {b} done [{time.time()-t0:.0f}s]", flush=True)
    json.dump(recs, open(out, "w"))
    print("saved", out)


if __name__ == "__main__":
    main()
