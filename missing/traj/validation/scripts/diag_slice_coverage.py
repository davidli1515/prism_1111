#!/usr/bin/env python3
"""Why does PRISM (traj surrogate) disagree with gem5 per benchmark?  Hypothesis: slice coverage.
The surrogate (traj_common.WLPredictor) averages the first 50 EVEN slice ids in LEXICOGRAPHIC order
(0, 10..18, 100..186), but every gem5 validation run simulates the contiguous range 0..186 (-I 18.7M after -W 20M).
For every distinct (bench, config) simulated in gem5_val..gem5_val7 this script recomputes PRISM CPI with
  m50   the surrogate's 50 slices (must reproduce WLPredictor),
  mall  all slices 0..186 (the instructions gem5 actually simulates),
  meven the even slices 0..186,
and reports per-benchmark mean signed error vs gem5.  Model, weights, normaliser: as WLPredictor (read-only).
Writes traj/results_calibrated/diag_slice_coverage.{txt,npz}; never overwrites."""
import os, sys, glob, json
import numpy as np, torch
from multiprocessing import Pool
TRAJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, TRAJ)
import traj_common as T
from dse_infer import DseEngine, load_precomputed
OUTD = os.path.join(TRAJ, "results_calibrated"); OUT = os.path.join(OUTD, "diag_slice_coverage")
SLICE, VALS = 100_000, ["gem5_val", "gem5_val2", "gem5_val3", "gem5_val4", "gem5_val5", "gem5_val6", "gem5_val7"]


def runs():
    R = {}
    for v in VALS:
        for f in glob.glob(os.path.join(TRAJ, v, "runs", "*", "*", "result.json")):
            d = json.load(open(f))
            if d.get("error") or not d.get("cpi"):
                continue
            c = T.S.project17({k: d["config"][k] for k in T.SEARCH})
            if any(c[k] != d["config"][k] for k in d["config"] if k in c):
                continue                                  # keep only runs whose config is a projected design
            R.setdefault(d["bench"], {}).setdefault((T.ckey(c), d["insts"]), (c, []))[1].append(d["cpi"])
    return R


def engine_multi(b, need):
    e = DseEngine.__new__(DseEngine); e.max_slices = 10**9; e.data, e.workload_names = {}, {b}
    for l1, sids in need.items():
        e.data[l1] = {b: [(s, load_precomputed(os.path.join(T.CM.PRECOMPUTE, f"l1_{l1}", b, f"slice_{s}", "dse_precomputed.npy"))) for s in sids]}
    return T.FastEngine(e)


def avail(b, l1):
    fs = sorted(glob.glob(os.path.join(T.CM.PRECOMPUTE, f"l1_{l1}", b, "slice_*", "dse_precomputed.npy")))
    return [int(os.path.basename(os.path.dirname(f)).split("_")[1]) for f in fs]       # lexicographic, as DseEngine


def run(item):
    b, D = item; torch.set_num_threads(1)
    l1s = sorted({c["l1_size"] for c, _ in D.values()}); A = {l1: avail(b, l1) for l1 in l1s}
    S50 = {l1: [i for i in A[l1] if i % 2 == 0][:50] for l1 in l1s}
    tops = {ins for (_, ins) in D}; need = {l1: sorted(set(range(max(tops) // SLICE)) & set(A[l1])) for l1 in l1s}
    eng = engine_multi(b, need)
    model = T.GEMModel(T.CM.GEM_INPUT_DIM, [512, 256], 0.0)
    model.load_state_dict(torch.load(T.CM.DEFAULT_CKPT, map_location="cpu")); model.eval(); norm = T.CM.get_normalizer("18k")
    ref = T.WLPredictor(b); out = []
    with torch.no_grad():
        for (k, ins), (c, g) in D.items():
            l1 = c["l1_size"]; sids = need[l1]
            X = np.stack([f[:T.CM.GEM_INPUT_DIM] for f in eng.get_features(c, b)])
            m = model(torch.from_numpy(norm.transform(X)).float())
            cs = torch.clamp(1.0 / (torch.clamp(m[:, T.CM.BASE_IDX], min=0.01) * float(T.compute_w_min(c)) + 1e-8), max=100.0).mean(1).numpy()
            pos = {s: i for i, s in enumerate(sids)}
            i50 = [pos[s] for s in S50[l1]]; rng = [pos[s] for s in sids if s < ins // SLICE]; iev = [pos[s] for s in sids if s < ins // SLICE and s % 2 == 0]
            cov = len(rng) / (ins // SLICE); full = np.full(400, np.nan); full[sids] = cs
            out.append((b, float(np.mean(g)), cs[i50].mean(), cs[rng].mean(), cs[iev].mean(), ref(c)[0], l1, cov, *full))
    print(b, len(out), flush=True)
    return out


def main():
    for e in (".txt", ".npz"):
        if os.path.exists(OUT + e):
            sys.exit(f"exists, refusing to overwrite: {OUT + e}")
    R = runs()
    with Pool(len(R)) as p:
        rows = [r for rs in p.map(run, sorted(R.items())) for r in rs]
    b = np.array([r[0] for r in rows]); v = np.array([r[1:8] for r in rows], float); per = np.array([r[8:] for r in rows], float)
    g, m50, mall, mev, mref = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4]
    L = [f"distinct (bench, config, insts) with gem5 CPI: {len(b)}; max |m50 - WLPredictor| = {np.abs(m50 - mref).max():.2e}; "
         f"min share of gem5 range with precompute {v[:, 6].min():.2f}",
         "signed error (model - gem5) / gem5, mean per benchmark; |err| = mean absolute",
         "bench            n  | 50 slices (used)  | all slices gem5 ran | even of those   | slice-set effect mall/m50-1"]
    def e(m, k): return (m[k] - g[k]) / g[k]
    for bb in sorted(set(b)):
        k = b == bb
        L.append(f"{bb:16s} {k.sum():3d} | {e(m50, k).mean():+6.1%} ({np.abs(e(m50, k)).mean():5.1%}) | {e(mall, k).mean():+6.1%} ({np.abs(e(mall, k)).mean():5.1%}) | "
                 f"{e(mev, k).mean():+6.1%} ({np.abs(e(mev, k)).mean():5.1%}) | {(mall[k] / m50[k] - 1).mean():+6.1%}")
    k = np.ones(len(b), bool)
    L.append(f"ALL              {len(b):3d} | {e(m50, k).mean():+6.1%} ({np.abs(e(m50, k)).mean():5.1%}) | {e(mall, k).mean():+6.1%} ({np.abs(e(mall, k)).mean():5.1%}) | "
             f"{e(mev, k).mean():+6.1%} ({np.abs(e(mev, k)).mean():5.1%})")
    for name, m in (("50 slices", m50), ("gem5 range", mall)):
        bias = np.array([e(m, b == bb).mean() for bb in sorted(set(b))])
        L.append(f"{name}: benchmarks with |mean bias| > 8%: {(np.abs(bias) > 0.08).sum()}/17; mean |bias| {np.abs(bias).mean():.1%}")
    print("\n".join(L)); os.makedirs(OUTD, exist_ok=True); open(OUT + ".txt", "w").write("\n".join(L) + "\n")
    np.savez_compressed(OUT + ".npz", bench=b, gem5=g, m50=m50, mall=mall, meven=mev, l1=v[:, 5], coverage=v[:, 6], per_slice=per)


if __name__ == "__main__":
    main()
