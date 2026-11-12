#!/usr/bin/env python3
"""Feature parity: for 18k training rows of benchmarks whose precompute checkpoint is in training, recompute the
4681 input features with the traj DSE engine (FastEngine, matched slice, the row's own config) and compare
(i) features, (ii) PRISM CPI on training features vs on engine features, (iii) the label.
Rows restricted to configs the DSE engine can express (renameWidth = dispatchWidth; issue = dispatch always).
Read-only. Writes traj/results_calibrated/diag_feature_parity.txt (never overwrites)."""
import os, sys, numpy as np, torch
TRAJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, TRAJ); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traj_common as T
from gem5_slice_pilot import KEYS
from make_simpoint_ids import window_ids_in_order, ARCH_START, CFG_TAGS
from dse_infer import DseEngine, load_precomputed
from multiprocessing import Pool
OUT = os.path.join(TRAJ, "results_calibrated", "diag_feature_parity.txt"); D = T.CM.GEM_INPUT_DIM; NROW = 60
B = ["508.namd_r", "600.perlbench_s", "602.gcc_s", "605.mcf_s", "607.cactuBSSN_s", "619.lbm_s", "620.omnetpp_s", "621.wrf_s",
     "641.leela_s", "644.nab_s", "648.exchange2_s", "654.roms_s", "657.xz_s"]


def run(b):
    torch.set_num_threads(1)
    F = np.load(f"/scratch2/wl730/micro_traindata_packed/gem/{b}.npy", mmap_mode="r"); A = np.asarray(F, np.float32)
    P = A[:, :D].astype(np.float64); w = window_ids_in_order(A); W = np.unique(w); within = np.zeros(D)
    for k in W:
        within = np.maximum(within, P[w == k].std(0))
    trace = (within < 1e-9) & (P.std(0) > 1e-9)
    cfgs = [{KEYS[t]: int(round(v)) for t, v in zip(CFG_TAGS, r[ARCH_START:ARCH_START + len(CFG_TAGS)])} for r in A]
    ok = [i for i, c in enumerate(cfgs) if c["renameWidth"] == c["dispatchWidth"] and c["l1_size"] in (16, 32, 64, 128, 256)]
    rows = list(np.random.RandomState(0).choice(ok, min(NROW, len(ok)), replace=False))
    l1s = sorted({cfgs[i]["l1_size"] for i in rows}); e = DseEngine.__new__(DseEngine); e.max_slices = 10**9; e.workload_names = {b}; e.data = {}
    for l1 in l1s:
        e.data[l1] = {b: [(s, load_precomputed(os.path.join(T.CM.PRECOMPUTE, f"l1_{l1}", b, f"slice_{s}", "dse_precomputed.npy")))
                          for s in range(400) if os.path.exists(os.path.join(T.CM.PRECOMPUTE, f"l1_{l1}", b, f"slice_{s}", "dse_precomputed.npy"))]}
    eng = T.FastEngine(e); sid = {l1: [s for s, _ in e.data[l1][b]] for l1 in l1s}
    model = T.GEMModel(D, [512, 256], 0.0); model.load_state_dict(torch.load(T.CM.DEFAULT_CKPT, map_location="cpu")); model.eval(); norm = T.CM.get_normalizer("18k")
    res = []
    for i in rows:
        c = cfgs[i]; l1 = c["l1_size"]; cd = T.S.project17({k: c[k] for k in T.SEARCH})
        cd = dict(cd, **{k: c[k] for k in c if k in cd})                     # keep the row's own fixed params where the engine has them
        Fe = np.stack(eng.get_features(cd, b))[:, :D].astype(np.float64)
        cols = np.where(trace)[0]; sc = np.maximum(np.abs(Fe[:, cols]).max(0), 1e-12)
        d = np.abs(P[i, cols] / sc - Fe[:, cols] / sc).max(1); j = int(d.argmin())
        if d[j] > 1e-5:
            continue
        x = np.stack([P[i], Fe[j]]); 
        with torch.no_grad():
            m = model(torch.from_numpy(norm.transform(x)).float())
        wmin = float(T.compute_w_min(cd)); wrow = min(c[k] for k in ("fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth", "issueWidth", "commitWidth"))
        cpi = (1.0 / (torch.clamp(m[:, T.CM.BASE_IDX], min=0.01) * torch.tensor([[wrow], [wmin]]) + 1e-8)).clamp(max=100).mean(1).numpy()
        lab = 1.0 / (A[i, 4681 + 12] * wrow)
        rel = np.abs(Fe[j] - P[i]) / np.maximum(np.abs(P[i]), 1e-6)
        res.append((cpi[0], cpi[1], lab, (rel > 1e-3).mean(), wrow == wmin, c["bp_type"]))
    r = np.array(res, float)
    if len(r) == 0:
        return f"{b:16s} no matched rows"
    e_tr = (r[:, 0] - r[:, 2]) / r[:, 2]; e_en = (r[:, 1] - r[:, 2]) / r[:, 2]; gap = r[:, 1] / r[:, 0] - 1
    return (f"{b:16s} rows {len(r):3d} | model on training features vs label {e_tr.mean():+6.1%} (|{np.abs(e_tr).mean():4.1%}|) | "
            f"on engine features vs label {e_en.mean():+6.1%} (|{np.abs(e_en).mean():4.1%}|) | engine/training CPI {gap.mean():+6.1%} | "
            f"share of features differing >0.1%: {r[:, 3].mean():.1%} | wmin agrees {r[:, 4].mean():.0%} | TAGE rows {r[:, 5].mean():.0%}")


if __name__ == "__main__":
    if os.path.exists(OUT):
        sys.exit(f"exists, refusing to overwrite: {OUT}")
    with Pool(len(B)) as p:
        L = p.map(run, B)
    print("\n".join(L)); open(OUT, "w").write("\n".join(L) + "\n")
