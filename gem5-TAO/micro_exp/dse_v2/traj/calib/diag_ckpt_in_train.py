#!/usr/bin/env python3
"""Is the checkpoint used by the traj surrogate / gem5 validation (dse_exp/precompute, validate_joint_dse.BENCH ticks)
the one PRISM was trained on (18k packed set)?  Compare trace-only feature columns:
  columns constant within every training window (vary only across windows) AND identical across two very
  different configs in the DSE engine.  Each training window's trace-only vector is matched to the nearest
  precompute slice (0..399, L1 = 64 KB); an exact match (rel. diff < 1e-5) means the same instructions.
Read-only. Writes traj/results_calibrated/diag_ckpt_in_train.txt (never overwrites)."""
import os, sys, glob, numpy as np
TRAJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, TRAJ)
sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/window_ids")
import traj_common as T
from make_simpoint_ids import window_ids_in_order
from dse_infer import DseEngine, load_precomputed
from multiprocessing import Pool
OUT = os.path.join(TRAJ, "results_calibrated", "diag_ckpt_in_train.txt"); D = T.CM.GEM_INPUT_DIM


def run(b):
    P = np.asarray(np.load(f"/scratch2/wl730/micro_traindata_packed/gem/{b}.npy", mmap_mode="r")[:, :D], np.float64)
    w = window_ids_in_order(np.asarray(np.load(f"/scratch2/wl730/micro_traindata_packed/gem/{b}.npy", mmap_mode="r"), np.float32))
    W = np.unique(w); within = np.zeros(D)
    for k in W:
        within = np.maximum(within, P[w == k].std(0))
    trace = (within < 1e-9) & (P.std(0) > 1e-9)
    l1 = 64; sids = list(range(400))
    e = DseEngine.__new__(DseEngine); e.max_slices = 10**9; e.workload_names = {b}
    e.data = {l1: {b: [(s, load_precomputed(os.path.join(T.CM.PRECOMPUTE, f"l1_{l1}", b, f"slice_{s}", "dse_precomputed.npy")))
                       for s in sids if os.path.exists(os.path.join(T.CM.PRECOMPUTE, f"l1_{l1}", b, f"slice_{s}", "dse_precomputed.npy"))]}}
    eng = T.FastEngine(e)
    c1 = T.S.project17({p: (T.ALL_PARAMS[p][0] if isinstance(T.ALL_PARAMS[p], list) else T.ALL_PARAMS[p][0]) for p in T.SEARCH} | {"l1_size": l1})
    c2 = T.S.project17({p: (T.ALL_PARAMS[p][-1] if isinstance(T.ALL_PARAMS[p], list) else T.ALL_PARAMS[p][1]) for p in T.SEARCH} | {"l1_size": l1})
    F1 = np.stack(eng.get_features(c1, b))[:, :D].astype(np.float64); F2 = np.stack(eng.get_features(c2, b))[:, :D].astype(np.float64)
    cols = np.where(trace & (np.abs(F1 - F2).max(0) < 1e-9))[0]
    if len(cols) == 0:
        return f"{b:16s} no trace-only columns"
    Tw = np.stack([P[w == k][0, cols] for k in W]); S = F1[:, cols]
    sc = np.maximum(np.abs(np.concatenate([Tw, S])).max(0), 1e-12)
    d = (np.abs(Tw[:, None] / sc - S[None] / sc)).max(2); best = d.min(1)
    exact = (best < 1e-5).mean(); med = np.median(best)
    selfd = np.abs(S[:, None] / sc - S[None] / sc).max(2); np.fill_diagonal(selfd, np.inf)
    return (f"{b:16s} trace-only cols {len(cols):4d} | training windows {len(W):4d} | exact match to a DSE slice {exact:6.1%} | "
            f"median nearest dist {med:.2e} (slice-to-slice median {np.median(selfd.min(1)):.2e})")


if __name__ == "__main__":
    if os.path.exists(OUT):
        sys.exit(f"exists, refusing to overwrite: {OUT}")
    B = sorted(os.listdir(os.path.join(T.CM.PRECOMPUTE, "l1_64")))
    with Pool(len(B)) as p:
        L = p.map(run, B)
    print("\n".join(L)); os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w").write("\n".join(L) + "\n")
