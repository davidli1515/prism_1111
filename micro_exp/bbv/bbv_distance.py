#!/usr/bin/env python3
"""
Train/test similarity in SimPoint terms: for every test window, the Manhattan
distance (range [0, 2]) between its normalized BBV and the nearest training
window of the same benchmark.

Protocols (SPEC benchmarks with o3 traces; 18k windows mapped to slices):
  L1   row split      exp1/exp4 split (RandomState(42) over sorted benchmarks, 15/16 rows).
                      A test row whose own window has training rows has distance 0;
                      we report that fraction and the NN distance to *other* windows.
  L1.5 window split   window_ids/window_split.split_by_window (as exp2/train_ood.py).
  L1.5b block split   hold out the last contiguous 1/16 of slices (25 of 400).
  L2   unseen simpoint windows of other checkpoints (simpletrace) vs all 400 18k windows.
References: adjacent slices (k, k+1); random window pairs within the benchmark.
gcc/lbm: only the traced checkpoint (18k windows 0-399) is used.

Also: check that the traced checkpoint is the 18k one, by Spearman correlation between
per-slice trace stats (ctrl/mem fraction) and config-invariant 18k feature columns.

Usage: python bbv_distance.py [--proj15]
"""
import os, sys, glob, json, pickle, hashlib, argparse
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

HERE = os.path.dirname(os.path.abspath(__file__))
MICRO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MICRO, "window_ids"))
from window_split import load_window_ids, split_by_window

PACKED_18K = "/data/micro_traindata_packed/gem"
SEED, TRAIN_RATIO = 42, 15 / 16
SLICE_OF_WINDOW = sorted(range(400), key=lambda k: f"slice_{k}")   # packed (lexicographic) order


def stable_hash(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)


def load_bbvs(bm):
    """-> (M_o3 [400 x D] indexed by slice, {ckpt: M [n x D]}, stats_o3)"""
    srcs = {os.path.basename(p)[len(bm) + 2:-4]: pickle.load(open(p, "rb"))
            for p in glob.glob(os.path.join(HERE, "raw", f"{bm}__*.pkl"))}
    keys = sorted({pc for d in srcs.values() for w in d["windows"] for pc in w})
    col = {pc: i for i, pc in enumerate(keys)}
    def mat(windows):
        M = np.zeros((len(windows), len(keys)))
        for r, w in enumerate(windows):
            for pc, c in w.items():
                M[r, col[pc]] = c
        return M / M.sum(1, keepdims=True)
    o3 = mat(srcs["o3"]["windows"])
    unseen = {s[2:]: mat(d["windows"]) for s, d in srcs.items() if s.startswith("st")}
    return o3, unseen, srcs["o3"]["stats"]


def manhattan(A, B):
    return cdist(A, B, metric="cityblock")


def summarize(d):
    d = np.asarray(d, float)
    return {"n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)),
            "p90": float(np.percentile(d, 90))}


def row_split_test_windows(names, sizes):
    """Reproduce exp1/exp4 row split; return {bm: (test_rows, train_rows)}."""
    rng = np.random.RandomState(SEED)
    out = {}
    for n in sorted(names):
        perm = rng.permutation(sizes[n]); s = int(sizes[n] * TRAIN_RATIO)
        out[n] = (perm[s:], perm[:s])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj15", action="store_true", help="also report SimPoint 15-d random projection")
    args = ap.parse_args()

    all_names = sorted(f[:-4] for f in os.listdir(PACKED_18K) if f.endswith(".npy"))
    sizes = {n: np.load(os.path.join(PACKED_18K, f"{n}.npy"), mmap_mode="r").shape[0] for n in all_names}
    rows = row_split_test_windows(all_names, sizes)
    inv_cols = np.load(os.path.join(HERE, "trace_invariant_cols.npy"))
    bms = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(os.path.join(HERE, "raw", "*__o3.pkl"))})

    results = {}
    for bm in bms:
        o3, unseen, stats = load_bbvs(bm)
        wid = load_window_ids(bm)
        nwin = 400                                   # traced checkpoint windows
        in_ckpt = wid < nwin
        # window w -> BBV row (slice)
        W = o3[[SLICE_OF_WINDOW[w] for w in range(nwin)]]
        D = manhattan(W, W)
        np.fill_diagonal(D, np.inf)
        r = {}

        # --- check traced ckpt == 18k ckpt -----------------------------------------
        X = np.load(os.path.join(PACKED_18K, f"{bm}.npy"), mmap_mode="r")
        first_row = np.array([np.argmax(wid == w) for w in range(nwin)])
        F = np.asarray(X[first_row][:, inv_cols])
        tstat = np.array([[stats[SLICE_OF_WINDOW[w]]["frac_ctrl"], stats[SLICE_OF_WINDOW[w]]["frac_mem"]]
                          for w in range(nwin)])
        best = 0.0
        for j in range(F.shape[1]):
            if F[:, j].std() == 0: continue
            for t in range(2):
                rho = spearmanr(F[:, j], tstat[:, t]).correlation
                if np.isfinite(rho): best = max(best, abs(rho))
        r["ckpt_check_max_spearman"] = best

        # --- L1 row split ------------------------------------------------------------
        te_rows, tr_rows = rows[bm]
        te_w = np.unique(wid[te_rows][wid[te_rows] < nwin])
        tr_w = np.unique(wid[tr_rows][wid[tr_rows] < nwin])
        r["L1_row_frac_own_window_in_train"] = float(np.isin(wid[te_rows][in_ckpt[te_rows]], tr_w).mean())
        r["L1_row_nn_excl_own"] = summarize(D[np.ix_(te_w, tr_w)].min(1))

        # --- L1.5 window split -------------------------------------------------------
        rng = np.random.RandomState(SEED + stable_hash(bm))
        tr_idx, te_idx = split_by_window(wid, TRAIN_RATIO, rng)
        te_w = np.unique(wid[te_idx]); te_w = te_w[te_w < nwin]
        tr_w = np.unique(wid[tr_idx]); tr_w = tr_w[tr_w < nwin]
        r["L1.5_window"] = summarize(D[np.ix_(te_w, tr_w)].min(1))

        # --- L1.5b contiguous block (last 25 slices) --------------------------------
        te_s = np.arange(375, 400); tr_s = np.arange(0, 375)
        Ds = manhattan(o3, o3)
        r["L1.5b_block"] = summarize(Ds[np.ix_(te_s, tr_s)].min(1))

        # --- L2 unseen simpoint -------------------------------------------------------
        if unseen:
            U = np.concatenate(list(unseen.values()))
            r["L2_unseen_simpoint"] = summarize(manhattan(U, o3).min(1))
            r["L2_ckpts"] = list(unseen)

        # --- references ---------------------------------------------------------------
        r["ref_adjacent_slices"] = summarize(np.abs(o3[1:] - o3[:-1]).sum(1))
        iu = np.triu_indices(len(o3), 1)
        r["ref_random_pairs"] = summarize(Ds[iu])

        if args.proj15:
            P = np.random.RandomState(0).uniform(-1, 1, size=(o3.shape[1], 15))
            Z = o3 @ P
            Dz = manhattan(Z, Z); np.fill_diagonal(Dz, np.inf)
            Zw = Z[[SLICE_OF_WINDOW[w] for w in range(nwin)]]
            Dzw = manhattan(Zw, Zw); np.fill_diagonal(Dzw, np.inf)
            r["proj15_L1.5_window"] = summarize(Dzw[np.ix_(te_w, tr_w)].min(1))
            if unseen:
                r["proj15_L2_unseen"] = summarize(manhattan(U @ P, Z).min(1))
        results[bm] = r

        L2 = r.get("L2_unseen_simpoint", {}).get("median", float("nan"))
        print(f"{bm:18s} ckpt-check rho={best:.2f} | L1 own-window-in-train={r['L1_row_frac_own_window_in_train']*100:5.1f}% "
              f"| median NN: L1(excl own)={r['L1_row_nn_excl_own']['median']:.3f} "
              f"L1.5 win={r['L1.5_window']['median']:.3f} L1.5b block={r['L1.5b_block']['median']:.3f} "
              f"L2 unseen={L2:.3f} | adjacent={r['ref_adjacent_slices']['median']:.3f} "
              f"random={r['ref_random_pairs']['median']:.3f}", flush=True)

    json.dump(results, open(os.path.join(HERE, "bbv_distance.json"), "w"), indent=2)
    print(f"\nSaved {os.path.join(HERE, 'bbv_distance.json')}")


if __name__ == "__main__":
    main()
