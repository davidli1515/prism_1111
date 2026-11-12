"""
Recover the trace window (ckpt/slice) of every row in the 18k dataset, so that
train/test can be split by window instead of by row.

Output: window_ids/{benchmark}.npy — int64 window id per row, same row order as
        /data/micro_traindata_packed/gem/{benchmark}.npy
        window_ids/summary.json

How windows are recovered
-------------------------
Raw layout: micro_traindata/<bm>/ckpt_*/slice_*/<config-string>_<hash>/gem_data.npy
One slice = one trace window simulated under 20/30/60 random configs.

* SPEC (17 bms): pack_npy.py concatenated rows in sorted(ckpt)/sorted(slice)/
  sorted(datapoint) order and the 18k files kept that order. Datapoint folder
  names are the config string, so within a window the config string is strictly
  increasing; a non-increase marks a new window. This rule can only MERGE
  adjacent windows, never split one. Validated against the raw tree for
  perlbench/gcc/lbm (400/800/800 windows, 0 split, 0 merged); every SPEC bm
  yields 400 x 20 (perlbench 400 x 60, gcc/lbm 800 x 30, xz 3 short windows).

* Google (6 bms): the packed files are shuffled, so windows are recovered by a
  signature: the values of feature columns that are constant within a window
  (config-independent trace features). Column sets per bm are in
  google_signature_cols.json, fitted on the full 16k google files so that groups
  are whole windows (20 rows; re2: 40) — see fit notes in summary.json.
  Groups whose size is not a multiple of the window size (rare fragments of a
  window) are merged into one group, so a window is never split. Some groups
  contain several identical-looking windows (merge = safe, just coarser).
  18k google rows are mapped to the full file by exact row match.

Usage:
    python make_window_ids.py
"""
import os, json
import numpy as np
from collections import Counter

PACKED_18K = "/data/micro_traindata_packed/gem"
GOOGLE_FULL = "/data/micro_traindata_packed_google"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_END = 4681
ARCH_START = 3850
CFG_TAGS = ["fw", "dw", "rw", "dpw", "iw", "wbw", "cw", "rob", "iq", "lq", "sq", "fb",
            "ialu", "imd", "falu", "fmd", "rdwr", "l1", "l2", "da", "ia", "la", "bp"]
GOOGLE_WINDOW_SIZE = {"re2": 40}   # default 20


def cfg_string(row):
    """Reproduce the datapoint folder-name prefix from arch params gd[3850:3873]."""
    vals = row[ARCH_START:ARCH_START + len(CFG_TAGS)]
    return "_".join(f"{t}{int(round(v))}" for t, v in zip(CFG_TAGS, vals))


def spec_window_ids(P):
    s = [cfg_string(r) for r in P]
    new = np.array([0] + [1 if s[i] <= s[i - 1] else 0 for i in range(1, len(s))])
    return np.cumsum(new).astype(np.int64)


def signature_groups(X, cols):
    K = np.ascontiguousarray(X[:, cols])
    _, gid = np.unique(K.view(np.dtype((np.void, K.dtype.itemsize * K.shape[1]))).ravel(),
                       return_inverse=True)
    return gid.astype(np.int64)


def google_window_ids(name, P, cols, wsize):
    G = np.load(os.path.join(GOOGLE_FULL, f"{name}.npy")).astype(np.float32)[:, :FEAT_END]
    gid = signature_groups(G, cols)
    sizes = np.bincount(gid)
    frag = sizes[gid] % wsize != 0
    gid[frag] = gid.max() + 1                       # merge fragments -> one group
    row_of = {r.tobytes(): i for i, r in enumerate(G)}
    idx = np.array([row_of[r[:FEAT_END].tobytes()] for r in P])   # KeyError = no match
    _, wid = np.unique(gid[idx], return_inverse=True)
    return wid.astype(np.int64), int(frag.sum())


def main():
    google_cols = json.load(open(os.path.join(OUT_DIR, "google_signature_cols.json")))
    summary = {}
    for f in sorted(os.listdir(PACKED_18K)):
        if not f.endswith(".npy"):
            continue
        name = f[:-4]
        P = np.asarray(np.load(os.path.join(PACKED_18K, f), mmap_mode="r"), dtype=np.float32)
        if name in google_cols:
            wsize = GOOGLE_WINDOW_SIZE.get(name, 20)
            wid, n_frag = google_window_ids(name, P, google_cols[name], wsize)
            method = f"signature({len(google_cols[name])} cols), fragments merged: {n_frag} rows (of full 16k)"
        else:
            wid = spec_window_ids(P)
            method = "config-string order segmentation"
        sizes = np.bincount(wid)
        np.save(os.path.join(OUT_DIR, f"{name}.npy"), wid)
        summary[name] = {
            "rows": int(len(P)), "windows": int(len(sizes)),
            "max_window_rows": int(sizes.max()),
            "largest_window_frac": round(float(sizes.max() / len(P)), 4),
            "size_hist_top": {int(k): int(v) for k, v in Counter(sizes).most_common(5)},
            "method": method,
        }
        print(f"{name:22s} rows={len(P):6d} windows={len(sizes):5d} "
              f"max={sizes.max():5d} ({sizes.max()/len(P):.1%})  {method}")
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
