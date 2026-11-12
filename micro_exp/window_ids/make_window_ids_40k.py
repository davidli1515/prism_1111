"""
Map every row of the 40k dataset (micro_traindata_packed_all) to the 18k window id
of the same trace window, or -1 if that window is not part of the 18k dataset
(e.g. another SimPoint). Used to drop extra (unlabeled) rows that belong to held-out
18k windows.

Output: window_ids/w40k/{bm}.npy (int64, one entry per 40k row), w40k/summary.json

SPEC: 40k rows keep pack order, so windows are recovered by config-string order
      segmentation (as make_window_ids.py). A 40k window inherits the 18k id of the
      18k rows it contains (exact row match on the first 4681 columns).
Google: 40k files are the full shuffled files; windows are the signature groups of
      make_window_ids.py (same columns, fragments merged); a group inherits the
      18k id of its 18k rows.
A 40k window that contains 18k rows of more than one 18k window is an error.

SPEC extra rows live in separate 400-window blocks (one block per generation round /
SimPoint), so they never share a segment with 18k rows. A block without 18k rows is
compared window-by-window (block ordinal k vs 18k window k) on config-invariant trace
columns (micro_exp/bbv/trace_invariant_cols.npy):
  match rate >= 0.95 -> same trace windows under other configs: map k -> 18k window k
                         (only if the 18k window signatures are discriminative, i.e.
                         >= 200 distinct of 400; otherwise the ordinal mapping cannot be
                         verified and the block is marked ambiguous)
  match rate <= 0.05 -> another SimPoint: -1
  otherwise / window count not a multiple of 400 -> ambiguous: -2 (callers drop these)
Labels: >=0 18k window id, -1 window not in 18k (other SimPoint), -2 unknown.

Usage: python make_window_ids_40k.py
"""
import os, json
import numpy as np
from make_window_ids import (spec_window_ids, signature_groups, FEAT_END,
                             GOOGLE_WINDOW_SIZE)

INV = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "bbv", "trace_invariant_cols.npy"))

HERE = os.path.dirname(os.path.abspath(__file__))
P18 = "/data/micro_traindata_packed/gem"
P40 = "/data/micro_traindata_packed_all/gem"
OUT = os.path.join(HERE, "w40k")


def main():
    os.makedirs(OUT, exist_ok=True)
    gcols = json.load(open(os.path.join(HERE, "google_signature_cols.json")))
    summary = {}
    for f in sorted(os.listdir(P18)):
        if not f.endswith(".npy"):
            continue
        bm = f[:-4]
        A = np.asarray(np.load(os.path.join(P18, f), mmap_mode="r"), dtype=np.float32)[:, :FEAT_END]
        X = np.asarray(np.load(os.path.join(P40, f), mmap_mode="r"), dtype=np.float32)
        wid18 = np.load(os.path.join(HERE, f"{bm}.npy"))
        if bm in gcols:
            seg = signature_groups(X[:, :FEAT_END], gcols[bm])
            sizes = np.bincount(seg)
            frag = sizes[seg] % GOOGLE_WINDOW_SIZE.get(bm, 20) != 0
            seg[frag] = seg.max() + 1
        else:
            seg = spec_window_ids(X)
        row18 = {r.tobytes(): i for i, r in enumerate(A)}
        hit = np.array([row18.get(r[:FEAT_END].tobytes(), -1) for r in X])
        out = np.full(len(X), -1, dtype=np.int64)
        conflicts = 0
        for s in np.unique(seg):
            m = seg == s
            ids = np.unique(wid18[hit[m & (hit >= 0)]])
            if len(ids) == 1:
                out[m] = ids[0]
            elif len(ids) > 1:
                conflicts += 1
                out[m] = -2           # ambiguous: drop
        blocks = {}
        if bm not in gcols:
            nseg = seg.max() + 1
            no18 = np.array([not (hit[seg == s] >= 0).any() for s in range(nseg)])
            if nseg % 400 != 0:
                out[(hit < 0) & no18[seg]] = -2
                blocks["note"] = f"{nseg} windows not a multiple of 400: extras dropped (-2)"
            else:
                sig = lambda R: np.ascontiguousarray(R[:, INV]).tobytes()
                sig18 = {w: sig(A[wid18 == w][:1]) for w in range(400)}
                for b in range(nseg // 400):
                    ws = range(b * 400, (b + 1) * 400)
                    if not all(no18[w] for w in ws):
                        continue
                    rate = np.mean([sig(X[seg == w][:1]) == sig18[k] for k, w in enumerate(ws)])
                    lab = "same-windows" if rate >= 0.95 else ("other-simpoint" if rate <= 0.05 else "ambiguous")
                    n_distinct = len(set(sig18.values()))
                    if lab == "same-windows" and n_distinct < 200:
                        lab = "ambiguous"     # signatures too coarse to verify k -> k
                    blocks[f"block{b}"] = {"match_rate": round(float(rate), 3), "type": lab,
                                           "distinct_18k_signatures": n_distinct}
                    for k, w in enumerate(ws):
                        out[seg == w] = k if lab == "same-windows" else (-1 if lab == "other-simpoint" else -2)
        np.save(os.path.join(OUT, f"{bm}.npy"), out)
        n_in18 = int((hit >= 0).sum())
        summary[bm] = {"rows_40k": int(len(X)), "rows_also_in_18k": n_in18,
                       "extra_rows": int(len(X) - n_in18),
                       "extra_rows_mapped_to_18k_window": int(((hit < 0) & (out >= 0)).sum()),
                       "extra_rows_other_windows": int(((hit < 0) & (out == -1)).sum()),
                       "extra_rows_dropped_unknown": int(((hit < 0) & (out == -2)).sum()),
                       "spec_extra_blocks": blocks,
                       "windows_with_multiple_18k_ids": conflicts}
        print(bm, summary[bm], flush=True)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
