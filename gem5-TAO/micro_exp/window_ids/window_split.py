"""
Trace-window-grouped train/test split (no window appears on both sides).

    from window_split import load_window_ids, split_by_window
    wid = load_window_ids("605.mcf_s")                 # int64 per row, 18k row order
    tr_idx, te_idx = split_by_window(wid, 15/16, np.random.RandomState(seed))

Window ids come from make_window_ids.py (see its docstring for how they were recovered).
"""
import os
import numpy as np

WINDOW_DIR = os.path.dirname(os.path.abspath(__file__))


def load_window_ids(name, expected_rows=None):
    wid = np.load(os.path.join(WINDOW_DIR, f"{name}.npy"))
    if expected_rows is not None and len(wid) != expected_rows:
        raise ValueError(f"window ids for {name}: {len(wid)} rows, data has {expected_rows}")
    return wid


def split_by_window(wid, train_ratio, rng):
    """
    Randomly assign whole windows to the held-out side until it holds
    ~(1 - train_ratio) of the rows. Windows larger than half the held-out
    budget (merged groups, e.g. brotli/re2) always stay on the train side.
    Returns (train_idx, test_idx), each in random order.
    """
    n = len(wid)
    budget = n - int(n * train_ratio)
    sizes = np.bincount(wid)
    held = np.zeros(len(sizes), bool)
    cur = 0
    for w in rng.permutation(len(sizes)):
        if cur >= budget:
            break
        if sizes[w] == 0 or sizes[w] > 0.5 * budget:
            continue
        held[w] = True
        cur += sizes[w]
    is_test = held[wid]
    tr_idx = rng.permutation(np.where(~is_test)[0])
    te_idx = rng.permutation(np.where(is_test)[0])
    return tr_idx, te_idx
