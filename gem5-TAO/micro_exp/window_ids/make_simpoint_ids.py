"""
Find simpoints (gem5 checkpoints, ckpt_*) of the 40k dataset that are NOT in the
18k dataset, giving an "unseen simpoint" test set for models trained on 18k.

Output: window_ids/unseen_simpoint/{benchmark}.npy — row indices into
        /scratch2/wl730/micro_traindata_packed_all/gem/{benchmark}.npy
        window_ids/unseen_simpoint/summary.json

How simpoints are recovered (SPEC only)
---------------------------------------
pack_npy.py concatenates rows in sorted(ckpt)/sorted(slice)/sorted(datapoint)
order, so in a packed file each ckpt is one contiguous block. Windows (slices)
are segmented by config-string order (see make_window_ids.py); every ckpt in the
raw tree has 400 slices, so ckpt k = windows [400k, 400k+400).

A benchmark is used only if the recovery is self-consistent:
  * window count is a multiple of 400, and
  * every 400-window block is either fully contained in 18k or shares no row
    with 18k (exact byte match on the 4681 feature columns).
Blocks with no 18k rows are the unseen simpoints.
Excluded: perlbench (1 ckpt), gcc/lbm (both ckpts already in 18k),
roms (682 windows — boundaries ambiguous), Google bms (shuffled, no ckpt info).

Usage:
    python make_simpoint_ids.py
"""
import os, json
import numpy as np

PACKED_18K = "/scratch2/wl730/micro_traindata_packed/gem"
PACKED_40K = "/scratch2/wl730/micro_traindata_packed_all/gem"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unseen_simpoint")
FEAT_END = 4681
SLICES_PER_CKPT = 400
ARCH_START = 3850
CFG_TAGS = ["fw", "dw", "rw", "dpw", "iw", "wbw", "cw", "rob", "iq", "lq", "sq", "fb",
            "ialu", "imd", "falu", "fmd", "rdwr", "l1", "l2", "da", "ia", "la", "bp"]


def cfg_string(row):
    vals = row[ARCH_START:ARCH_START + len(CFG_TAGS)]
    return "_".join(f"{t}{int(round(v))}" for t, v in zip(CFG_TAGS, vals))


def window_ids_in_order(P):
    s = [cfg_string(r) for r in P]
    new = np.array([0] + [1 if s[i] <= s[i - 1] else 0 for i in range(1, len(s))])
    return np.cumsum(new)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {}
    for f in sorted(os.listdir(PACKED_40K)):
        if not (f.endswith(".npy") and f[0].isdigit()):     # SPEC only
            continue
        name = f[:-4]
        P40 = np.asarray(np.load(os.path.join(PACKED_40K, f), mmap_mode="r"), dtype=np.float32)
        P18 = np.asarray(np.load(os.path.join(PACKED_18K, f), mmap_mode="r"), dtype=np.float32)
        wid = window_ids_in_order(P40)
        n_win = int(wid[-1] + 1)
        in18 = {r[:FEAT_END].tobytes() for r in P18}
        is18 = np.fromiter((r[:FEAT_END].tobytes() in in18 for r in P40), bool, len(P40))
        ckpt = wid // SLICES_PER_CKPT

        info = {"rows_40k": int(len(P40)), "windows_40k": n_win, "blocks": {}}
        ok = n_win % SLICES_PER_CKPT == 0 and is18.sum() == len(P18)
        unseen = []
        for k in np.unique(ckpt):
            m = ckpt == k
            n18 = int(is18[m].sum())
            info["blocks"][int(k)] = {"rows": int(m.sum()), "rows_in_18k": n18}
            if n18 == 0:
                unseen.append(np.where(m)[0])
            elif n18 != m.sum():
                ok = False                                  # block partly in 18k
        if ok and unseen:
            idx = np.concatenate(unseen).astype(np.int64)
            np.save(os.path.join(OUT_DIR, f"{name}.npy"), idx)
            info["status"] = "used"
            info["unseen_rows"] = int(len(idx))
        else:
            info["status"] = "excluded: " + ("no unseen simpoint" if ok else "inconsistent blocks")
        summary[name] = info
        print(f"{name:20s} windows={n_win:5d} blocks={info['blocks']}  -> {info['status']}"
              + (f" ({info['unseen_rows']} rows)" if info["status"] == "used" else ""))
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
