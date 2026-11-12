#!/usr/bin/env python3
"""Random search with an 18k budget per benchmark: the first 18,000 designs (idx 0..17999, i.e. a prefix of the
random stream) of results_rw/<bench>/, written to results_rw18k/<bench>/part_000.npz, part_001.npz (same keys) plus a
copy of meta.json. 17 x 18k = 306k designs. Read-only on results_rw; refuses to overwrite."""
import os, sys, glob, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SRC, DST, N = os.path.join(HERE, "results_rw"), os.path.join(HERE, "results_rw18k"), 18000


def main():
    if os.path.exists(DST):
        sys.exit(f"exists, refusing to overwrite: {DST}")
    tot = 0
    for b in sorted(d for d in os.listdir(SRC) if d[0].isdigit()):
        os.makedirs(os.path.join(DST, b))
        Z = [np.load(f) for f in sorted(glob.glob(os.path.join(SRC, b, "part_*.npz")))]
        cat = {k: np.concatenate([z[k] for z in Z]) for k in Z[0].files}
        o = np.argsort(cat["idx"], kind="stable")[:N]
        assert (cat["idx"][o] == np.arange(N)).all()
        for p, (lo, hi) in enumerate(((0, 10000), (10000, N))):
            np.savez_compressed(os.path.join(DST, b, f"part_{p:03d}.npz"), **{k: v[o][lo:hi] for k, v in cat.items()})
        if os.path.exists(os.path.join(SRC, b, "meta.json")):
            shutil.copy(os.path.join(SRC, b, "meta.json"), os.path.join(DST, b, "meta.json"))
        tot += N
    open(os.path.join(DST, "README.txt"), "w").write(f"first {N} designs (idx order) of results_rw per benchmark; {tot} total\n")
    print("designs", tot)


if __name__ == "__main__":
    main()
