#!/usr/bin/env python3
"""After collection: race-free McPAT on a random sample of trajectory states (and their accepted moves)
to report the error of the fast area model actually stored in the trajectories.
Writes results/area_check.json (refuses to overwrite)."""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import traj_common as T
from area17 import AreaStore


def main():
    out = os.path.join(HERE, "results", "area_check.json")
    if os.path.exists(out):
        sys.exit(f"exists: {out}")
    rng = np.random.RandomState(7)
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "results", "*", "part_*.npz"))):
        z = np.load(f)
        for i in rng.choice(len(z["area"]), min(4, len(z["area"])), replace=False):
            c = T.S.project17(dict(zip(T.SEARCH, map(int, z["cfg"][i]))))
            rows.append((os.path.basename(os.path.dirname(f)), c, float(z["area"][i])))
    rows = [rows[i] for i in rng.choice(len(rows), min(600, len(rows)), replace=False)]
    A = AreaStore(24); A.prefetch([c for _, c, _ in rows]); A.flush()
    rel = np.array([abs(fa - A(c)) / A(c) for _, c, fa in rows if A(c)])
    rep = {"n": int(len(rel)), "mcpat_fail": int(sum(A(c) is None for _, c, _ in rows)),
           "abs_rel_err_median": float(np.median(rel)), "p90": float(np.percentile(rel, 90)),
           "max": float(rel.max())}
    json.dump(rep, open(out, "w"), indent=1); print(rep)


if __name__ == "__main__":
    main()
