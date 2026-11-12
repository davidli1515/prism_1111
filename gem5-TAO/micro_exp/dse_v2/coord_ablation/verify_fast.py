#!/usr/bin/env python3
"""
Bit-exactness + speed check of fast_infer.FastEngine against the original dse_infer (read-only,
writes nothing).  For each workload: random configs and, from each, every single-parameter +/-1
neighbour (the access pattern of a refine step, exercising memo hits); every slice compared with
np.array_equal.  Usage: python verify_fast.py [--workloads ...] [--n 6]
"""
import os, sys, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import coord as C
from fast_infer import FastEngine
from dse_infer import dse_infer
from config_dse import sample_random_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="+", default=["602.gcc_s", "605.mcf_s", "623.xalancbmk_s"])
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    import run_gem_dse as R
    ok = True
    for wl in args.workloads:
        base = C.single_workload_engine(R.PRECOMPUTE_BASE, wl)
        fast = FastEngine(base)
        rng = np.random.RandomState(1)
        cfgs = []
        for _ in range(args.n):
            c = C._project_to_valid(sample_random_config(rng)); cfgs.append(c)
            for name in C.GRADIENT_PARAMS:
                for fn in (C._increase_by_stride, C._decrease_by_stride):
                    v, _ = fn(c, name)
                    if v is not None:
                        cn = dict(c); cn[name] = v; cfgs.append(C._project_to_valid(cn))
        t_o = t_f = 0.0; n_bad = n_sl = 0
        for c in cfgs:
            t = time.time(); a = [dse_infer(pre, c) for _, pre in base.data[c["l1_size"]][wl]]; t_o += time.time() - t
            t = time.time(); b = fast.get_features(c, wl); t_f += time.time() - t
            n_sl += len(a); n_bad += sum(not np.array_equal(x, y) for x, y in zip(a, b))
        ok &= n_bad == 0
        print(f"{wl:16s} configs {len(cfgs):4d} slices compared {n_sl:6d} mismatches {n_bad}  "
              f"original {t_o:.1f}s  fast {t_f:.1f}s  speed-up {t_o / t_f:.1f}x", flush=True)
    print("VERIFY_FAST", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
