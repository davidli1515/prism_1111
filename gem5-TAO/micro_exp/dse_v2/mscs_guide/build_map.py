#!/usr/bin/env python3
"""
Offline relief map: on n random configs, apply +1 to each of the 21 parameters and record
how every per-workload MSCS component changes (all 17 workloads in one query).
Saves maps/map_n{n}_s{seed}.npz with
  R (W,P,16)      signed relative relief  sum_s(m(x)-m(x+p)) / sum_s m(x)
  mean_gain (W,P) mean relative CPI gain of +1 p  (static, MSCS-free prior; baseline)
  cnt (P,)        number of states where the +1 move was valid
and prints the top parameters per pressure component (sanity check vs. physical intuition).

Usage: python mscs_guide/build_map.py --n-states 40 --seed 0
"""
import os, sys, time, argparse
from guide_common import HERE, PARAMS, PRESSURE, comp_name, _shift
import numpy as np
from common import SPEC17, DEFAULT_CKPT, sample_random_config
from refine_v3 import SurrogateWL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "maps", f"map_n{args.n_states}_s{args.seed}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        sys.exit(f"exists, refusing to overwrite: {out}")

    S = SurrogateWL(DEFAULT_CKPT, SPEC17, "18k", device=args.device, max_slices=50)
    rng = np.random.RandomState(args.seed)
    W, P = len(SPEC17), len(PARAMS)
    num, den = np.zeros((W, P, 16)), np.zeros((W, P, 16))
    gsum, cnt = np.zeros((W, P)), np.zeros(P)
    t0 = time.time()
    for s in range(args.n_states):
        c = sample_random_config(rng)
        q0 = S.query(c); m0, c0 = q0["mscs_wl"], np.array(q0["per_wl_cpi"])
        for p, n in enumerate(PARAMS):
            cn = _shift(c, n, +1)
            if cn is None:
                continue
            q1 = S.query(cn)
            num[:, p] += m0 - q1["mscs_wl"]; den[:, p] += m0
            gsum[:, p] += (c0 - np.array(q1["per_wl_cpi"])) / c0; cnt[p] += 1
        print(f"  state {s+1}/{args.n_states} q={S.n_queries} [{time.time()-t0:.0f}s]", flush=True)
    R = num / np.maximum(den, 1e-9)
    mean_gain = gsum / np.maximum(cnt, 1)
    np.savez(out, R=R, mean_gain=mean_gain, cnt=cnt, workloads=np.array(SPEC17),
             params=np.array(PARAMS), n_states=args.n_states, seed=args.seed)
    Rm = R.mean(0)
    print("\nTop-3 parameters by relief, per pressure component (mean over 17 workloads):")
    for i in PRESSURE:
        top = np.argsort(-Rm[:, i])[:3]
        print(f"  {comp_name(i):9s} " + "  ".join(f"{PARAMS[p]}({Rm[p, i]:+.3f})" for p in top))
    print(f"-> {out} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
