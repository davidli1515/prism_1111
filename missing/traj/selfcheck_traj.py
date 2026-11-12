#!/usr/bin/env python3
"""Checks before collection (writes nothing): (1) WLPredictor == refine_v3.SurrogateWL (the gem5-validated
surrogate) on random 17-param configs; (2) move table; (3) one timed walk per benchmark given (needs the
dse17 fast area model)."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import traj_common as T


def main():
    wls = sys.argv[1:] or ["602.gcc_s", "605.mcf_s"]
    print("step sizes:", T.STEP)
    from refine_v3 import SurrogateWL
    rng = np.random.RandomState(0)
    cfgs = [T.S.sample17(rng) for _ in range(4)]
    ok = True
    for wl in wls:
        ref = SurrogateWL(T.CM.DEFAULT_CKPT, [wl], "18k", device="cpu", max_slices=T.MAX_SLICES)
        P = T.WLPredictor(wl)
        for c in cfgs:
            r = ref.query(c); a, m = P(c)
            dc = abs(a - r["cpi"]) / r["cpi"]; dm = float(np.abs(m - r["mscs_wl"][0]).max())
            ok &= dc < 1e-5 and dm < 1e-5
            print(f"  {wl} cpi {r['cpi']:.5f} vs {a:.5f} (rel {dc:.1e})  mscs max diff {dm:.1e}")
    print("PREDICTOR", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)
    if os.path.exists(os.path.join(T.DSE_V2, "dse17", "area_model.pkl")):
        from area17 import FastArea
        A = FastArea().many
        P = T.WLPredictor(wls[0])
        ev = lambda cs: [{"cpi": a, "mscs": b} for a, b in (P(c) for c in cs)]
        t0 = time.time(); recs = T.walk(cfgs[0], ev, A)
        print(f"timed walk ({wls[0]}): {len(recs)} states, {time.time() - t0:.1f}s, stop={recs[-1]['stop']}, "
              f"kinds={[r['kind'] for r in recs]}")
        print(f"  cpi {recs[0]['ev']['cpi']:.3f} -> {recs[-1]['ev']['cpi']:.3f}, area {recs[0]['area']:.1f} -> {recs[-1]['area']:.1f}")
    print("SELFCHECK_TRAJ PASS")


if __name__ == "__main__":
    main()
