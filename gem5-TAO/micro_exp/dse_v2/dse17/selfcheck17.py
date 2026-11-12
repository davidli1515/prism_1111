#!/usr/bin/env python3
"""
Checks before any dse17 run (writes nothing except temporary McPAT dirs that are removed):
  1. space: 2000 samples -> fixed params held, valid, L1 assoc within the L1-size cap; 17 free params vary
  2. step: steps17.coord_step17(band=None, strides=(1,)) == coord_ablation/coord.coord_step for every
     arm (and therefore == dse_engine._coordinated_step for mscs_protect, checked in coord_ablation)
  3. area: race-free McPAT, 24 configs serial == parallel (processes) == parallel again
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import space17 as S
C = S.C
import steps17
from area17 import area_isolated
from concurrent.futures import ProcessPoolExecutor


def main():
    ok = True
    rng = np.random.RandomState(0)
    cs = [S.sample17(rng) for _ in range(2000)]
    fixed_ok = all(all(c[k] == v for k, v in S.FIXED.items()) for c in cs)
    valid_ok = all(S.is_valid_config(c) for c in cs)
    assoc_ok = all(max(c["l1d_assoc"], c["l1i_assoc"]) <= max(1, c["l1_size"] * 1024 // (16 * 64)) for c in cs)
    varied = sum(len({c[p] for c in cs}) > 1 for p in S.SEARCH)
    print(f"1. space: fixed {fixed_ok} valid {valid_ok} l1-assoc cap {assoc_ok} varied params {varied}/17")
    ok &= fixed_ok and valid_ok and assoc_ok and varied == 17

    from fast_infer import FastEngine
    model, norm, R = C.load_original_model()
    wl = "605.mcf_s"
    P = C.CachedPredictor(C.GEMPredictor(model, norm, FastEngine(C.single_workload_engine(R.PRECOMPUTE_BASE, wl)), workload=wl))
    same_all = True
    for c in cs[:6]:
        for arm, (inc, prot) in C.ARMS.items():
            a, ca, _ = C.coord_step(c, P, inc, prot)
            b, cb, _ = steps17.coord_step17(c, P, inc, prot, None, (1,))
            same = C.ckey(a) == C.ckey(b) and {k: v["action"] for k, v in ca.items()} == {k: v["action"] for k, v in cb.items()}
            same_all &= same
    print(f"2. step mirror (6 configs x 4 arms): {same_all}")
    ok &= same_all

    probe = cs[:24]
    serial = [area_isolated(c) for c in probe]
    with ProcessPoolExecutor(12) as ex:
        par = list(ex.map(area_isolated, probe))
    with ProcessPoolExecutor(12) as ex:
        par2 = list(ex.map(area_isolated, probe[::-1]))[::-1]
    eq = all((a is None and b is None and d is None) or (a is not None and a == b == d) for a, b, d in zip(serial, par, par2))
    print(f"3. area serial == parallel == repeat: {eq}  (failures {sum(a is None for a in serial)}/24)")
    ok &= eq
    print("SELFCHECK17", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
