#!/usr/bin/env python3
"""
Expert intervention at step k of a greedy walk (model only, no gem5). Variant of expert_walk.py (which intervenes at
step 0); same greedy rules (traj_common moves, reclaim / advance, G_MIN, L_MAX, MAX_STEPS).
  1. replay the original walk from results_small (must match exactly)
  2. take the original state at step --at, apply --set (the decision), keep --pin bounds (p>=v or p<=v) for the rest
  3. greedy continues; states already visited by the original walk up to step --at stay visited
Expert record i is the state i steps after the decision (record 0 = right after it, drawn at search step at + 1).
Writes expert/<bench>_s<start>_at<k>_<tag>.json; never overwrites.
  python expert_walk_at.py --bench 607.cactuBSSN_s --start 840 --at 10 --tag port2 --set RdWrPort=2 --pin "RdWrPort>=2"
"""
import os, sys, glob, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import traj_common as T
from traj_common import MOVES, G_MIN, L_MAX, MAX_STEPS, ckey, move
from plot_stage_jain_optimizers import cat
from expert_walk import show
from area17 import FastArea


def walk(c0, evaluate, area_many, ok=lambda c: True, visited0=(), max_steps=MAX_STEPS):
    """traj_common.walk with an admissibility test and a pre-seeded visited set"""
    c, visited, recs = c0, {ckey(c0)} | set(visited0), []
    for step in range(max_steps + 1):
        cands = [move(c, p, d) for p, d in MOVES]
        cands = [(cn, cp) if cn is not None and ok(cn) else (None, None) for cn, cp in cands]
        cfgs = [c] + [cn for cn, _ in cands if cn is not None]
        ev = evaluate(cfgs); ar = area_many(cfgs)
        s0, a0 = ev[0], float(ar[0])
        slot, j = [], 1
        for cn, cp in cands:
            if cn is None:
                slot.append(None)
            else:
                slot.append((cn, cp, ev[j], float(ar[j]))); j += 1
        rec = {"step": step, "cfg": {p: int(c[p]) for p in T.SEARCH}, "area": a0, "cpi": s0["cpi"],
               "mscs": np.asarray(s0["mscs"]).tolist(), "act": None, "kind": 0}
        if step == max_steps:
            recs.append(rec); break
        cpi0 = s0["cpi"]; best, kind = None, 0
        rc = [(i, a0 - x[3]) for i, x in enumerate(slot) if x is not None and MOVES[i][1] < 0
              and (x[2]["cpi"] - cpi0) / cpi0 <= L_MAX and x[3] < a0 and ckey(x[0]) not in visited]
        if rc:
            best, kind = max(rc, key=lambda t: t[1])[0], 2
        else:
            up = [(i, ((cpi0 - x[2]["cpi"]) / cpi0) / max((x[3] - a0) / a0, 1e-3))
                  for i, x in enumerate(slot) if x is not None and MOVES[i][1] > 0
                  and (cpi0 - x[2]["cpi"]) / cpi0 >= G_MIN and ckey(x[0]) not in visited]
            if up:
                best, kind = max(up, key=lambda t: t[1])[0], 1
        if best is None:
            recs.append(rec); break
        rec["act"], rec["kind"] = list(MOVES[best]), kind
        recs.append(rec)
        c = slot[best][0]; visited.add(ckey(c))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--at", type=int, required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--set", action="append", default=[]); ap.add_argument("--pin", action="append", default=[])
    a = ap.parse_args()
    out = os.path.join(HERE, "expert", f"{a.bench}_s{a.start}_at{a.at}_{a.tag}.json")
    if os.path.exists(out):
        sys.exit(f"exists, refusing to overwrite: {out}")
    g = cat(sorted(glob.glob(os.path.join(HERE, "results_small", a.bench, "part_*.npz"))), ("start", "step", "cfg", "cpi"))
    m = g["start"] == a.start
    c0 = T.S.project17(dict(zip(T.SEARCH, map(int, g["cfg"][m][0]))))
    P = T.WLPredictor(a.bench); area = FastArea().many
    evaluate = lambda cfgs: [{"cpi": x, "mscs": y} for x, y in (P(c) for c in cfgs)]
    orig = walk(c0, evaluate, area)
    same = len(orig) == m.sum() and all(list(r["cfg"].values()) == list(map(int, c)) for r, c in zip(orig, g["cfg"][m]))
    print(f"replay of the original walk matches results_small: {same}")
    if a.at >= len(orig):
        sys.exit(f"--at {a.at} beyond the original walk ({len(orig)} states)")
    pins = [(p.split(op)[0], op, int(p.split(op)[1])) for p in a.pin for op in (">=", "<=") if op in p]
    ok = lambda c: all(c[p] >= v if op == ">=" else c[p] <= v for p, op, v in pins)
    ck = dict(orig[a.at]["cfg"])
    c1 = T.S.project17(dict(ck, **{s.split("=")[0]: int(s.split("=")[1]) for s in a.set}))
    seen = [ckey(T.S.project17(dict(r["cfg"]))) for r in orig[:a.at + 1]]
    exp = walk(c1, evaluate, area, ok, visited0=seen, max_steps=MAX_STEPS - a.at - 1)
    show("original", orig); show(f"expert at step {a.at} ({a.set} {a.pin})", exp)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"bench": a.bench, "start": a.start, "at": a.at, "tag": a.tag, "set": a.set, "pin": a.pin, "replay_ok": same,
               "c0": {k: int(v) for k, v in c0.items()}, "original": orig, "expert": exp}, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
