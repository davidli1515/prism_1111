#!/usr/bin/env python3
"""
Expert intervention on a greedy walk (model only, no gem5). Same greedy as traj_common.walk (same moves, reclaim /
advance rules, G_MIN, L_MAX, MAX_STEPS), plus an expert decision taken at step 0 and kept for the rest of the walk:
  --set p=v   change applied once before the walk continues (the decision)
  --pin p>=v  lower bound the greedy may not undo (the decision is locked, as a code change would be)
The original walk is replayed first (must match results_small exactly), then the expert walk starts from the same
design. Writes expert/<bench>_s<start>_<tag>.json; never overwrites.
  python expert_walk.py --bench 631.deepsjeng_s --start 761 --tag lockI_L1up --set l1_size=64 --pin l1i_assoc>=4 --pin l1_size>=64
"""
import os, sys, glob, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import traj_common as T
from traj_common import MOVES, G_MIN, L_MAX, MAX_STEPS, ckey, move
from plot_stage_jain_optimizers import jain4, cat
from area17 import FastArea


def walk(c0, evaluate, area_many, ok=lambda c: True):
    """traj_common.walk with an extra admissibility test on every candidate design"""
    c, visited, recs = c0, {ckey(c0)}, []
    for step in range(MAX_STEPS + 1):
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
        if step == MAX_STEPS:
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


def show(name, recs):
    cpi = np.array([r["cpi"] for r in recs]); m = np.array([r["mscs"] for r in recs]); J = jain4(cpi, m)
    print(f"--- {name}: {len(recs)} states")
    for r, j in zip(recs, J):
        x = np.array(r["mscs"]) * r["cpi"]
        act = "" if r["act"] is None else f"{'reclaim' if r['kind'] == 2 else 'advance'} {r['act'][0]}{'+' if r['act'][1] > 0 else '-'}"
        print(f"  s{r['step']:2d} CPI {r['cpi']:.3f} area {r['area']:5.2f} | FP Dec {x[1]:.2f} Iss {x[9]:.2f} (Iss/Dec {x[9]/max(x[1],1e-9):.2f}) "
              f"| Jain Dec {j[0]:.2f} Disp {j[1]:.2f} Iss {j[2]:.2f} Com {j[3]:.2f} | l1 {r['cfg']['l1_size']} l1i {r['cfg']['l1i_assoc']} | {act}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", action="append", default=[]); ap.add_argument("--pin", action="append", default=[])
    a = ap.parse_args()
    out = os.path.join(HERE, "expert", f"{a.bench}_s{a.start}_{a.tag}.json")
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
    pins = [(p.split(">=")[0], int(p.split(">=")[1])) for p in a.pin]
    ok = lambda c: all(c[p] >= v for p, v in pins)
    c1 = T.S.project17(dict(c0, **{s.split("=")[0]: int(s.split("=")[1]) for s in a.set}))
    exp = walk(c1, evaluate, area, ok)
    show("original", orig); show(f"expert ({a.set} {a.pin})", exp)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"bench": a.bench, "start": a.start, "set": a.set, "pin": a.pin, "replay_ok": same,
               "c0": {k: int(v) for k, v in c0.items()}, "original": orig, "expert": exp}, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
