#!/usr/bin/env python3
"""
Collect greedy optimisation trajectories (see traj_common.py for model, moves and greedy rule).
The SAME random starts (seeded) are used for every benchmark and for the joint run, so trajectories
are directly comparable across workloads.

  python run_traj.py --bench 602.gcc_s [--n-starts 1000]          # single benchmark
  python run_traj.py --joint --shard 0 --n-shards 2                # 17-SPEC geomean, per-workload MSCS logged

Output: results/<bench|joint_17>/part_XXX.npz (50 starts each; existing parts are skipped, never
overwritten) + meta.json. Arrays per state row: start, step, cfg (17 params, order in meta), area
(fast model), cpi, kind (0 stop/1 advance/2 reclaim), act (move slot or -1), stop reason, mscs (16)
[joint: cpi_wl (17), mscs_wl (17,16)], and for the 34 candidate moves of every state: k_valid,
k_coupled, k_cpi, k_area, k_mscs (16) [joint: k_cpi_wl (17), k_mscs = workload mean].
"""
import os, sys, json, time, argparse
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import traj_common as T

START_SEED, CHUNK = 2026, 50


def starts(n, mode="random", seed_offset=0):
    if mode == "random":
        rng = np.random.RandomState(START_SEED + seed_offset)
        return [T.S.sample17(rng) for _ in range(n)]
    # "small": every free parameter drawn uniformly from the lowest quarter of its value list (>= 1 value),
    # then the dse17 projection (validity rules, e.g. ROB >= 2*dispatch + IQ + LQ + SQ); own seed
    rng = np.random.RandomState(START_SEED + 1 + seed_offset)
    out = []
    for _ in range(n):
        c = T.S.sample17(rng)
        for p in T.SEARCH:
            vals = T.S.C.get_param_values(p) if hasattr(T.S.C, "get_param_values") else None
            if vals is None:
                from config_dse import get_param_values; vals = get_param_values(p)
            q = max(1, int(np.ceil(len(vals) * 0.25)))
            c[p] = vals[rng.randint(q)]
        out.append(T.S.project17(c))
    return out


def _wl_worker(wl, conn):
    P = T.WLPredictor(wl)
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg is None:
            break
        conn.send([P(c) for c in msg])


class JointEval:
    def __init__(self, wls):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for wl in wls:
            a, b = ctx.Pipe(); p = ctx.Process(target=_wl_worker, args=(wl, b), daemon=True)
            p.start(); self.conns.append(a); self.procs.append(p)
        for c in self.conns:
            assert c.recv() == "ready"

    def __call__(self, cfgs):
        for c in self.conns:
            c.send(cfgs)
        res = [c.recv() for c in self.conns]                      # W x n x (cpi, mscs)
        out = []
        for j in range(len(cfgs)):
            cw = np.array([res[w][j][0] for w in range(len(res))], np.float32)
            mw = np.stack([res[w][j][1] for w in range(len(res))])
            out.append({"cpi": float(np.exp(np.log(cw).mean())), "cpi_wl": cw, "mscs_wl": mw})
        return out

    def close(self):
        for c in self.conns:
            c.send(None)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bench", choices=T.CM.SPEC17)
    g.add_argument("--joint", action="store_true")
    ap.add_argument("--n-starts", type=int, default=1000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--start-mode", choices=["random", "small"], default="random")
    ap.add_argument("--seed-offset", type=int, default=0, help="new start seed = 2026 (+1 for small) + offset")
    args = ap.parse_args()
    from area17 import FastArea
    area = FastArea().many
    tag = "joint_17" if args.joint else args.bench
    d = os.path.join(args.out, tag); os.makedirs(d, exist_ok=True)
    meta = os.path.join(d, "meta.json")
    if not os.path.exists(meta):
        json.dump({"tag": tag, "params": T.SEARCH, "moves": T.MOVES, "step": T.STEP, "fixed": T.S.FIXED,
                   "G_MIN": T.G_MIN, "L_MAX": T.L_MAX, "MAX_STEPS": T.MAX_STEPS, "max_slices": T.MAX_SLICES,
                   "ckpt": T.CM.DEFAULT_CKPT, "start_seed": START_SEED + (args.start_mode == "small") + args.seed_offset, "start_mode": args.start_mode, "n_starts": args.n_starts,
                   "workloads": T.CM.SPEC17 if args.joint else [args.bench],
                   "area": "dse17 fast area model (decisions and stored areas)"},
                  open(meta, "w"), indent=1)
    if args.joint:
        ev = JointEval(T.CM.SPEC17)
        evaluate = ev
    else:
        P = T.WLPredictor(args.bench)
        evaluate = lambda cfgs: [{"cpi": a, "mscs": b} for a, b in (P(c) for c in cfgs)]
    st = starts(args.n_starts, args.start_mode, args.seed_offset)
    parts = [i for i in range(0, args.n_starts, CHUNK) if (i // CHUNK) % args.n_shards == args.shard]
    t0 = time.time()
    for i in parts:
        path = os.path.join(d, f"part_{i // CHUNK:03d}.npz")
        if os.path.exists(path):
            continue
        ids = list(range(i, min(i + CHUNK, args.n_starts)))
        walks = [T.walk(st[s], evaluate, area) for s in ids]
        arr = T.pack(walks, ids, joint_W=len(T.CM.SPEC17) if args.joint else None)
        np.savez_compressed(path + ".tmp.npz", **arr)
        os.replace(path + ".tmp.npz", path)
        n = len(arr["start"])
        print(f"[{tag}] part {i // CHUNK:03d}: {len(ids)} walks, {n} states "
              f"({n / len(ids):.1f}/walk) [{time.time() - t0:.0f}s]", flush=True)
    if args.joint:
        ev.close()
    print(f"[{tag}] DONE [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
