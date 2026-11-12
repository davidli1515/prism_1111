#!/usr/bin/env python3
"""
Driver for the coordinated-swap 2x2 ablation (see coord.py).  Joint 17 SPEC benchmarks, exactly the
paper's BO + refine schedule (run_gem_dse.py): BO (budget 500) -> refine rounds (round 0: every
Pareto point, later rounds: only new Pareto points); each point is refined 17 times, each time
NAVIGATED by one benchmark's predictor, and each refined config is scored by the 17-benchmark
geomean CPI.  Only refined end points enter the archive (as in the original).

Fairness additions: one shared BO run for all arms; every new (config, workload) evaluation is
counted (17 = 1 joint query) and stored per archive point, so arms can be compared at equal cost.

17 worker processes, one per benchmark (each loads only its own precompute, ~3 GB).
Writes only under coord_ablation/ (results/, area_cache/, mcpat_work/); refuses to overwrite.

Usage:
  python run_coord.py --selfcheck                 # original vs mirrored step + predictor (writes nothing)
  python run_coord.py --arm bo                    # shared BO base -> results/bo_base_s42.json
  python run_coord.py --arm mscs_protect          # one arm -> results/mscs_protect_s42.json
"""
import os, sys, json, time, glob, hashlib, argparse
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coord as C

OLD_AREA_CACHE = os.path.join(os.path.dirname(HERE), "area_cache")     # dse_v2 cache, read only
NEW_AREA_CACHE = os.path.join(HERE, "area_cache")
MCPAT_WORK = os.path.join(HERE, "mcpat_work")


# ---------------------------------------------------------------- area (read old cache, write new)
def _k(c):                       # same key as dse_v2/common.Area, so the old cache is reusable
    return hashlib.md5(json.dumps(tuple((k, c[k]) for k in sorted(c))).encode()).hexdigest()


class AreaRO:
    def __init__(self):
        self.cache = {}
        for d in (OLD_AREA_CACHE, NEW_AREA_CACHE):
            for f in glob.glob(os.path.join(d, "*.json")):
                try:
                    self.cache.update(json.load(open(f)))
                except Exception:
                    pass
        os.makedirs(NEW_AREA_CACHE, exist_ok=True); os.makedirs(MCPAT_WORK, exist_ok=True)
        self.path = os.path.join(NEW_AREA_CACHE, f"area_{os.getpid()}.json")
        self.new = {}

    def _compute(self, c):
        from area_model import compute_area
        try:
            return compute_area(c, work_dir=MCPAT_WORK)
        except Exception:
            return None

    def prefetch(self, cs, workers=8):
        todo = [c for c in cs if _k(c) not in self.cache]
        with ThreadPoolExecutor(workers) as ex:
            for c, a in zip(todo, ex.map(self._compute, todo)):
                self.cache[_k(c)] = a; self.new[_k(c)] = a

    def __call__(self, c):
        k = _k(c)
        if k not in self.cache:
            self.cache[k] = self.new[k] = self._compute(c)
        return self.cache[k]

    def flush(self):
        json.dump(self.new, open(self.path, "w"))


# ---------------------------------------------------------------- one worker per benchmark
def worker_loop(wl, conn, max_slices, fast=True):
    import torch
    torch.set_num_threads(1)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        model, norm, R = C.load_original_model()
        eng = C.single_workload_engine(R.PRECOMPUTE_BASE, wl, max_slices)
        if fast:                                  # bit-exact (verify_fast.py), ~5x faster features
            from fast_infer import FastEngine
            eng = FastEngine(eng)
        gp = C.GEMPredictor(model, norm, eng, workload=wl, device="cpu")
    P = C.CachedPredictor(gp)
    conn.send(("ready", wl))
    while True:
        msg = conn.recv()
        n0 = P.n_eval
        if msg[0] == "stop":
            break
        if msg[0] == "cpi":
            out = [P.predict_cpi(c) for c in msg[1]]
        elif msg[0] == "refine":
            _, cfg, arm, steps = msg
            out = C.refine(cfg, P, C.make_step(arm), steps)
        conn.send((out, P.n_eval - n0))


class Workers:
    def __init__(self, workloads, max_slices, fast=True):
        ctx = mp.get_context("spawn")
        self.wls, self.conns, self.procs = list(workloads), [], []
        for wl in self.wls:
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker_loop, args=(wl, b, max_slices, fast), daemon=True)
            p.start(); self.conns.append(a); self.procs.append(p)
        for c in self.conns:
            c.recv()
        self.n_eval = 0                                   # (config, workload) evaluations

    def _all(self, msgs):
        for c, m in zip(self.conns, msgs):
            c.send(m)
        outs = []
        for c in self.conns:
            o, n = c.recv(); self.n_eval += n; outs.append(o)
        return outs

    def gmean_cpi(self, configs):
        per = np.array(self._all([("cpi", configs)] * len(self.wls)))          # (W, n)
        return np.exp(np.log(per).mean(0)).tolist()

    def refine_all(self, cfg, arm, steps):
        return self._all([("refine", cfg, arm, steps)] * len(self.wls))

    @property
    def cost(self):                                       # joint-query equivalents
        return self.n_eval / len(self.wls)

    def close(self):
        for c in self.conns:
            c.send(("stop",))
        for p in self.procs:
            p.join(timeout=30)


class JointRemote:
    """predict_cpi for dse_engine.run_bo_gem (geomean over the 17 workers)."""
    def __init__(self, W):
        self.W, self.cost_at = W, {}

    def predict_cpi(self, c):
        v = self.W.gmean_cpi([c])[0]
        self.cost_at.setdefault(C.ckey(c), self.W.cost)
        return v


# ---------------------------------------------------------------- self-check (writes nothing)
def selfcheck(n=6, wl="602.gcc_s"):
    """original engine + GEMPredictor + _coordinated_step  vs  FastEngine + CachedPredictor + coord_step"""
    from config_dse import sample_random_config, DEFAULT_CONFIG
    from fast_infer import FastEngine
    model, norm, R = C.load_original_model()
    eng = C.single_workload_engine(R.PRECOMPUTE_BASE, wl)
    gp = C.GEMPredictor(model, norm, eng, workload=wl, device="cpu")
    P = C.CachedPredictor(C.GEMPredictor(model, norm, FastEngine(eng), workload=wl, device="cpu"))
    rng = np.random.RandomState(0)
    cfgs = [DEFAULT_CONFIG] + [C._project_to_valid(sample_random_config(rng)) for _ in range(n - 1)]
    ok = True
    for c in cfgs:
        d1 = abs(P.predict_cpi(c) - gp.predict_cpi(c)); d2 = np.abs(P.predict_mscs(c) - gp.predict_mscs(c)).max()
        a, ca, _ = C.DE._coordinated_step(c, gp, None)
        b, cb, _ = C.coord_step(c, P, "mscs", True)
        same = C.ckey(a) == C.ckey(b) and {k: v["action"] for k, v in ca.items()} == {k: v["action"] for k, v in cb.items()}
        ok &= d1 < 1e-6 and d2 < 1e-6 and same
        print(f"  cpi diff {d1:.1e}  mscs diff {d2:.1e}  original step == mirror: {same}  changes {list(ca)}")
    print("SELFCHECK", "PASS" if ok else "FAIL")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["bo"] + list(C.ARMS))
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bo-budget", type=int, default=500)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--max-slices", type=int, default=200)
    ap.add_argument("--no-fast", action="store_true", help="use the original dse_infer features")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    import run_gem_dse as R
    os.makedirs(args.out, exist_ok=True)
    bo_path = os.path.join(args.out, f"bo_base_s{args.seed}.json")
    path = bo_path if args.arm == "bo" else os.path.join(args.out, f"{args.arm}_s{args.seed}.json")
    if os.path.exists(path):
        sys.exit(f"exists, refusing to overwrite: {path}")
    if args.arm != "bo" and not os.path.exists(bo_path):
        sys.exit(f"run --arm bo first ({bo_path} missing)")

    t0 = time.time(); A = AreaRO()
    W = Workers(R.BENCHMARKS, args.max_slices, fast=not args.no_fast)
    print(f"[{args.arm} s{args.seed}] {len(R.BENCHMARKS)} workers ready [{time.time()-t0:.0f}s]", flush=True)

    if args.arm == "bo":
        J = JointRemote(W)
        bo = C.DE.run_bo_gem(J, A, budget=args.bo_budget, seed=args.seed, verbose=True)
        costs = [J.cost_at.get(C.ckey(c), W.cost) for c in bo["configs"]]
        res = {"arm": "bo", "seed": args.seed, "configs": bo["configs"], "cpis": bo["cpis"],
               "areas": bo["areas"], "cost": costs, "total_cost": W.cost, "wall": time.time() - t0}
    else:
        base = json.load(open(bo_path))
        cfgs, cpis, areas = list(base["configs"]), list(base["cpis"]), list(base["areas"])
        cost0 = base["total_cost"]; costs = list(base["cost"]); origin = ["bo"] * len(cfgs)
        logs, prev_len = [], 0
        for rnd in range(args.rounds):
            pidx = C.DE.compute_pareto_front(list(zip(cpis, areas)))
            todo = list(pidx.tolist()) if rnd == 0 else [i for i in pidx if i >= prev_len]
            prev_len = len(cpis)
            if not todo:
                print(f"  round {rnd+1}: no new Pareto points, stop", flush=True); break
            print(f"  round {rnd+1}: refining {len(todo)} points", flush=True)
            for j, pi in enumerate(todo):
                outs = W.refine_all(cfgs[pi], args.arm, args.steps)
                ends = [o[0] for o in outs]
                A.prefetch(ends)
                g = W.gmean_cpi(ends)
                for wl, (end, lg), cpi in zip(W.wls, outs, g):
                    a = A(end)
                    logs.append({"round": rnd, "start": int(pi), "nav": wl, "steps": lg, "kept": a is not None})
                    if a is None:
                        continue
                    cfgs.append(end); cpis.append(cpi); areas.append(a)
                    costs.append(cost0 + W.cost); origin.append(f"r{rnd}:{wl}")
                print(f"    [{j+1}/{len(todo)}] start {pi} cpi {cpis[pi]:.4f} -> best {min(g):.4f} "
                      f"cost {cost0 + W.cost:.0f} [{time.time()-t0:.0f}s]", flush=True)
            A.flush()
            json.dump({"partial": True, "round": rnd, "n": len(cpis)},
                      open(path.replace(".json", ".progress.json"), "w"))
        res = {"arm": args.arm, "seed": args.seed, "configs": cfgs, "cpis": cpis, "areas": areas,
               "cost": costs, "origin": origin, "total_cost": cost0 + W.cost, "logs": logs,
               "args": vars(args), "wall": time.time() - t0}
    A.flush(); W.close()
    json.dump(res, open(path, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[{args.arm} s{args.seed}] done: {len(res['cpis'])} pts, cost {res['total_cost']:.0f} "
          f"joint queries, {res['wall']:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
