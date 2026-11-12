#!/usr/bin/env python3
"""
DSE in the 17-parameter space (space17.FIXED held constant), joint 17 SPEC benchmarks, ORIGINAL
pipeline (gem_full.pt, dse_exp normalizer, GEMPredictor slices, commit-stage CPI) with bit-exact
fast features (coord_ablation/fast_infer.py) and race-free McPAT (area17.py).

  --arm bo                 dse_engine.run_bo_gem itself, sampler swapped to the 17-param space
  --arm <arm> --mode orig  paper schedule (run_gem_dse.py): refine rounds over Pareto points, each
                           point refined 17x (one benchmark navigates), END points archived
  --arm <arm> --mode iso   same, but every step must stay within +delta of the walk's start area
                           (FastArea, ratio) and EVERY step's config is archived (geomean-scored)
arms: mscs_protect | mscs_noprotect | scalar_protect | scalar_noprotect
Cost: every new (config, workload) evaluation is counted; 17 = 1 joint query; stored per point.
Writes only dse17/results/ (refuses to overwrite), dse17/area_cache/, dse17/mcpat_work/.
"""
import os, sys, json, time, argparse
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import space17 as S
C = S.C
ARMS = C.ARMS


def worker_loop(wl, conn, max_slices, use_area_model):
    import io, contextlib, torch
    torch.set_num_threads(1)
    sys.path.insert(0, HERE)
    import steps17
    from fast_infer import FastEngine
    with contextlib.redirect_stdout(io.StringIO()):
        model, norm, R = C.load_original_model()
        eng = FastEngine(C.single_workload_engine(R.PRECOMPUTE_BASE, wl, max_slices))
        gp = C.GEMPredictor(model, norm, eng, workload=wl, device="cpu")
    P = C.CachedPredictor(gp)
    FA = None
    if use_area_model:
        from area17 import FastArea
        FA = FastArea()
    conn.send(("ready", wl))
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            break
        n0 = P.n_eval
        if msg[0] == "cpi":
            out = [P.predict_cpi(c) for c in msg[1]]
        else:                                                    # refine
            _, cfg, arm, steps, mode, delta, strides = msg
            inc, prot = ARMS[arm]
            out = steps17.refine17(cfg, P, inc, prot, steps,
                                   band_fn=FA if mode == "iso" else None, delta=delta,
                                   strides=tuple(strides) if mode == "iso" else (1,))
        conn.send((out, P.n_eval - n0))


class Workers:
    def __init__(self, wls, max_slices, use_area_model):
        ctx = mp.get_context("spawn")
        self.wls, self.conns, self.procs, self.n_eval = list(wls), [], [], 0
        for wl in self.wls:
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker_loop, args=(wl, b, max_slices, use_area_model), daemon=True)
            p.start(); self.conns.append(a); self.procs.append(p)
        for c in self.conns:
            c.recv()

    def _all(self, msg):
        for c in self.conns:
            c.send(msg)
        outs = []
        for c in self.conns:
            o, n = c.recv(); self.n_eval += n; outs.append(o)
        return outs

    def gmean_cpi(self, cfgs):
        if not cfgs:
            return []
        per = np.array(self._all(("cpi", cfgs)))
        return np.exp(np.log(per).mean(0)).tolist()

    @property
    def cost(self):
        return self.n_eval / len(self.wls)

    def close(self):
        for c in self.conns:
            c.send(("stop",))
        for p in self.procs:
            p.join(timeout=30)


class JointRemote:
    def __init__(self, W):
        self.W, self.cost_at = W, {}

    def predict_cpi(self, c):
        v = self.W.gmean_cpi([c])[0]
        self.cost_at.setdefault(C.ckey(c), self.W.cost)
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["bo"] + list(ARMS))
    ap.add_argument("--mode", default="orig", choices=["orig", "iso"])
    ap.add_argument("--delta", type=float, default=0.03)
    ap.add_argument("--strides", default="1,2,4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bo-budget", type=int, default=500)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--max-slices", type=int, default=200)
    ap.add_argument("--workers-area", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    import run_gem_dse as R
    from area17 import AreaStore, MODEL
    os.makedirs(args.out, exist_ok=True)
    bo_path = os.path.join(args.out, f"bo_base_s{args.seed}.json")
    tag = "" if args.mode == "orig" else f"_d{args.delta:g}"
    path = bo_path if args.arm == "bo" else os.path.join(args.out, f"{args.arm}_{args.mode}{tag}_s{args.seed}.json")
    if os.path.exists(path):
        sys.exit(f"exists, refusing to overwrite: {path}")
    if args.arm != "bo" and not os.path.exists(bo_path):
        sys.exit(f"run --arm bo first ({bo_path} missing)")
    if args.mode == "iso" and not os.path.exists(MODEL):
        sys.exit("run fit_area_model.py first")

    t0 = time.time(); A = AreaStore(args.workers_area)
    W = Workers(R.BENCHMARKS, args.max_slices, args.mode == "iso")
    print(f"[{args.arm} {args.mode} s{args.seed}] workers ready [{time.time()-t0:.0f}s]", flush=True)

    if args.arm == "bo":
        J = JointRemote(W)
        with S.patched_bo_sampler():
            bo = C.DE.run_bo_gem(J, A, budget=args.bo_budget, seed=args.seed, verbose=True)
        assert all(all(c[k] == v for k, v in S.FIXED.items()) for c in bo["configs"])
        res = {"arm": "bo", "seed": args.seed, "configs": bo["configs"], "cpis": bo["cpis"],
               "areas": bo["areas"], "cost": [J.cost_at.get(C.ckey(c), W.cost) for c in bo["configs"]],
               "total_cost": W.cost, "fixed": S.FIXED, "wall": time.time() - t0}
    else:
        base = json.load(open(bo_path))
        cfgs, cpis, areas = list(base["configs"]), list(base["cpis"]), list(base["areas"])
        costs, origin = list(base["cost"]), ["bo"] * len(cfgs)
        cost0, seen = base["total_cost"], {C.ckey(c) for c in cfgs}
        strides = [int(s) for s in args.strides.split(",")]
        walks, prev_len = [], 0
        for rnd in range(args.rounds):
            pidx = C.DE.compute_pareto_front(list(zip(cpis, areas)))
            todo = list(pidx.tolist()) if rnd == 0 else [i for i in pidx if i >= prev_len]
            prev_len = len(cpis)
            if not todo:
                break
            print(f"  round {rnd+1}: refining {len(todo)} points", flush=True)
            for j, pi in enumerate(todo):
                outs = W._all(("refine", cfgs[pi], args.arm, args.steps, args.mode, args.delta, strides))
                new = []
                for wl, (end, pth, lg) in zip(W.wls, outs):
                    cand = pth if args.mode == "iso" else ([end] if pth else [])
                    for c in cand:
                        k = C.ckey(c)
                        if k not in seen:
                            seen.add(k); new.append((wl, c))
                    walks.append({"round": rnd, "start": int(pi), "nav": wl, "n_steps": len(pth), "log": lg,
                                  "start_area": areas[pi], "end_key": list(map(list, C.ckey(end)))})
                A.prefetch([c for _, c in new])
                keep = [(wl, c) for wl, c in new if A(c) is not None]
                g = W.gmean_cpi([c for _, c in keep])
                for (wl, c), cpi in zip(keep, g):
                    cfgs.append(c); cpis.append(cpi); areas.append(A(c))
                    costs.append(cost0 + W.cost); origin.append(f"r{rnd}:{wl}")
                print(f"    [{j+1}/{len(todo)}] start {pi} area {areas[pi]:.1f} cpi {cpis[pi]:.4f} "
                      f"+{len(keep)} pts (dropped {len(new)-len(keep)}) best {min(g) if g else float('nan'):.4f} "
                      f"cost {cost0 + W.cost:.0f} [{time.time()-t0:.0f}s]", flush=True)
            A.flush()
        res = {"arm": args.arm, "mode": args.mode, "seed": args.seed, "configs": cfgs, "cpis": cpis,
               "areas": areas, "cost": costs, "origin": origin, "total_cost": cost0 + W.cost,
               "walks": walks, "args": vars(args), "fixed": S.FIXED, "wall": time.time() - t0}
    A.flush(); W.close()
    json.dump(res, open(path, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[{args.arm} {args.mode} s{args.seed}] done: {len(res['cpis'])} pts, cost {res['total_cost']:.0f}, "
          f"{res['wall']:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
