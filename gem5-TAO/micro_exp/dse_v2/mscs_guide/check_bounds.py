#!/usr/bin/env python3
"""
Check C (bounds) -- DATA COLLECTION ONLY; every metric is computed offline by summarize_bounds.py,
so mappings / bound definitions can be changed without re-running.

Idea (Eyerman et al., CAL'18): per-stage CPI stacks bound the CPI reduction from removing a stall
class: gain in [min_s comp_s, max_s comp_s].  A bound covers a whole SET of designs (all configs
reachable by investing in one direction) from ONE query -> branch-and-bound / budget allocation.
Our labels are per-stage top-down (gem5 decode.cc/commit.cc): FP = stage starved by upstream,
BP = blocked by downstream, SP = squash; FP falls and BP rises along the pipeline as in Eyerman.

Per front point p (pure-BO front = first 500 queries, and the final front, of an existing v4 run):
  stack(p): per-stage CPI shares of Base/FP/BP/SP (CPI-weighted per slice, like Surrogate._eval)
  for each direction d: every +1 single of d's params and the "maxed" config (all d params at max;
  bp_type: other value), exactly evaluated; for each: cpi, area, its stack shares.
Directions (widths excluded: they act through W_min/Base, no stall class bounds them):
  FE  fetchBufferSize, l1i_assoc          L1  l1_size (shared I/D)
  BE  window, FUs, ports, l1d_assoc, L2, wbWidth                SP  bp_type
  ALL all of the above (maxed together, both predictors)
Writes results/bounds_<bench>.json (refuses to overwrite).
Usage: python mscs_guide/check_bounds.py --bench 641.leela_s
"""
import os, sys, json, time, argparse
from guide_common import HERE, ROOT, PARAMS, _shift
import numpy as np
import torch
from common import (Area, DEFAULT_CKPT, config_key, pareto_mask, get_param_values,
                    GEM_INPUT_DIM, BASE_IDX, compute_w_min)
from dse_engine import _project_to_valid
from refine_v3 import SurrogateWL
from mscs_reclaim_v2 import prefetch_areas

CATS = {"Base": 0, "FP": 1, "BP": 2, "SP": 3}
DIRS = {
    "FE": ["fetchBufferSize", "l1i_assoc"],
    "L1": ["l1_size"],
    "BE": ["numROBEntries", "numIQEntries", "LQEntries", "SQEntries", "IntALU", "IntMultDiv",
           "FP_ALU", "FP_MultDiv", "RdWrPort", "l1d_assoc", "l2_size", "l2_assoc", "wbWidth"],
    "SP": ["bp_type"],
}
DIRS["ALL"] = sorted({n for v in DIRS.values() for n in v}, key=PARAMS.index)


@torch.no_grad()
def stack(S, c):
    """(cpi, shares {cat: (4,) share of each stage's CPI}) computed per slice as in Surrogate._eval."""
    w = float(compute_w_min(c))
    X = np.stack([f[:GEM_INPUT_DIM] for f in S.engine.get_features(c, workload=S.workloads[0])])
    m = S.model(torch.from_numpy(S.norm.transform(X)).float().to(S.device))
    base = torch.clamp(m[:, BASE_IDX], min=0.01)
    cs = torch.clamp(1.0 / (base * w + 1e-8), max=100.0)                  # (n,4) per-stage CPI
    cpi_s = cs.mean(0)
    sh = {k: ((cs * m[:, [4 * s + cat for s in range(4)]]).mean(0) / cpi_s).cpu().numpy().round(6).tolist()
          for k, cat in CATS.items()}
    return float(cs.mean(1).mean()), sh


def maxed(c, d):
    if d == "SP":
        return [dict(c, bp_type=v) for v in get_param_values("bp_type") if v != c["bp_type"]]
    cn = dict(c)
    for n in DIRS[d]:
        if n != "bp_type":
            cn[n] = max(get_param_values(n))
    cn = _project_to_valid(cn)
    out = [cn] if config_key(cn) != config_key(c) else []
    if d == "ALL":
        out += [dict(cn, bp_type=v) for v in get_param_values("bp_type") if v != cn["bp_type"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--src", default=None, help="default: results_v4_1007/<bench>/bo_v4_mscs_s42.json")
    ap.add_argument("--bo-queries", type=int, default=500)
    ap.add_argument("--max-points", type=int, default=25)
    ap.add_argument("--mcpat-workers", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    path = os.path.join(args.out, f"bounds_{args.bench}.json")
    if os.path.exists(path):
        sys.exit(f"exists, refusing to overwrite: {path}")
    srcp = args.src or os.path.join(ROOT, "results_v4_1007", args.bench, "bo_v4_mscs_s42.json")
    src = json.load(open(srcp))
    C, P = src["configs"], np.c_[src["cpis"], src["areas"]]
    q, tags = np.array(src["queries"]), src["tags"]
    sets = {"bo500": np.array([i for i, t in enumerate(tags) if t.startswith("bo") and q[i] <= args.bo_queries]),
            "final": np.arange(len(C))}

    S = SurrogateWL(DEFAULT_CKPT, [args.bench], "18k", device="cpu", max_slices=50)
    A = Area()
    pre = lambda cs: prefetch_areas(A, cs, args.mcpat_workers)
    t0 = time.time()
    fronts, points = {}, []
    for fname, idx in sets.items():
        Fi = idx[pareto_mask(P[idx])]; Fi = Fi[np.argsort(P[Fi, 1])]
        fronts[fname] = P[Fi].tolist()
        pick = Fi[np.unique(np.linspace(0, len(Fi) - 1, min(args.max_points, len(Fi))).round().astype(int))]
        for i in pick:
            c = C[int(i)]; cpi_p, area_p = S.cpi(c), A(c)
            cpi_k, sh = stack(S, c)
            assert abs(cpi_k - cpi_p) / cpi_p < 1e-4, (cpi_k, cpi_p)
            pt = {"front": fname, "idx": int(i), "cpi": cpi_p, "area": area_p, "w_min": float(compute_w_min(c)),
                  "shares": sh, "dirs": {}}
            for d in DIRS:
                cand = [(n, cn) for n in DIRS[d] for cn in [_shift(c, n, +1)] if cn is not None]
                cand += [("max", cn) for cn in maxed(c, d)]
                pre([cn for _, cn in cand])
                ev = []
                for tag, cn in cand:
                    a = A(cn)
                    if a is None:
                        continue
                    cc, sc = stack(S, cn)
                    ev.append({"tag": tag, "cpi": S.cpi(cn), "area": a, "shares": sc})
                pt["dirs"][d] = ev
            points.append(pt)
            mx = {d: [round((cpi_p - e["cpi"]) / cpi_p, 3) for e in ev if e["tag"] == "max"] for d, ev in pt["dirs"].items()}
            print(f"  [{fname}] point {int(i)} cpi={cpi_p:.3f} area={area_p:.1f} "
                  f"FP/BP/SP max-stage={max(sh['FP']):.2f}/{max(sh['BP']):.2f}/{max(sh['SP']):.2f} "
                  f"maxed gain={mx} q={S.n_queries} [{time.time()-t0:.0f}s]", flush=True)
            A.flush()
    json.dump({"bench": args.bench, "src": srcp, "args": vars(args), "dirs": DIRS,
               "ref": [float(P[:, 0].max()), float(P[:, 1].max())], "fronts": fronts, "points": points,
               "queries": S.n_queries, "wall": time.time() - t0}, open(path, "w"))
    print(f"[{args.bench}] done -> {path} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
