#!/usr/bin/env python3
"""
Summarize DSE v2 runs with ONE common reference point for all methods/seeds.

Outputs (in <dir>):
  summary.json        HV at query checkpoints, final HV (mean/std over seeds), queries to
                      reach 95% of the best final HV, Pareto sizes
  hv_vs_queries.png/pdf
  pareto_<seed>.png   Pareto fronts of all methods for one seed
Also exports <method>_s<seed>_pareto.json in the old results format
(configs/cpis/areas/pareto_indices) so dse_exp/validate_joint_dse.py --json can read it.

Usage: python summarize_dse_v2.py results_v2_1005
"""
import os, sys, json, glob, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import pareto_mask, hypervolume

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'font.size': 10, 'axes.labelsize': 11, 'figure.dpi': 300,
                     'xtick.direction': 'in', 'ytick.direction': 'in'})
ORDER = ["random", "bo", "greedy_scalar", "greedy_mscs", "greedy_mscs_decouple",
         "greedy_mscs_protect", "bo_scalar", "bo_mscs"]


def hv_at(r, q, ref):
    idx = [i for i, qq in enumerate(r["queries"]) if qq <= q]
    return hypervolume(np.c_[np.array(r["cpis"])[idx], np.array(r["areas"])[idx]], ref) if idx else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[250, 500, 1000, 2000, 3000])
    args = ap.parse_args()

    runs = {}
    for f in sorted(glob.glob(os.path.join(args.dir, "*_s*.json"))):
        if f.endswith("_pareto.json") or os.path.basename(f) == "summary.json":
            continue
        r = json.load(open(f))
        runs.setdefault(r["method"], []).append(r)
    allP = np.vstack([np.c_[r["cpis"], r["areas"]] for rs in runs.values() for r in rs])
    ref = (float(allP[:, 0].max() * 1.05), float(allP[:, 1].max() * 1.05))
    best = max(hypervolume(np.c_[r["cpis"], r["areas"]], ref) for rs in runs.values() for r in rs)

    summ = {"ref_point": ref, "methods": {}}
    print(f"Common ref point: CPI={ref[0]:.3f} area={ref[1]:.1f}\n")
    hdr = f"{'method':22s} {'n':>2s} " + " ".join(f"{'HV@'+str(q):>14s}" for q in args.checkpoints) \
        + f" {'final HV':>14s} {'q→95%best':>10s} {'|PF|':>5s}"
    print(hdr)
    for m in [m for m in ORDER if m in runs] + [m for m in runs if m not in ORDER]:
        rs = runs[m]
        at = {q: [hv_at(r, q, ref) for r in rs] for q in args.checkpoints}
        fin = [hypervolume(np.c_[r["cpis"], r["areas"]], ref) for r in rs]
        q95 = []
        for r in rs:
            hit = [qq for qq in sorted(set(r["queries"])) if hv_at(r, qq, ref) >= 0.95 * best]
            q95.append(hit[0] if hit else np.nan)
        pf = [len(r["pareto_indices"]) for r in rs]
        summ["methods"][m] = {"n_seeds": len(rs),
                              "hv_at": {q: [float(np.mean(v)), float(np.std(v))] for q, v in at.items()},
                              "final_hv": [float(np.mean(fin)), float(np.std(fin))],
                              "queries_to_95pct_best": [float(np.nanmean(q95)), float(np.nanstd(q95))],
                              "queries_used": [r["n_queries"] for r in rs],
                              "pareto_size": pf}
        cell = lambda v: f"{np.mean(v):7.1f}±{np.std(v):5.1f}"
        print(f"{m:22s} {len(rs):2d} " + " ".join(f"{cell(at[q]):>14s}" for q in args.checkpoints)
              + f" {cell(fin):>14s} {np.nanmean(q95):10.0f} {np.mean(pf):5.0f}")
        for r in rs:   # export for gem5 validation
            json.dump({k: r[k] for k in ("method", "configs", "cpis", "areas", "pareto_indices")},
                      open(os.path.join(args.dir, f"{m}_s{r['seed']}_pareto.json"), "w"), default=str)
    json.dump(summ, open(os.path.join(args.dir, "summary.json"), "w"), indent=2)

    # HV vs queries
    fig, ax = plt.subplots(figsize=(6, 3.6))
    grid = np.linspace(0, max(max(r["queries"]) for rs in runs.values() for r in rs), 60)
    for m, rs in runs.items():
        H = np.array([[hv_at(r, q, ref) for q in grid] for r in rs])
        ax.plot(grid, H.mean(0), label=m)
        ax.fill_between(grid, H.mean(0) - H.std(0), H.mean(0) + H.std(0), alpha=0.15)
    ax.set_xlabel("Surrogate queries"); ax.set_ylabel("Hypervolume"); ax.grid(ls=":")
    ax.legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.dir, f"hv_vs_queries.{ext}"), bbox_inches="tight")

    # Pareto fronts for each seed
    seeds = sorted({r["seed"] for rs in runs.values() for r in rs})
    for s in seeds:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for m, rs in runs.items():
            for r in rs:
                if r["seed"] != s:
                    continue
                P = np.c_[r["cpis"], r["areas"]]; F = P[pareto_mask(P)]; F = F[np.argsort(F[:, 1])]
                ax.step(F[:, 1], F[:, 0], where="post", label=m)
        ax.set_xlabel("Area (mm²)"); ax.set_ylabel("Geomean CPI"); ax.grid(ls=":"); ax.legend(fontsize=7)
        fig.savefig(os.path.join(args.dir, f"pareto_s{s}.png"), bbox_inches="tight")
    print(f"\nSaved summary.json, hv_vs_queries.png/pdf, pareto_s*.png in {args.dir}")


if __name__ == "__main__":
    main()
