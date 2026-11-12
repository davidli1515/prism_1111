"""
Summarize exp2 v2 OOD results (results_ood/{variant}/{target}/n{N}_s{seed}.json).

Prints, per variant x n_shot: mean target MAPE over targets (mean ± std over seeds),
plus the number of finished (target, seed) runs. Optionally per-target tables.

Usage:
    python summarize_ood.py
    python summarize_ood.py --metric target_mape_last_epoch --per-target
"""
import os, re, json, glob, argparse
import numpy as np
from collections import defaultdict

VARIANT_ORDER = ["concorde", "vanilla", "supervised", "prism"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/path/to/gem5/micro_exp/exp2/results_ood")
    ap.add_argument("--metric", default="target_mape",
                    choices=["target_mape", "target_p90", "target_mape_last_epoch",
                             "target_mape_oracle"])
    ap.add_argument("--per-target", action="store_true")
    ap.add_argument("--out", default=None, help="Optional JSON dump of the summary")
    args = ap.parse_args()

    # res[variant][n][seed][target] = value
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    pat = re.compile(r"n(\d+)_s(\d+)\.json$")
    for f in glob.glob(os.path.join(args.results_dir, "*", "*", "n*_s*.json")):
        m = pat.search(f)
        if not m: continue
        d = json.load(open(f))
        res[d["variant"]][int(m.group(1))][int(m.group(2))][d["target"]] = d[args.metric]

    variants = [v for v in VARIANT_ORDER if v in res] + \
               sorted(v for v in res if v not in VARIANT_ORDER)
    shots = sorted({n for v in res for n in res[v]})
    targets = sorted({t for v in res for n in res[v] for s in res[v][n] for t in res[v][n][s]})

    print(f"Metric: {args.metric}  (mean over targets; ± = std over seeds; [runs])\n")
    print(f"{'variant':12s}" + "".join(f"{'N='+str(n):>20s}" for n in shots))
    summary = {}
    for v in variants:
        row = f"{v:12s}"
        for n in shots:
            per_seed = [np.mean(list(res[v][n][s].values())) for s in res[v][n]
                        if len(res[v][n][s]) == len(targets)]   # only complete seeds
            n_runs = sum(len(res[v][n][s]) for s in res[v][n])
            if per_seed:
                cell = f"{np.mean(per_seed):.2f}±{np.std(per_seed):.2f} [{n_runs}]"
                summary.setdefault(v, {})[n] = {"mean": float(np.mean(per_seed)),
                                                 "std": float(np.std(per_seed)),
                                                 "n_seeds": len(per_seed)}
            else:
                cell = f"- [{n_runs}]" if n_runs else "-"
            row += f"{cell:>20s}"
        print(row)
    print(f"\n({len(targets)} targets; a cell is filled only when a seed covers all targets)")

    if args.per_target:
        for n in shots:
            print(f"\n=== N={n} (mean over available seeds) ===")
            print(f"{'target':22s}" + "".join(f"{v:>12s}" for v in variants))
            for t in targets:
                row = f"{t:22s}"
                for v in variants:
                    vals = [res[v][n][s][t] for s in res[v][n] if t in res[v][n][s]]
                    row += f"{np.mean(vals):>12.2f}" if vals else f"{'-':>12s}"
                print(row)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
