#!/usr/bin/env python3
"""
Collect the no-D3 low-label unseen-workload OOD grid into one file: overall CPI MAPE
(mean target_mape over the 23 held-out benchmarks, seed 42, best-src-val epoch)
per source-label ratio x method x N. All methods use the 3873 Concorde input features.

Sources (per-run {dir}/{src}/{variant}/{target}/n{N}_s42.json):
  PRISM-V2, PRISM   results_ood_lowlabel_grid_nod3_1025
  Concorde          results_ood_lowlabel_grid_1008, except src5/src20 at N<=1024:
                    results_ood_lowlabel_1006 (8 targets) / results_ood_lowlabel23_1007 (other 15)
Output: results_ood_lowlabel_grid_nod3_1025/exp2_ood_nod3_1031.json
Usage: python3 make_ood_nod3_json.py
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_ood_lowlabel_grid_nod3_1025", "exp2_ood_nod3_1031.json")
ALL = sorted(f[:-4] for f in os.listdir('/scratch2/wl730/micro_traindata_packed/gem') if f.endswith('.npy'))
OLD = {"602.gcc_s", "605.mcf_s", "607.cactuBSSN_s", "638.imagick_s", "508.namd_r", "snappy", "re2", "657.xz_s"}
RATIOS = ["src5", "src20", "src50", "src80", "src100"]
NS = [32, 256, 1024, 2048]
METHODS = {"Concorde": "concorde", "PRISM-V2": "supervised", "PRISM": "prism"}   # name -> result-dir variant


def path(r, v, t, n):
    if v != 'concorde':
        d = 'results_ood_lowlabel_grid_nod3_1025'
    elif r in ('src5', 'src20') and n in (32, 256, 1024):
        d = 'results_ood_lowlabel_1006' if t in OLD else 'results_ood_lowlabel23_1007'
    else:
        d = 'results_ood_lowlabel_grid_1008'
    return os.path.join(HERE, d, r, v, t, f"n{n}_s42.json")


def main():
    assert len(ALL) == 23, len(ALL)
    assert not os.path.exists(OUT), f"{OUT} exists; not overwriting"
    res, wins = {}, {}
    for r in RATIOS:
        res[r] = {}
        runs = {m: {n: [json.load(open(path(r, v, t, n))) for t in ALL] for n in NS} for m, v in METHODS.items()}
        for m, v in METHODS.items():
            if v != 'concorde':
                assert all(x['input_features'].startswith('concorde only') and x['variant'] == v
                           for n in NS for x in runs[m][n]), (r, m)
            res[r][m] = {str(n): round(float(np.mean([x['target_mape'] for x in runs[m][n]])), 4) for n in NS}
        wins[r] = {str(n): sum(a['target_mape'] < b['target_mape'] for a, b in zip(runs['PRISM'][n], runs['Concorde'][n])) for n in NS}
    out = {"description": "Unseen-workload OOD (leave-one-benchmark-out, window-isolated), identical 3873-dim input "
                          "features for all methods: overall CPI MAPE (%) = mean over 23 held-out benchmarks of "
                          "target_mape, seed 42, model selected by source-val",
           "keys": "results[source_label_ratio][method][N]",
           "source_label_ratio": {"src5": 0.05, "src20": 0.20, "src50": 0.50, "src80": 0.80, "src100": 1.00},
           "N": NS, "targets": ALL,
           "methods": {"Concorde": "concorde", "PRISM-V2": "supervised (labeled only)",
                       "PRISM": "prism (V3 + BN fix + 40k extra unlabeled, held-out windows removed)"},
           "lambdas": {"task": 10.0, "const": 1.0, "mono": 0.01, "unlab": 1.0}, "loss_norm": "subset",
           "sources": {"PRISM-V2, PRISM": "results_ood_lowlabel_grid_nod3_1025",
                       "Concorde": "results_ood_lowlabel_grid_1008; src5,src20 & N<=1024: results_ood_lowlabel_1006 "
                                   "(8 targets: " + ", ".join(sorted(OLD)) + ") / results_ood_lowlabel23_1007 (other 15)"},
           "prism_beats_concorde_targets_of_23": wins, "results": res}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"Saved {OUT}")
    print(f"{'':7s}" + "".join(f"{'N='+str(n):>22s}" for n in NS) + "   (Concorde / V2 / PRISM  [PRISM wins])")
    for r in RATIOS:
        print(f"{r:7s}" + "".join(f"{res[r]['Concorde'][str(n)]:6.2f}/{res[r]['PRISM-V2'][str(n)]:5.2f}/{res[r]['PRISM'][str(n)]:5.2f} [{wins[r][str(n)]:2d}]" for n in NS))


if __name__ == "__main__":
    main()
