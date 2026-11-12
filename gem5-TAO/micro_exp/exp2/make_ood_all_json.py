#!/usr/bin/env python3
"""
Collect the low-label unseen-workload OOD grid into one file: overall CPI MAPE
(mean target_mape over the 23 held-out benchmarks, seed 42, best-src-val epoch)
per source-label ratio x method x N.

Sources (per-run {dir}/{src}/{variant}/{target}/n{N}_s42.json):
  results_ood_lowlabel_grid_1008                 all cells, except:
  results_ood_lowlabel_1006 (8 old targets) /
  results_ood_lowlabel23_1007 (other 15)         Concorde & V2 at src5/src20, N<=1024
                                                 (identical split; PRISM always from grid_1008)
Output: results_ood_lowlabel_grid_1008/exp2_ood_all_1011.json
Usage: python3 make_ood_all_json.py
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_ood_lowlabel_grid_1008", "exp2_ood_all_1011.json")
ALL = sorted(f[:-4] for f in os.listdir('/scratch2/wl730/micro_traindata_packed/gem') if f.endswith('.npy'))
OLD = {"602.gcc_s", "605.mcf_s", "607.cactuBSSN_s", "638.imagick_s", "508.namd_r", "snappy", "re2", "657.xz_s"}
RATIOS = ["src5", "src20", "src50", "src80", "src100"]
NS = [32, 256, 1024, 2048]
METHODS = {"Concorde": "concorde", "PRISM-V2": "supervised", "PRISM": "prism"}   # name -> result-dir variant


def path(r, v, t, n):
    if v in ('concorde', 'supervised') and r in ('src5', 'src20') and n in (32, 256, 1024):
        d = 'results_ood_lowlabel_1006' if t in OLD else 'results_ood_lowlabel23_1007'
    else:
        d = 'results_ood_lowlabel_grid_1008'
    return os.path.join(HERE, d, r, v, t, f"n{n}_s42.json")


def main():
    assert len(ALL) == 23, len(ALL)
    assert not os.path.exists(OUT), f"{OUT} exists; not overwriting"
    res = {}
    for r in RATIOS:
        res[r] = {}
        for m, v in METHODS.items():
            res[r][m] = {str(n): round(float(np.mean(
                [json.load(open(path(r, v, t, n)))['target_mape'] for t in ALL])), 4) for n in NS}
    out = {"description": "Unseen-workload OOD (leave-one-benchmark-out, window-isolated): overall CPI MAPE (%) "
                          "= mean over 23 held-out benchmarks of target_mape, seed 42, model selected by source-val",
           "keys": "results[source_label_ratio][method][N]",
           "source_label_ratio": {"src5": 0.05, "src20": 0.20, "src50": 0.50, "src80": 0.80, "src100": 1.00},
           "N": NS, "targets": ALL,
           "methods": {"Concorde": "concorde", "PRISM-V2": "supervised (labeled only)",
                       "PRISM": "prism (V3 + BN fix + 40k extra unlabeled, held-out windows removed)"},
           "lambdas": {"task": 10.0, "const": 1.0, "mono": 0.01, "unlab": 1.0}, "loss_norm": "subset",
           "sources": {"default": "results_ood_lowlabel_grid_1008",
                       "Concorde/PRISM-V2 at src5,src20 & N<=1024": "results_ood_lowlabel_1006 (8 targets: "
                       + ", ".join(sorted(OLD)) + ") / results_ood_lowlabel23_1007 (other 15)"},
           "results": res}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
