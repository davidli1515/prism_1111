#!/usr/bin/env python3
"""Tables for check A (prune_*.json) and check B (pairs_*.json).
Usage: python mscs_guide/summarize.py [--dir mscs_guide/results]"""
import os, json, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INC = ["mscs_relief", "mscs_relief_area", "mscs_dom", "mscs_dom_area",
       "static", "static_area", "prev", "prev_area", "area_only"]
PAIR = ["mscs_cover", "mscs_2b", "scalar_eff", "scalar_gain", "random"]


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "results"))
    d = ap.parse_args().dir

    print("=== A1  increase pruning: oracle (max gain/area) in top-k / best eff of top-k vs oracle ===")
    allrows = []
    for f in sorted(glob.glob(os.path.join(d, "prune_*.json"))):
        z = json.load(open(f)); rows = z["inc"]; allrows += rows
        print(f"\n{z['bench']}  states={len(rows)}  mean n={mean([r['n'] for r in rows]):.1f}")
        for m in INC:
            rr = [r[m] for r in rows if m in r]
            if rr:
                print(f"  {m:17s} hit@1 {mean([x['hit@1'] for x in rr]):.2f}  hit@3 {mean([x['hit@3'] for x in rr]):.2f}"
                      f"  hit@5 {mean([x['hit@5'] for x in rr]):.2f}  eff@3 {mean([x['eff@3'] for x in rr]):.2f}"
                      f"  eff@5 {mean([x['eff@5'] for x in rr]):.2f}  mean rank {mean([x['rank'] for x in rr]):.1f}")
        n = mean([r["n"] for r in rows])
        print(f"  {'random(analytic)':17s} hit@1 {1/n:.2f}  hit@3 {3/n:.2f}  hit@5 {5/n:.2f}")
    if allrows:
        print("\nPOOLED")
        for m in INC:
            rr = [r[m] for r in allrows if m in r]
            print(f"  {m:17s} hit@3 {mean([x['hit@3'] for x in rr]):.2f}  hit@5 {mean([x['hit@5'] for x in rr]):.2f}"
                  f"  eff@5 {mean([x['eff@5'] for x in rr]):.2f}  (n={len(rr)})")

    print("\n=== A2  reclaim ordering: queries to reach the scalar reclaim ===")
    for f in sorted(glob.glob(os.path.join(d, "prune_*.json"))):
        z = json.load(open(f)); rows = z["dec"]
        if rows:
            print(f"  {z['bench']:17s} cand {mean([r['n_cand'] for r in rows]):.1f}  q_area {mean([r['q_area'] for r in rows]):.1f}"
                  f"  q_mscs {mean([r['q_mscs'] for r in rows]):.1f}  first-acceptable(area)=oracle "
                  f"{mean([r['first_area_is_oracle'] for r in rows]):.2f}  mscs first-acc saved/oracle "
                  f"{mean([r['first_mscs_saved_ratio'] for r in rows]):.2f}")

    print("\n=== B  pair moves on low-gain states ===")
    for f in sorted(glob.glob(os.path.join(d, "pairs_*.json"))):
        z = json.load(open(f)); rows = z["rows"]
        if not rows:
            continue
        print(f"\n{z['bench']}  states={len(rows)}  oracle pair/best single: eff {mean([r['pair_vs_single_eff'] for r in rows]):.2f}"
              f"  gain {mean([r['pair_vs_single_gain'] for r in rows]):.2f}  frac pairs with synergy>0 "
              f"{mean([r['frac_syn_pos'] for r in rows]):.2f}")
        for m in PAIR:
            rr = [r["res"][m] for r in rows]
            hit = mean([r["top_syn_hit"].get(m) for r in rows]) if m != "random" else float("nan")
            print(f"  {m:12s} eff/oracle {mean([x['eff_ratio'] for x in rr]):.2f}  gain/oracle {mean([x['gain_ratio'] for x in rr]):.2f}"
                  f"  beats single eff {mean([x['beats_single_eff'] for x in rr]):.2f}"
                  f"  gain {mean([x['beats_single_gain'] for x in rr]):.2f}  top10-synergy hits {hit:.1f}")


if __name__ == "__main__":
    main()
