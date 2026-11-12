#!/usr/bin/env python3
"""
Collect the values of the original "Semi vs. PT+FT" figure into one JSON, unrounded. Same runs and same omitted
ratios as plot_fig2 in plot_exp4_final_717.py (one run per point). The 100% lines were hardcoded at 2.7 there, here
they are the measured values. Refuses to overwrite the output.

Usage: python collect_fig_semi_vs_ptft_1101.py
"""
import glob, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures_final", "fig_semi_vs_ptft_1101.json")
SOURCES = {"V3": ("results_77/exp4_18k", "exp4_*.json", "overall_mape"),
           "Type1": ("results_77/exp4_18k_pretrain_ft_type1", "exp4_*.json", "overall_mape"),
           "Type2": ("../exp5/results_pt_ft_v2_18k", "exp4_pt_ft_v2_*.json", "pretrain_ft_v2")}
REFS = {"PRISM": "results_77/exp4_18k_extra_unlabeled/exp4_100pct.json",
        "Concorde": "results_77/exp4_18k_concorde/exp4_100pct.json"}
SKIP_PCT = {10, 25, 65, 68}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists, refusing to overwrite")
    curves = {}
    for key, (d, pattern, field) in SOURCES.items():
        pts = {}
        for f in glob.glob(os.path.join(HERE, d, pattern)):
            r = json.load(open(f))
            pct = round(r.get("labeled_ratio", r.get("ratio")) * 100)
            if pct not in SKIP_PCT:
                pts[pct] = r[field]
        curves[key] = {"dir": d, "field": field, "pct": sorted(pts), "overall_mape": [pts[p] for p in sorted(pts)]}
    refs = {key: {"file": f, "overall_mape": json.load(open(os.path.join(HERE, f)))["overall_mape"]}
            for key, f in REFS.items()}
    meta = {"metric": "overall_mape (%), test set",
            "source": "plot_fig2 in plot_exp4_final_717.py, one run per point",
            "omitted_pct": sorted(SKIP_PCT)}
    json.dump({"meta": meta, "curves": curves, "refs_100pct": refs}, open(OUT, "w"), indent=2)
    print(f"Saved {OUT}")
    for key, c in curves.items():
        print(f"  {key:<6}", " ".join(f"{p}:{v:.2f}" for p, v in zip(c["pct"], c["overall_mape"])))
    print("  100% lines:", {k: round(v["overall_mape"], 3) for k, v in refs.items()})


if __name__ == "__main__":
    main()
