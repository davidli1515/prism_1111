#!/usr/bin/env python3
"""
Collect the values of the original Fig. 4(a) into one JSON, unrounded. Same runs and same omitted ratios as
plot_fig1 in plot_exp4_final_717.py (results_77, one run per point). Refuses to overwrite the output.

Usage: python collect_fig4a_label_efficiency_1101.py
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures_final", "fig4a_label_efficiency_1101.json")
DIRS = {"Concorde": "results_77/exp4_18k_concorde",
        "V1": "results_77/exp4_18k_vanilla",
        "V2": "results_77/exp4_18k_supervised_only",
        "V3": "results_77/exp4_18k",
        "PRISM": "results_77/exp4_18k_extra_unlabeled"}
SKIP_PCT = {10, 25, 65, 68}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists, refusing to overwrite")
    curves = {}
    for key, d in DIRS.items():
        pts = {}
        for fname in sorted(os.listdir(os.path.join(HERE, d))):
            if not (fname.startswith("exp4_") and fname.endswith(".json")):
                continue
            r = json.load(open(os.path.join(HERE, d, fname)))
            pct = round(r.get("labeled_ratio", r.get("ratio")) * 100)
            if pct not in SKIP_PCT:
                pts[pct] = r["overall_mape"]
        curves[key] = {"dir": d, "pct": sorted(pts), "overall_mape": [pts[p] for p in sorted(pts)]}
    meta = {"metric": "overall_mape (%), test set",
            "source": "plot_fig1 in plot_exp4_final_717.py, one run per point",
            "omitted_pct": sorted(SKIP_PCT)}
    json.dump({"meta": meta, "curves": curves}, open(OUT, "w"), indent=2)
    print(f"Saved {OUT}")
    for key, c in curves.items():
        print(f"  {key:<9}", " ".join(f"{p}:{v:.2f}" for p, v in zip(c["pct"], c["overall_mape"])))


if __name__ == "__main__":
    main()
