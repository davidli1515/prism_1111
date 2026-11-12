#!/usr/bin/env python3
"""
Collect the values of Fig. 3 (model capacity, [256,128] vs [512,256]) into one JSON, unrounded.
Every value is read from a result file; a cell with no result file is null. PRISM [256,128] has no
result file, so it is null. Refuses to overwrite the output.

Usage: python collect_fig3_capacity_1102.py
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "fig3_capacity_1102.json")
R = "results"
EXP4 = "../exp4"


def from_file(path):
    d = json.load(open(os.path.join(HERE, path)))
    return {"value": d["overall_mape"], "source": path}


MISSING = {"value": None, "source": None}

ROWS = [("Concorde", from_file(f"{R}/exp1_v2_run2/exp1_256_128_183k/concorde.json"),
         from_file(f"{R}/exp1_v2_run2/exp1_512_256_183k/concorde.json")),
        ("V1", from_file(f"{R}/exp1_v2_run2/exp1_256_128_183k/gem_vanilla.json"),
         from_file(f"{R}/exp1_v2_run2/exp1_512_256_183k/gem_vanilla.json")),
        ("V2", from_file(f"{R}/exp1_v3_run2/exp1_256_128_183k/gem_full.json"),
         from_file(f"{R}/exp1_v3_run2/exp1_512_256_183k/gem_full.json")),
        ("PRISM", MISSING, from_file(f"{EXP4}/results_run2/exp4_18k_extra_unlabeled/exp4_100pct.json"))]


def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists, refusing to overwrite")
    rows = [{"label": label, "s256": s256, "s512": s512} for label, s256, s512 in ROWS]
    meta = {"metric": "CPI MAPE (%), test set, 183k", "source": "result files listed per cell"}
    json.dump({"meta": meta, "rows": rows}, open(OUT, "w"), indent=2)
    print(f"Saved {OUT}")
    for r in rows:
        print(f"  {r['label']:<9} [256,128]={fmt(r['s256']['value'])}  [512,256]={fmt(r['s512']['value'])}")


if __name__ == "__main__":
    main()
