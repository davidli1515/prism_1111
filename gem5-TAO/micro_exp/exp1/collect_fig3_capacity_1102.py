#!/usr/bin/env python3
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "fig3_capacity_1102.json")
R = "results"
EXP4 = "/scratch/gem5-TAO/micro_exp/exp4"


def from_file(path):
    d = json.load(open(path if path.startswith("/") else os.path.join(HERE, path)))
    return {"value": d["overall_mape"], "source": path}


def typed(value, reference=None):
    v = {"value": value, "source": "typed into plot_exp1_scale.py"}
    if reference:
        v["reference_file"] = from_file(reference)
    return v


ROWS = [("Concorde", typed(2.9, f"{R}/exp1_v2_74/exp1_256_128_183k/concorde.json"),
         typed(2.7, f"{R}/exp1_v2_74/exp1_512_256_183k/concorde.json")),
        ("V1", from_file(f"{R}/exp1_v2_74/exp1_256_128_183k/gem_vanilla.json"),
         from_file(f"{R}/exp1_v2_74/exp1_512_256_183k/gem_vanilla.json")),
        ("V2", from_file(f"{R}/exp1_v3_76/exp1_256_128_183k/gem_full.json"),
         from_file(f"{R}/exp1_v3_76/exp1_512_256_183k/gem_full.json")),
        ("PRISM", typed(3.2), typed(2.7, f"{EXP4}/results_77/exp4_18k_extra_unlabeled/exp4_100pct.json"))]


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists, refusing to overwrite")
    rows = [{"label": label, "s256": s256, "s512": s512} for label, s256, s512 in ROWS]
    meta = {"metric": "CPI MAPE (%), test set, 183k", "source": "main figure of plot_exp1_scale.py"}
    json.dump({"meta": meta, "rows": rows}, open(OUT, "w"), indent=2)
    print(f"Saved {OUT}")
    for r in rows:
        print(f"  {r['label']:<9} [256,128]={r['s256']['value']:.3f}  [512,256]={r['s512']['value']:.3f}")


if __name__ == "__main__":
    main()
