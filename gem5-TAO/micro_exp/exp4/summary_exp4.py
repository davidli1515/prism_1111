"""Quick summary of all exp4 results.
Usage: python summary_exp4.py --dataset 40k
"""
import json, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=["18k", "40k"])
args = parser.parse_args()

ds = args.dataset
BASE = f"./results/exp4_{ds}"

VARIANTS = {
    "(g) full":       "",
    "(e) const only": "_const_only",
    "(f) mono only":  "_mono_only",
    "(h) sup only":   "_supervised_only",
    "(i) concorde":   "_concorde",
    "(j) vanilla":    "_vanilla",
}

data = {}
for label, suffix in VARIANTS.items():
    dirpath = f"{BASE}{suffix}"
    if not os.path.isdir(dirpath):
        continue
    data[label] = {}
    for f in sorted(os.listdir(dirpath)):
        if f.startswith("exp4_") and f.endswith(".json"):
            with open(os.path.join(dirpath, f)) as fh:
                d = json.load(fh)
            data[label][d["labeled_ratio"]] = d["overall_mape"]

all_ratios = sorted(set(r for v in data.values() for r in v))

header = f"{'Ratio':<8}"
for label in data:
    header += f"  {label:>16}"
print(f"{'='*len(header)}")
print(f"OVERALL MAPE SUMMARY (dataset={ds})")
print(f"{'='*len(header)}")
print(header)
print(f"{'-'*len(header)}")
for r in all_ratios:
    line = f"{r*100:>5.0f}%  "
    for label in data:
        if r in data[label]:
            line += f"  {data[label][r]:>14.2f}%"
        else:
            line += f"  {'—':>15}"
    print(line)
print(f"{'='*len(header)}")