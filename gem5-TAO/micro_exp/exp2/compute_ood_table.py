"""
Compute GEM vs Concorde win/loss average MAPE table.
Usage: python compute_ood_table.py --dir results_ood_ablation/
"""
import json, glob, argparse, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="/scratch/gem5-TAO/micro_exp/exp2/results_ood_ablation")
args = parser.parse_args()

files = sorted(glob.glob(f"{args.dir}/ood_abl_*.json"))
results = []
for f in files:
    r = json.load(open(f))
    if "concorde" in r:
        results.append(r)

gem_wins, con_wins = [], []
for r in results:
    gem_best = min(r["full"], r["const_only"], r["mono_only"])
    if gem_best < r["concorde"]:
        gem_wins.append({"bm": r["target"], "gem": gem_best, "con": r["concorde"]})
    else:
        con_wins.append({"bm": r["target"], "gem": gem_best, "con": r["concorde"]})

print(f"{'':20} {'GEM avg MAPE':>14} {'Concorde avg MAPE':>18} {'Difference':>12}")
print("-" * 66)

if gem_wins:
    g = np.mean([x["gem"] for x in gem_wins])
    c = np.mean([x["con"] for x in gem_wins])
    print(f"{'GEM wins (%d/%d)' % (len(gem_wins), len(results)):20} {g:>13.2f}% {c:>17.2f}% {c-g:>11.2f}%")

if con_wins:
    g = np.mean([x["gem"] for x in con_wins])
    c = np.mean([x["con"] for x in con_wins])
    print(f"{'Concorde wins (%d/%d)' % (len(con_wins), len(results)):20} {g:>13.2f}% {c:>17.2f}% {g-c:>11.2f}%")

print("-" * 66)
all_g = np.mean([min(r["full"], r["const_only"], r["mono_only"]) for r in results])
all_c = np.mean([r["concorde"] for r in results])
print(f"{'Overall':20} {all_g:>13.2f}% {all_c:>17.2f}% {abs(all_g-all_c):>11.2f}%")

print(f"\nGEM wins benchmarks: {', '.join(x['bm'] for x in gem_wins)}")
print(f"Concorde wins benchmarks: {', '.join(x['bm'] for x in con_wins)}")
