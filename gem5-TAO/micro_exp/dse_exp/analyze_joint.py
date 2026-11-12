#!/usr/bin/env python3
"""
Per-benchmark calibration analysis for joint 17-bm DSE validation.

Reads joint_validation_raw.json (per config × benchmark CPI), applies a
per-benchmark affine calibration (fit gem5 = a*surrogate + b using the
surrogate PER-BENCHMARK slice CPI is not stored, so here we calibrate the
config-level gmean instead), and reports Spearman + MAPE on config gmean.

Since the raw file stores only gem5 per-bench CPI (not surrogate per-bench),
we compute:
  - gem5 gmean per config from whichever benchmarks succeeded
  - surrogate gmean per config from the summary
and calibrate at the gmean level (single global affine), plus report the
raw correlation.

Usage:
  python analyze_joint.py validate_joint_mscs [min_ok]
"""
import json, sys, os
import numpy as np
from scipy import stats as st

outdir = sys.argv[1] if len(sys.argv) > 1 else "validate_joint_mscs"
min_ok = int(sys.argv[2]) if len(sys.argv) > 2 else None

raw = json.load(open(os.path.join(outdir, "joint_validation_raw.json")))
summary = json.load(open(os.path.join(outdir, "joint_validation_summary.json")))


def parse_cpi(stats_path):
    if not os.path.isfile(stats_path):
        return None
    insts = cycles = cpi_direct = None
    with open(stats_path) as f:
        for line in f:
            if "switch_cpus_1" not in line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key, val = parts[0], parts[1]
            try:
                v = float(val)
            except ValueError:
                continue
            if key.endswith(".cpi"):
                cpi_direct = v
            elif key.endswith(".numInsts") or key.endswith(".committedInsts"):
                insts = v
            elif key.endswith(".numCycles"):
                cycles = v
    if cpi_direct is not None:
        return cpi_direct
    if insts and cycles and insts > 0:
        return cycles / insts
    return None


# Re-parse any parse_failed entry from stats.txt on disk (handles misplaced/late stats)
n_recovered = 0
for r in raw:
    if not r.get("cpi"):
        sp = os.path.join(outdir, "runs", f"cfg{r['cfg_id']}", r["bench"], "stats.txt")
        cpi = parse_cpi(sp)
        if cpi:
            r["cpi"] = cpi
            r["error"] = None
            n_recovered += 1
if n_recovered:
    print(f"Recovered {n_recovered} previously-failed runs from stats.txt on disk.\n")


# Which benchmarks succeeded everywhere?
from collections import defaultdict
ok_by_bench = defaultdict(int)
tot_by_bench = defaultdict(int)
for r in raw:
    tot_by_bench[r["bench"]] += 1
    if r.get("cpi"):
        ok_by_bench[r["bench"]] += 1

print("Benchmark success:")
good_benches = []
for b in sorted(tot_by_bench):
    ok, tot = ok_by_bench[b], tot_by_bench[b]
    flag = "OK" if ok == tot else "PARTIAL/FAIL"
    if ok == tot:
        good_benches.append(b)
    print(f"  {b:<18} {ok}/{tot}  {flag}")
print(f"\nUsing {len(good_benches)} fully-successful benchmarks for gmean.\n")

# Recompute gem5 gmean per config using ONLY good benches (consistent set)
cfg_gem5 = {}
cfg_cpis = defaultdict(dict)
for r in raw:
    if r["bench"] in good_benches and r.get("cpi"):
        cfg_cpis[r["cfg_id"]][r["bench"]] = r["cpi"]

for ci, d in cfg_cpis.items():
    if len(d) == len(good_benches):
        vals = np.array(list(d.values()))
        cfg_gem5[ci] = float(np.exp(np.mean(np.log(vals))))

# surrogate gmean from summary
cfg_surr = {s["cfg_id"]: s["surrogate_gmean"] for s in summary}

cids = sorted(set(cfg_gem5) & set(cfg_surr))
su = np.array([cfg_surr[c] for c in cids])
ge = np.array([cfg_gem5[c] for c in cids])

print(f"Configs with complete gmean: {len(cids)}\n")

# Raw metrics
mape = np.mean(np.abs(su - ge) / ge) * 100
pear = np.corrcoef(su, ge)[0, 1]
rho, _ = st.spearmanr(su, ge)
print(f"RAW (surrogate vs gem5 gmean over {len(good_benches)} benches):")
print(f"  MAPE={mape:.2f}%  Pearson={pear:.4f}  Spearman={rho:.4f}\n")

# Global affine calibration gem5 = a*surrogate + b
a, b = np.polyfit(su, ge, 1)
ge_pred = a * su + b
mape_cal = np.mean(np.abs(ge_pred - ge) / ge) * 100
print(f"After global affine calibration (gem5 = {a:.3f}*surr + {b:+.3f}):")
print(f"  MAPE={mape_cal:.2f}%  (Spearman unchanged={rho:.4f})\n")

# Signed bias
rel = (su - ge) / ge * 100
print(f"Signed bias (surrogate-gem5)/gem5: mean {rel.mean():+.2f}%, "
      f"std {rel.std():.2f}%")

# Per-config table
print(f"\n{'cfg':>6} {'surrogate':>10} {'gem5':>10} {'err%':>7}")
print("-" * 38)
for c in sorted(cids, key=lambda x: cfg_gem5[x]):
    e = (cfg_surr[c] - cfg_gem5[c]) / cfg_gem5[c] * 100
    print(f"{c:>6} {cfg_surr[c]:>10.4f} {cfg_gem5[c]:>10.4f} {e:>+6.2f}%")