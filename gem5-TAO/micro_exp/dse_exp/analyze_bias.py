#!/usr/bin/env python3
"""Analyze systematic bias in gem5 validation results."""
import json, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/scratch/gem5-TAO/micro_exp/dse_exp/validate_out_full/validation_results.json"

with open(path) as f:
    results = json.load(f)

valid = [r for r in results if r.get("gem5_cpi")]
s = np.array([r["surrogate_cpi"] for r in valid])
g = np.array([r["gem5_cpi"] for r in valid])

print(f"Valid points: {len(valid)}\n")

# Signed relative error: surrogate vs gem5
rel = (s - g) / g * 100
print(f"Signed relative error (surrogate - gem5)/gem5:")
print(f"  mean:   {rel.mean():+.2f}%   (systematic bias)")
print(f"  median: {np.median(rel):+.2f}%")
print(f"  std:    {rel.std():.2f}%     (spread around bias)")
print(f"  min:    {rel.min():+.2f}%")
print(f"  max:    {rel.max():+.2f}%\n")

# Absolute
print(f"MAPE (abs): {np.mean(np.abs(rel)):.2f}%\n")

# Linear fit gem5 = a*surrogate + b
a, b = np.polyfit(s, g, 1)
print(f"Linear fit  gem5 = {a:.4f} * surrogate + {b:+.4f}")
# ratio fit gem5 = k*surrogate (through origin)
k = np.sum(s * g) / np.sum(s * s)
print(f"Ratio fit   gem5 = {k:.4f} * surrogate   (surrogate underestimates by {(k-1)*100:.1f}%)\n")

# After ratio calibration, residual MAPE
s_cal = s * k
mape_cal = np.mean(np.abs(s_cal - g) / g) * 100
print(f"MAPE after ratio calibration: {mape_cal:.2f}%")

# Per-benchmark bias + per-benchmark linear fit + calibrated MAPE
print(f"\nPer-benchmark analysis:")
print(f"{'benchmark':<18} {'bias':>8} {'fit a':>8} {'fit b':>9} "
      f"{'ratio k':>8} {'MAPE':>7} {'calMAPE':>8}")
print("-" * 70)
all_cal_err = []
for bench in sorted(set(r["bench"] for r in valid)):
    bs = np.array([r["surrogate_cpi"] for r in valid if r["bench"] == bench])
    bg = np.array([r["gem5_cpi"] for r in valid if r["bench"] == bench])
    brel = (bs - bg) / bg * 100
    # linear fit gem5 = a*surrogate + b
    ba, bb = np.polyfit(bs, bg, 1)
    # ratio fit through origin
    bk = np.sum(bs * bg) / np.sum(bs * bs)
    # raw MAPE (surrogate vs gem5)
    bmape = np.mean(np.abs(bs - bg) / bg) * 100
    # calibrated: use per-benchmark linear fit
    bg_pred = ba * bs + bb
    bcal = np.mean(np.abs(bg_pred - bg) / bg) * 100
    all_cal_err.extend(list(np.abs(bg_pred - bg) / bg * 100))
    print(f"  {bench:<16} {brel.mean():>+7.2f}% {ba:>8.4f} {bb:>+9.4f} "
          f"{bk:>8.4f} {bmape:>6.2f}% {bcal:>7.2f}%")

print("-" * 70)
print(f"  Overall MAPE after per-benchmark linear calibration: "
      f"{np.mean(all_cal_err):.2f}%")