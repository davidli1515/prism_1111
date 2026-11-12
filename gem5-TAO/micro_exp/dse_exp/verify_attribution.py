"""
Attribution Verification: predict MSCS before/after parameter changes.
Validates that the bottleneck type identified by attribution is indeed
reduced when we adjust the corresponding parameter.

Usage:
    python verify_attribution.py
"""

import os
import sys
import numpy as np
import torch

np.seterr(divide='ignore', invalid='ignore')

EXP1_DIR = "/scratch/gem5-TAO/micro_exp/exp1"
DSE_DIR  = "/scratch/gem5-TAO/micro_exp/dse_exp"
sys.path.insert(0, EXP1_DIR)
sys.path.insert(0, DSE_DIR)

from model import GEMModel
from dataset import FeatureNormalizer
from dse_infer import DseEngine
from config_dse import compute_w_min

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GEM_INPUT_DIM   = 4681
GEM_HIDDEN_DIMS = [512, 256]
GEM_WEIGHT_PATH = "/scratch/gem5-TAO/micro_exp/exp1/results/exp1_v2/exp1_512_256/gem_full.pt"
NORMALIZER_CACHE = os.path.join(DSE_DIR, "normalizer_cache")
OUTPUT_DIR = os.path.join(DSE_DIR, "results")

PRECOMPUTE_DIRS = {
    "605.mcf_s":       os.path.join(DSE_DIR, "precompute/605.mcf_s"),
    "623.xalancbmk_s": os.path.join(DSE_DIR, "precompute/623.xalancbmk_s"),
}

STAGE_NAMES = ['Decode', 'Dispatch', 'Issue', 'Commit']

ARM_N1_CONFIG = {
    "fetchWidth": 4, "decodeWidth": 4, "dispatchWidth": 4,
    "commitWidth": 8, "wbWidth": 8,
    "numROBEntries": 128, "numIQEntries": 32,
    "LQEntries": 12, "SQEntries": 18, "fetchBufferSize": 1,
    "IntALU": 3, "IntMultDiv": 1, "FP_ALU": 2, "FP_MultDiv": 1, "RdWrPort": 2,
    "l1_size": 64, "l2_size": 1024,
    "l1d_assoc": 4, "l1i_assoc": 4, "l2_assoc": 8, "bp_type": 0,
}

# ---- Experiments based on attribution analysis ----
# Each: (benchmark, param_name, old_val, new_val, expected_bottleneck_type, rationale)
EXPERIMENTS = [
    # mcf: LQ is top-1 BP (90%) → increase should reduce BP
    ("605.mcf_s", "LQEntries", 12, 48,
     "BP", "mcf top-1 BP param (90%), memory-bound workload"),
    # mcf: IntALU is mainly SP (60%) → increase should reduce SP, not BP
    ("605.mcf_s", "IntALU", 3, 6,
     "SP", "mcf mainly SP param (60%), not compute-bound"),
    # mcf: numROBEntries is top-2 BP (84%) → increase should reduce BP
    ("605.mcf_s", "numROBEntries", 128, 256,
     "BP", "mcf top-2 BP param (84%), more MLP"),
    # xalan: IntALU is top-2 BP (91%) → increase should reduce BP
    ("623.xalancbmk_s", "IntALU", 3, 6,
     "BP", "xalan top-2 BP param (91%), integer-heavy workload"),
    # xalan: LQEntries is top-1 BP (94%) → increase should reduce BP
    ("623.xalancbmk_s", "LQEntries", 12, 48,
     "BP", "xalan top-1 BP param (94%)"),
    # xalan: fetchWidth should mainly reduce FP (55%)
    ("623.xalancbmk_s", "fetchWidth", 4, 8,
     "FP", "xalan FP param (55%), branch-intensive frontend"),
    # mcf: numIQEntries is 81% FP → increase should reduce FP
    ("605.mcf_s", "numIQEntries", 32, 64,
     "FP", "mcf FP param (81%), IQ buffering helps frontend"),
]


def predict_mscs(model, normalizer, dse_engine, workload, config):
    all_feats = dse_engine.get_features(config, workload=workload)
    batch = np.stack([f[:GEM_INPUT_DIM] for f in all_feats])
    x = torch.from_numpy(normalizer.transform(batch)).float()
    with torch.no_grad():
        return model(x).mean(dim=0).cpu().numpy()


def mscs_to_cpi(mscs, config):
    w_min = compute_w_min(config)
    return 1.0 / (mscs[12] * w_min + 1e-8)


def print_mscs(label, mscs, cpi):
    print(f"  {label}: CPI={cpi:.4f}")
    for s in range(4):
        b, f, bp, sp = mscs[s*4], mscs[s*4+1], mscs[s*4+2], mscs[s*4+3]
        print(f"    {STAGE_NAMES[s]:>10s}: Base={b:.4f} FP={f:.4f} BP={bp:.4f} SP={sp:.4f}")


def compute_delta(mscs_before, mscs_after):
    """Compute change in FP/BP/SP averaged across stages."""
    FP_IDX = [1, 5, 9, 13]
    BP_IDX = [2, 6, 10, 14]
    SP_IDX = [3, 7, 11, 15]
    d_fp = np.mean([mscs_after[i] - mscs_before[i] for i in FP_IDX])
    d_bp = np.mean([mscs_after[i] - mscs_before[i] for i in BP_IDX])
    d_sp = np.mean([mscs_after[i] - mscs_before[i] for i in SP_IDX])
    return d_fp, d_bp, d_sp


def plot_before_after(results, output_path):
    """Bar chart showing ΔCPI and which bottleneck changed most."""
    n = len(results)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[1, 1.2])

    labels = [f"{r['bench_short']}\n{r['param']} {r['old']}→{r['new']}" for r in results]
    x = np.arange(n)
    w = 0.6

    # Top: CPI change
    dcpi = [r['cpi_after'] - r['cpi_before'] for r in results]
    colors_cpi = ['#2ecc71' if d < 0 else '#e74c3c' for d in dcpi]
    ax1.bar(x, dcpi, w, color=colors_cpi, alpha=0.85, edgecolor='white')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel('ΔCPI')
    ax1.set_title('Attribution-Guided Parameter Changes — CPI Impact')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    for i, d in enumerate(dcpi):
        ax1.text(i, d + (0.01 if d >= 0 else -0.03), f'{d:+.3f}', ha='center',
                 fontsize=8, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Bottom: FP/BP/SP deltas (stacked, showing which bottleneck changed)
    dfp = [r['d_fp'] for r in results]
    dbp = [r['d_bp'] for r in results]
    dsp = [r['d_sp'] for r in results]

    ax2.bar(x - 0.22, dfp, 0.2, label='ΔFP', color='#e74c3c', alpha=0.85)
    ax2.bar(x, dbp, 0.2, label='ΔBP', color='#3498db', alpha=0.85)
    ax2.bar(x + 0.22, dsp, 0.2, label='ΔSP', color='#2ecc71', alpha=0.85)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('Δ Pressure (avg across stages)')
    ax2.set_title('Bottleneck Decomposition of Changes')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.legend(loc='upper right', ncol=3)
    ax2.grid(True, alpha=0.3, axis='y')

    # Mark expected bottleneck with star
    for i, r in enumerate(results):
        expected = r['expected']
        offset = {"FP": -0.22, "BP": 0, "SP": 0.22}[expected]
        val = {"FP": dfp[i], "BP": dbp[i], "SP": dsp[i]}[expected]
        ax2.annotate('★', (x[i] + offset, val),
                     fontsize=12, ha='center', va='bottom' if val < 0 else 'top',
                     color='gold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def main():
    # Load model
    model = GEMModel(input_dim=GEM_INPUT_DIM, hidden_dims=GEM_HIDDEN_DIMS, dropout=0.1)
    model.load_state_dict(torch.load(GEM_WEIGHT_PATH, map_location="cpu"))
    model.eval()

    gem_norm = FeatureNormalizer()
    gem_norm.mean = np.load(os.path.join(NORMALIZER_CACHE, "gem_mean.npy"))
    gem_norm.std  = np.load(os.path.join(NORMALIZER_CACHE, "gem_std.npy"))

    dse_engine = DseEngine(list(PRECOMPUTE_DIRS.values()))

    print("=" * 70)
    print("Attribution Verification: MSCS before/after parameter changes")
    print("=" * 70)

    results = []
    bench_short = {"605.mcf_s": "mcf", "623.xalancbmk_s": "xalan"}

    for bench, param, old_val, new_val, expected, rationale in EXPERIMENTS:
        print(f"\n{'─'*60}")
        print(f"  {bench_short[bench]}: {param} {old_val} → {new_val}")
        print(f"  Rationale: {rationale}")
        print(f"  Expected: primarily reduces {expected}")

        # Before
        config_before = ARM_N1_CONFIG.copy()
        mscs_before = predict_mscs(model, gem_norm, dse_engine, bench, config_before)
        cpi_before = mscs_to_cpi(mscs_before, config_before)

        # After
        config_after = ARM_N1_CONFIG.copy()
        config_after[param] = new_val
        mscs_after = predict_mscs(model, gem_norm, dse_engine, bench, config_after)
        cpi_after = mscs_to_cpi(mscs_after, config_after)

        d_fp, d_bp, d_sp = compute_delta(mscs_before, mscs_after)

        print_mscs("Before", mscs_before, cpi_before)
        print_mscs("After ", mscs_after, cpi_after)
        print(f"  ΔCPI = {cpi_after - cpi_before:+.4f}")
        print(f"  ΔFP = {d_fp:+.4f}  ΔBP = {d_bp:+.4f}  ΔSP = {d_sp:+.4f}")

        # Which bottleneck changed most (in reduction)?
        deltas = {"FP": d_fp, "BP": d_bp, "SP": d_sp}
        most_reduced = min(deltas, key=deltas.get)
        matched = "✓" if most_reduced == expected else "✗"
        print(f"  Most reduced: {most_reduced} ({deltas[most_reduced]:+.4f}) "
              f"| Expected: {expected} {matched}")

        results.append({
            "bench": bench, "bench_short": bench_short[bench],
            "param": param, "old": old_val, "new": new_val,
            "expected": expected, "rationale": rationale,
            "cpi_before": cpi_before, "cpi_after": cpi_after,
            "mscs_before": mscs_before.tolist(), "mscs_after": mscs_after.tolist(),
            "d_fp": d_fp, "d_bp": d_bp, "d_sp": d_sp,
            "most_reduced": most_reduced, "matched": most_reduced == expected,
        })

    # Summary
    n_match = sum(1 for r in results if r['matched'])
    print(f"\n{'='*60}")
    print(f"Summary: {n_match}/{len(results)} experiments matched attribution prediction")
    print(f"{'='*60}")
    for r in results:
        m = "✓" if r['matched'] else "✗"
        print(f"  {m} {r['bench_short']:>5s} {r['param']:<16s} {r['old']}→{r['new']:>3d} | "
              f"ΔCPI={r['cpi_after']-r['cpi_before']:+.3f} | "
              f"expected={r['expected']} actual={r['most_reduced']}")

    # Plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_before_after(results, os.path.join(OUTPUT_DIR, "attribution_verification.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()