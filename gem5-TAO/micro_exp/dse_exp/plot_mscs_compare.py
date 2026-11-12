"""
Cross-benchmark MSCS comparison — absolute CPI stacked bars.
Like Concorde Figure 16, but with MSCS (Base/FP/BP/SP × 4 stages).

Usage:
    python plot_mscs_compare.py
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

ARM_N1_CONFIG = {
    "fetchWidth": 4, "decodeWidth": 4, "dispatchWidth": 4,
    "commitWidth": 8, "wbWidth": 8,
    "numROBEntries": 128, "numIQEntries": 32,
    "LQEntries": 16, "SQEntries": 18, "fetchBufferSize": 1,
    "IntALU": 3, "IntMultDiv": 1, "FP_ALU": 2, "FP_MultDiv": 1, "RdWrPort": 2,
    "l1_size": 64, "l2_size": 1024,
    "l1d_assoc": 4, "l1i_assoc": 4, "l2_assoc": 8, "bp_type": 0,
}

STAGE_NAMES = ['Decode', 'Dispatch', 'Issue', 'Commit']
BENCH_LABELS = {"605.mcf_s": "mcf", "623.xalancbmk_s": "xalancbmk"}


def predict_mscs(model, normalizer, dse_engine, workload, config, n=200):
    all_feats = dse_engine.get_features(config, workload=workload)
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_feats), size=min(n, len(all_feats)), replace=False)
    batch = np.stack([all_feats[i][:GEM_INPUT_DIM] for i in indices])
    x = torch.from_numpy(normalizer.transform(batch)).float()
    with torch.no_grad():
        return model(x).mean(dim=0).cpu().numpy()


def main():
    # Load model
    model = GEMModel(input_dim=GEM_INPUT_DIM, hidden_dims=GEM_HIDDEN_DIMS, dropout=0.1)
    model.load_state_dict(torch.load(GEM_WEIGHT_PATH, map_location="cpu"))
    model.eval()

    # Load normalizer
    gem_norm = FeatureNormalizer()
    gem_norm.mean = np.load(os.path.join(NORMALIZER_CACHE, "gem_mean.npy"))
    gem_norm.std  = np.load(os.path.join(NORMALIZER_CACHE, "gem_std.npy"))

    # Load engine
    dse_engine = DseEngine(list(PRECOMPUTE_DIRS.values()))

    config = ARM_N1_CONFIG
    w_min = compute_w_min(config)

    # Predict for both benchmarks
    results = {}
    for bench in PRECOMPUTE_DIRS:
        mscs = predict_mscs(model, gem_norm, dse_engine, bench, config)
        cpi = 1.0 / (mscs[12] * w_min + 1e-8)
        results[bench] = {"mscs": mscs, "cpi": cpi}
        print(f"{bench}: CPI={cpi:.4f}")
        for s in range(4):
            vals = mscs[s*4:(s+1)*4]
            print(f"  {STAGE_NAMES[s]}: Base={vals[0]:.3f} FP={vals[1]:.3f} "
                  f"BP={vals[2]:.3f} SP={vals[3]:.3f}")

    # ---- Plot 1: Side-by-side absolute CPI (Figure 16 style) ----
    fig, ax = plt.subplots(figsize=(12, 6))

    n_stages = 4
    n_bench = len(results)
    group_width = 0.35
    gap = 0.15

    colors = {'Base': '#999999', 'FP': '#e74c3c', 'BP': '#3498db', 'SP': '#2ecc71'}

    for bi, (bench, data) in enumerate(results.items()):
        mscs = data["mscs"]
        cpi = data["cpi"]
        label = BENCH_LABELS.get(bench, bench)

        for si in range(n_stages):
            x_pos = si * (n_bench * group_width + gap) + bi * group_width

            base = mscs[si * 4 + 0] * cpi
            fp   = mscs[si * 4 + 1] * cpi
            bp   = mscs[si * 4 + 2] * cpi
            sp   = mscs[si * 4 + 3] * cpi

            # Only add labels on first stage
            ax.bar(x_pos, base, group_width, color=colors['Base'],
                   edgecolor='white', linewidth=0.5,
                   label='Base' if si == 0 and bi == 0 else '')
            ax.bar(x_pos, fp, group_width, bottom=base, color=colors['FP'],
                   edgecolor='white', linewidth=0.5,
                   label='FP' if si == 0 and bi == 0 else '')
            ax.bar(x_pos, bp, group_width, bottom=base + fp, color=colors['BP'],
                   edgecolor='white', linewidth=0.5,
                   label='BP' if si == 0 and bi == 0 else '')
            ax.bar(x_pos, sp, group_width, bottom=base + fp + bp, color=colors['SP'],
                   edgecolor='white', linewidth=0.5,
                   label='SP' if si == 0 and bi == 0 else '')

            # Benchmark label at bottom
            if si == 0:
                ax.text(x_pos, -0.15, label, ha='center', va='top', fontsize=8,
                        fontweight='bold')

    # X-axis: stage labels centered on each group
    stage_centers = [si * (n_bench * group_width + gap) + (n_bench - 1) * group_width / 2
                     for si in range(n_stages)]
    ax.set_xticks(stage_centers)
    ax.set_xticklabels(STAGE_NAMES, fontsize=11)

    # Add benchmark names below stage labels
    for si in range(n_stages):
        for bi, bench in enumerate(results):
            x_pos = si * (n_bench * group_width + gap) + bi * group_width
            label = BENCH_LABELS.get(bench, bench)
            ax.text(x_pos, -0.08, label, ha='center', va='top', fontsize=7,
                    color='#555', transform=ax.get_xaxis_transform())

    ax.set_ylabel('CPI', fontsize=12)
    ax.set_title('Multi-Stage CPI Stack — Absolute CPI Decomposition (ARM N1 config)',
                 fontsize=13)
    ax.legend(loc='upper left', ncol=4, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "mscs_compare_absolute.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ---- Plot 2: Both benchmarks, all stages in one row (Concorde Fig16 style) ----
    fig, ax = plt.subplots(figsize=(10, 5))

    x_positions = []
    x_labels = []
    x_i = 0
    bar_w = 0.6

    for bench in results:
        data = results[bench]
        mscs = data["mscs"]
        cpi = data["cpi"]
        label = BENCH_LABELS.get(bench, bench)

        for si in range(n_stages):
            base = mscs[si * 4 + 0] * cpi
            fp   = mscs[si * 4 + 1] * cpi
            bp   = mscs[si * 4 + 2] * cpi
            sp   = mscs[si * 4 + 3] * cpi

            ax.bar(x_i, base, bar_w, color=colors['Base'], edgecolor='white', linewidth=0.5,
                   label='Base' if x_i == 0 else '')
            ax.bar(x_i, fp, bar_w, bottom=base, color=colors['FP'],
                   edgecolor='white', linewidth=0.5,
                   label='FP' if x_i == 0 else '')
            ax.bar(x_i, bp, bar_w, bottom=base + fp, color=colors['BP'],
                   edgecolor='white', linewidth=0.5,
                   label='BP' if x_i == 0 else '')
            ax.bar(x_i, sp, bar_w, bottom=base + fp + bp, color=colors['SP'],
                   edgecolor='white', linewidth=0.5,
                   label='SP' if x_i == 0 else '')

            x_positions.append(x_i)
            x_labels.append(f"{STAGE_NAMES[si][:3]}")
            x_i += 1
        x_i += 0.8  # gap between benchmarks

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=9)

    # Add benchmark group labels
    bench_list = list(results.keys())
    mid0 = 1.5  # center of first 4 bars
    mid1 = 4 + 0.8 + 1.5  # center of second 4 bars
    ax.text(mid0, -0.12, BENCH_LABELS[bench_list[0]], ha='center', fontsize=11,
            fontweight='bold', transform=ax.get_xaxis_transform())
    ax.text(mid1, -0.12, BENCH_LABELS[bench_list[1]], ha='center', fontsize=11,
            fontweight='bold', transform=ax.get_xaxis_transform())

    ax.set_ylabel('CPI', fontsize=12)
    ax.set_title('Multi-Stage CPI Stack Comparison (ARM N1 config)', fontsize=13)
    ax.legend(loc='upper right', ncol=4, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "mscs_compare_fig16.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
