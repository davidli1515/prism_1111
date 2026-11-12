"""
MSCS Jacobian Attribution Analysis

Computes ∂MSCS/∂θ via backprop through ML model, aggregates by bottleneck type,
and generates a stacked bar chart showing each parameter's contribution
to FP, BP, SP pressures — the multi-stage equivalent of Shapley attribution.

Also computes scalar ∂CPI/∂θ for comparison, showing that scalar gradient
misses bottleneck-type information.

Usage:
    python analyze_attribution.py --benchmark 605.mcf_s
    python analyze_attribution.py --benchmark 623.xalancbmk_s
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch

np.seterr(divide='ignore', invalid='ignore')

EXP1_DIR = "/scratch/gem5-TAO/micro_exp/exp1"
DSE_DIR  = "/scratch/gem5-TAO/micro_exp/dse_exp"
sys.path.insert(0, EXP1_DIR)
sys.path.insert(0, DSE_DIR)

from config import GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES
from model import GEMModel, ConcordeModel
from dataset import load_all_benchmarks, create_exp1_dataloaders, FeatureNormalizer
from dse_infer import DseEngine
from config_dse import (
    ALL_PARAMS, GRADIENT_PARAMS, PARAM_TO_GD_INDEX,
    DEFAULT_CONFIG, compute_w_min, get_param_values,
    GEM_DATA_ARCH_START, GEM_DATA_ARCH_END,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Model dims (must match trained weights) ----
GEM_INPUT_DIM        = 4681
CONCORDE_INPUT_DIM   = 3873
GEM_HIDDEN_DIMS      = [512, 256]
CONCORDE_HIDDEN_DIMS = [512, 256]
DROPOUT              = 0.1

# ---- Paths ----
GEM_WEIGHT_PATH      = "/scratch/gem5-TAO/micro_exp/exp1/results/exp1_v2/exp1_512_256/gem_full.pt"
CONCORDE_WEIGHT_PATH = "/scratch/gem5-TAO/micro_exp/exp1/results/exp1_v2/exp1_512_256/concorde.pt"
RESULTS_DIR          = os.path.join(DSE_DIR, "results")
NORMALIZER_CACHE     = os.path.join(DSE_DIR, "normalizer_cache")

PRECOMPUTE_DIRS = {
    "605.mcf_s":       os.path.join(DSE_DIR, "precompute/605.mcf_s"),
    "623.xalancbmk_s": os.path.join(DSE_DIR, "precompute/623.xalancbmk_s"),
}

# ---- MSCS layout ----
# 16 values: 4 stages × 4 types
# Stage: decode(0), dispatch(1), issue(2), commit(3)
# Type:  Base(0), FP(1), BP(2), SP(3)
# Index = stage * 4 + type
STAGE_NAMES = ["decode", "dispatch", "issue", "commit"]
FP_INDICES = [1, 5, 9, 13]   # FP at 4 stages
BP_INDICES = [2, 6, 10, 14]  # BP at 4 stages
SP_INDICES = [3, 7, 11, 15]  # SP at 4 stages


def get_normalizers():
    paths = [os.path.join(NORMALIZER_CACHE, f"{p}.npy")
             for p in ["gem_mean", "gem_std", "con_mean", "con_std"]]
    if all(os.path.isfile(p) for p in paths):
        gem_norm = FeatureNormalizer()
        gem_norm.mean = np.load(paths[0]); gem_norm.std = np.load(paths[1])
        con_norm = FeatureNormalizer()
        con_norm.mean = np.load(paths[2]); con_norm.std = np.load(paths[3])
        return gem_norm, con_norm
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    (_, _, gem_norm, con_norm, _) = create_exp1_dataloaders(all_data)
    os.makedirs(NORMALIZER_CACHE, exist_ok=True)
    np.save(paths[0], gem_norm.mean); np.save(paths[1], gem_norm.std)
    np.save(paths[2], con_norm.mean); np.save(paths[3], con_norm.std)
    return gem_norm, con_norm


def predict_mscs(model, normalizer, dse_engine, workload, config, n_slices=200):
    """Predict 16-dim MSCS averaged over slices."""
    all_feats = dse_engine.get_features(config, workload=workload)
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_feats), size=min(n_slices, len(all_feats)), replace=False)
    batch = np.stack([all_feats[i][:GEM_INPUT_DIM] for i in indices])
    normalized = normalizer.transform(batch)
    x = torch.from_numpy(normalized).float()
    with torch.no_grad():
        mscs = model(x)
    return mscs.mean(dim=0).cpu().numpy()


def predict_cpi_concorde(model, normalizer, dse_engine, workload, config, n_slices=200):
    """Predict scalar CPI from Concorde model."""
    all_feats = dse_engine.get_features(config, workload=workload)
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_feats), size=min(n_slices, len(all_feats)), replace=False)
    batch = np.stack([all_feats[i][:CONCORDE_INPUT_DIM] for i in indices])
    normalized = normalizer.transform(batch)
    x = torch.from_numpy(normalized).float()
    with torch.no_grad():
        cpi = model(x)
    return cpi.mean().item()



def compute_mscs_jacobian(model, normalizer, dse_engine, workload, config,
                           n_grad_slices=None):
    """
    Backprop Jacobian: ∂MSCS/∂(arch_params_23dim) directly through ML model.
    n_grad_slices: number of slices to average over. None = all slices.
    Returns: param_names, jacobian (n_params × 16), param_ranges, mscs_base
    """
    all_feats = dse_engine.get_features(config, workload=workload)
    total_slices = len(all_feats)

    a_s, a_e = GEM_DATA_ARCH_START, GEM_DATA_ARCH_END
    norm_std = normalizer.std[a_s:a_e]

    if n_grad_slices is None:
        n_use = total_slices
        indices = list(range(total_slices))
    else:
        n_use = min(n_grad_slices, total_slices)
        rng = np.random.RandomState(42)
        indices = rng.choice(total_slices, size=n_use, replace=False)

    avg_jacobian_full = np.zeros((a_e - a_s, 16))
    mscs_sum = np.zeros(16)

    for si_idx, si in enumerate(indices):
        feat = all_feats[si][:GEM_INPUT_DIM].copy()
        normalized = normalizer.transform(feat.reshape(1, -1)).flatten()
        x = torch.from_numpy(normalized).float()

        for oi in range(16):
            x_arch = x[a_s:a_e].detach().requires_grad_(True)
            x_full = torch.cat([x[:a_s].detach(), x_arch, x[a_e:].detach()])
            mscs_out = model(x_full.unsqueeze(0))
            mscs_out[0, oi].backward()
            avg_jacobian_full[:, oi] += x_arch.grad.detach().cpu().numpy()

        with torch.no_grad():
            x_full_nd = torch.from_numpy(
                normalizer.transform(all_feats[si][:GEM_INPUT_DIM].reshape(1, -1))
            ).float()
            mscs_sum += model(x_full_nd).squeeze().cpu().numpy()

    avg_jacobian_full /= n_use
    mscs_base = mscs_sum / n_use

    for j in range(16):
        avg_jacobian_full[:, j] /= (norm_std + 1e-8)

    param_names = []
    jacobian = []
    param_ranges = []
    for name in GRADIENT_PARAMS:
        if name not in PARAM_TO_GD_INDEX:
            continue
        local_idx = PARAM_TO_GD_INDEX[name] - GEM_DATA_ARCH_START
        vals = get_param_values(name)
        param_names.append(name)
        jacobian.append(avg_jacobian_full[local_idx, :])
        param_ranges.append(max(vals) - min(vals))

    return param_names, np.array(jacobian), np.array(param_ranges), mscs_base


def compute_cpi_gradient(concorde_model, con_norm, dse_engine, workload, config):
    """
    Backprop ∂CPI/∂(arch_params) through Concorde model.
    """
    all_feats = dse_engine.get_features(config, workload=workload)
    rng = np.random.RandomState(42)
    n = min(200, len(all_feats))
    indices = rng.choice(len(all_feats), size=n, replace=False)

    a_s = GEM_DATA_ARCH_START
    a_e = CONCORDE_INPUT_DIM
    norm_std = con_norm.std[a_s:a_e]
    n_arch = a_e - a_s

    n_grad_slices = min(50, n)
    avg_grad = np.zeros(n_arch)
    cpi_sum = 0.0

    for si in range(n_grad_slices):
        feat = all_feats[indices[si]][:CONCORDE_INPUT_DIM].copy()
        normalized = con_norm.transform(feat.reshape(1, -1)).flatten()
        x = torch.from_numpy(normalized).float()

        x_arch = x[a_s:a_e].detach().requires_grad_(True)
        x_full = torch.cat([x[:a_s].detach(), x_arch])
        cpi = concorde_model(x_full.unsqueeze(0))
        cpi.backward()
        avg_grad += x_arch.grad.detach().cpu().numpy()
        cpi_sum += cpi.item()

    avg_grad /= n_grad_slices
    real = avg_grad / (norm_std + 1e-8)
    cpi_base = cpi_sum / n_grad_slices

    param_names = []
    gradients = []
    param_ranges = []
    for name in GRADIENT_PARAMS:
        if name not in PARAM_TO_GD_INDEX:
            continue
        local_idx = PARAM_TO_GD_INDEX[name] - GEM_DATA_ARCH_START
        if local_idx < len(real):
            vals = get_param_values(name)
            param_names.append(name)
            gradients.append(real[local_idx])
            param_ranges.append(max(vals) - min(vals))

    return param_names, np.array(gradients), np.array(param_ranges), cpi_base




def plot_mscs_attribution(param_names, jacobian, param_ranges, mscs_base,
                           benchmark, output_dir):
    """
    Stacked bar chart: each parameter's full-range effect on FP, BP, SP.
    Effect = sum of pressure reduction × param_range across 4 stages.
    This answers: "if this param varies from min to max, how much does each
    pressure type change?" — making all parameters comparable.
    """
    n_params = len(param_names)

    # Range-normalized: |gradient| × range = full-range effect
    fp_effects = np.array([np.maximum(0, -jacobian[i, FP_INDICES]).sum() * param_ranges[i]
                           for i in range(n_params)])
    bp_effects = np.array([np.maximum(0, -jacobian[i, BP_INDICES]).sum() * param_ranges[i]
                           for i in range(n_params)])
    sp_effects = np.array([np.maximum(0, -jacobian[i, SP_INDICES]).sum() * param_ranges[i]
                           for i in range(n_params)])

    # Sort by total effect (descending)
    total_effects = fp_effects + bp_effects + sp_effects
    sort_idx = np.argsort(total_effects)[::-1]

    sorted_names = [param_names[i] for i in sort_idx]
    sorted_fp = fp_effects[sort_idx]
    sorted_bp = bp_effects[sort_idx]
    sorted_sp = sp_effects[sort_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_params)
    w = 0.6

    ax.bar(x, sorted_fp, w, label='Frontend Pressure (FP)', color='#e74c3c', alpha=0.85)
    ax.bar(x, sorted_bp, w, bottom=sorted_fp, label='Backend Pressure (BP)',
           color='#3498db', alpha=0.85)
    ax.bar(x, sorted_sp, w, bottom=sorted_fp + sorted_bp,
           label='Speculative Pressure (SP)', color='#2ecc71', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Pressure Reduction (full range)\nΣpressure reduction × param_range')
    ax.set_title(f'MSCS Bottleneck Attribution — {benchmark} (ARM N1 config)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(output_dir, f"{benchmark}_mscs_attribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    return sorted_names, sorted_fp, sorted_bp, sorted_sp


def plot_stage_profile(param_names, jacobian, param_ranges, benchmark,
                        output_dir, top_k=6):
    """
    For top-K most impactful parameters, show how their effect varies
    across 4 pipeline stages — the key multi-stage insight.
    """
    n_params = len(param_names)
    total_effects = np.array([np.maximum(0, -jacobian[i]).sum() * param_ranges[i]
                              for i in range(n_params)])
    top_idx = np.argsort(total_effects)[::-1][:top_k]

    fig, axes = plt.subplots(1, top_k, figsize=(3 * top_k, 4), sharey=True)
    if top_k == 1:
        axes = [axes]

    for ax_i, pi in enumerate(top_idx):
        name = param_names[pi]
        r = param_ranges[pi]
        fp_profile = np.maximum(0, -jacobian[pi, FP_INDICES]) * r
        bp_profile = np.maximum(0, -jacobian[pi, BP_INDICES]) * r
        sp_profile = np.maximum(0, -jacobian[pi, SP_INDICES]) * r

        x = np.arange(4)
        w = 0.25
        axes[ax_i].bar(x - w, fp_profile, w, label='FP', color='#e74c3c', alpha=0.85)
        axes[ax_i].bar(x, bp_profile, w, label='BP', color='#3498db', alpha=0.85)
        axes[ax_i].bar(x + w, sp_profile, w, label='SP', color='#2ecc71', alpha=0.85)
        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(['Dec', 'Dis', 'Iss', 'Com'], fontsize=8)
        axes[ax_i].set_title(name, fontsize=10)
        if ax_i == 0:
            axes[ax_i].set_ylabel('Full-Range Effect')
        if ax_i == top_k - 1:
            axes[ax_i].legend(fontsize=7)
        axes[ax_i].grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Stage-wise Bottleneck Profile — {benchmark} (ARM N1)', fontsize=12)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{benchmark}_stage_profile.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_scalar_vs_mscs(param_names_mscs, jacobian, param_ranges_mscs,
                         param_names_cpi, cpi_grads, param_ranges_cpi,
                         benchmark, output_dir):
    """
    Side-by-side comparison: scalar CPI gradient vs MSCS attribution.
    Both normalized by parameter range for fair comparison.
    """
    common = [n for n in param_names_mscs if n in param_names_cpi]
    mscs_idx = {n: i for i, n in enumerate(param_names_mscs)}
    cpi_idx = {n: i for i, n in enumerate(param_names_cpi)}

    n = len(common)
    fp_eff = np.array([np.maximum(0, -jacobian[mscs_idx[n], FP_INDICES]).sum() * param_ranges_mscs[mscs_idx[n]]
                       for n in common])
    bp_eff = np.array([np.maximum(0, -jacobian[mscs_idx[n], BP_INDICES]).sum() * param_ranges_mscs[mscs_idx[n]]
                       for n in common])
    sp_eff = np.array([np.maximum(0, -jacobian[mscs_idx[n], SP_INDICES]).sum() * param_ranges_mscs[mscs_idx[n]]
                       for n in common])
    total_mscs = fp_eff + bp_eff + sp_eff
    cpi_abs = np.array([abs(cpi_grads[cpi_idx[n]]) * param_ranges_cpi[cpi_idx[n]]
                        for n in common])

    # Sort by MSCS total
    sort_idx = np.argsort(total_mscs)[::-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    x = np.arange(n)
    w = 0.6

    # Top: MSCS attribution (stacked)
    ax1.bar(x, fp_eff[sort_idx], w, label='FP', color='#e74c3c', alpha=0.85)
    ax1.bar(x, bp_eff[sort_idx], w, bottom=fp_eff[sort_idx], label='BP',
            color='#3498db', alpha=0.85)
    ax1.bar(x, sp_eff[sort_idx], w, bottom=fp_eff[sort_idx] + bp_eff[sort_idx],
            label='SP', color='#2ecc71', alpha=0.85)
    ax1.set_ylabel('MSCS Full-Range Effect')
    ax1.set_title(f'MSCS vs Scalar CPI Gradient — {benchmark} (ARM N1)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Bottom: Scalar CPI gradient
    colors = ['#e74c3c' if cpi_grads[cpi_idx[common[sort_idx[i]]]] > 0 else '#3498db'
              for i in range(n)]
    ax2.bar(x, cpi_abs[sort_idx], w, color=colors, alpha=0.85)
    ax2.set_ylabel('Scalar CPI Full-Range Effect')
    ax2.set_xticks(x)
    ax2.set_xticklabels([common[i] for i in sort_idx], rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add annotation
    ax2.text(0.98, 0.95,
             'Red = increasing param worsens CPI\nBlue = increasing param improves CPI',
             transform=ax2.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    path = os.path.join(output_dir, f"{benchmark}_scalar_vs_mscs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_percentage_decomposition(param_names, jacobian, param_ranges, benchmark,
                                    output_dir):
    """
    100% stacked bar: each parameter's effect decomposed into FP/BP/SP percentage.
    Shows WHAT TYPE of bottleneck each parameter affects, regardless of magnitude.
    """
    n_params = len(param_names)

    fp_eff = np.array([np.maximum(0, -jacobian[i, FP_INDICES]).sum() * param_ranges[i]
                       for i in range(n_params)])
    bp_eff = np.array([np.maximum(0, -jacobian[i, BP_INDICES]).sum() * param_ranges[i]
                       for i in range(n_params)])
    sp_eff = np.array([np.maximum(0, -jacobian[i, SP_INDICES]).sum() * param_ranges[i]
                       for i in range(n_params)])
    total = fp_eff + bp_eff + sp_eff + 1e-10

    fp_pct = fp_eff / total * 100
    bp_pct = bp_eff / total * 100
    sp_pct = sp_eff / total * 100

    # Sort by BP percentage (backend-heavy params first)
    sort_idx = np.argsort(bp_pct)[::-1]
    sorted_names = [param_names[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_params)
    w = 0.65

    ax.bar(x, fp_pct[sort_idx], w, label='Frontend (FP)', color='#e74c3c', alpha=0.85)
    ax.bar(x, bp_pct[sort_idx], w, bottom=fp_pct[sort_idx],
           label='Backend (BP)', color='#3498db', alpha=0.85)
    ax.bar(x, sp_pct[sort_idx], w, bottom=fp_pct[sort_idx] + bp_pct[sort_idx],
           label='Speculative (SP)', color='#2ecc71', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Bottleneck Type Decomposition (%)')
    ax.set_title(f'Parameter Bottleneck Profile — {benchmark} (ARM N1)')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on the dominant type
    for i, si in enumerate(sort_idx):
        dominant_val = max(fp_pct[si], bp_pct[si], sp_pct[si])
        if dominant_val > 40:
            if bp_pct[si] == dominant_val:
                y_pos = fp_pct[si] + bp_pct[si] / 2
                ax.text(i, y_pos, f'{bp_pct[si]:.0f}%', ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold')
            elif fp_pct[si] == dominant_val:
                y_pos = fp_pct[si] / 2
                ax.text(i, y_pos, f'{fp_pct[si]:.0f}%', ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, f"{benchmark}_pct_decomposition.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_jacobian_heatmap(param_names, jacobian, param_ranges, benchmark, output_dir):
    """
    Heatmap of signed Jacobian: params × 12 MSCS components.
    Shows the full picture: which parameter affects which bottleneck at which stage.
    Values are range-normalized (gradient × param_range).
    """
    n_params = len(param_names)

    # Range-normalize
    jac_scaled = jacobian.copy()
    for i in range(n_params):
        jac_scaled[i, :] *= param_ranges[i]

    # Column labels: FP×4stages, BP×4stages, SP×4stages
    col_labels = [
        'FP\ndec', 'FP\ndisp', 'FP\niss', 'FP\ncom',
        'BP\ndec', 'BP\ndisp', 'BP\niss', 'BP\ncom',
        'SP\ndec', 'SP\ndisp', 'SP\niss', 'SP\ncom',
    ]

    # Sort params by total absolute effect (most impactful first)
    total_eff = np.abs(jac_scaled).sum(axis=1)
    sort_idx = np.argsort(total_eff)[::-1]
    sorted_names = [param_names[i] for i in sort_idx]
    jac_sorted = jac_scaled[sort_idx]

    # Symmetric colorscale
    vmax = np.abs(jac_sorted).max()

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(jac_sorted, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')

    ax.set_xticks(range(12))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_params))
    ax.set_yticklabels(sorted_names, fontsize=9)

    # Add value annotations
    for i in range(n_params):
        for j in range(12):
            val = jac_sorted[i, j]
            if abs(val) > vmax * 0.15:  # only annotate significant cells
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=6, color=color)

    # Add vertical lines separating FP/BP/SP
    ax.axvline(3.5, color='black', linewidth=1.5)
    ax.axvline(7.5, color='black', linewidth=1.5)

    # Add type labels at top
    ax.text(1.5, -1.2, 'Frontend (FP)', ha='center', fontsize=11,
            fontweight='bold', color='#e74c3c')
    ax.text(5.5, -1.2, 'Backend (BP)', ha='center', fontsize=11,
            fontweight='bold', color='#3498db')
    ax.text(9.5, -1.2, 'Speculative (SP)', ha='center', fontsize=11,
            fontweight='bold', color='#2ecc71')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('∂pressure/∂θ × param_range', fontsize=10)

    ax.set_title(f'MSCS Jacobian Heatmap — {benchmark} (ARM N1)', fontsize=13, pad=25)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{benchmark}_jacobian_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_mscs_bars(mscs, benchmark, output_dir, config_label="default"):
    """Plot MSCS as 4-stage stacked bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    stage_labels = ['Decode', 'Dispatch', 'Issue', 'Commit']
    x = np.arange(4)
    w = 0.5

    base_vals = [mscs[s * 4 + 0] for s in range(4)]
    fp_vals   = [mscs[s * 4 + 1] for s in range(4)]
    bp_vals   = [mscs[s * 4 + 2] for s in range(4)]
    sp_vals   = [mscs[s * 4 + 3] for s in range(4)]

    ax.bar(x, base_vals, w, label='Base', color='#cccccc')
    ax.bar(x, fp_vals, w, bottom=base_vals, label='FP', color='#e74c3c', alpha=0.85)
    ax.bar(x, bp_vals, w, bottom=np.array(base_vals) + np.array(fp_vals),
           label='BP', color='#3498db', alpha=0.85)
    ax.bar(x, sp_vals, w,
           bottom=np.array(base_vals) + np.array(fp_vals) + np.array(bp_vals),
           label='SP', color='#2ecc71', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_ylabel('CPI Proportion')
    ax.set_title(f'Multi-Stage CPI Stack — {benchmark} ({config_label})')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(output_dir, f"{benchmark}_mscs_bars_{config_label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MSCS Jacobian Attribution Analysis")
    parser.add_argument("--benchmark", required=True,
                        choices=list(PRECOMPUTE_DIRS.keys()))
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    benchmark = args.benchmark
    output_dir = os.path.join(args.output_dir, benchmark)
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"MSCS Jacobian Attribution Analysis — {benchmark}")
    print(f"{'='*60}")

    # ---- Load everything ----
    print(f"\nLoading models...")
    device = torch.device("cpu")

    gem_model = GEMModel(input_dim=GEM_INPUT_DIM, hidden_dims=GEM_HIDDEN_DIMS, dropout=DROPOUT)
    gem_model.load_state_dict(torch.load(GEM_WEIGHT_PATH, map_location=device))
    gem_model.eval()

    concorde_model = ConcordeModel(input_dim=CONCORDE_INPUT_DIM,
                                     hidden_dims=CONCORDE_HIDDEN_DIMS, dropout=0.0)
    concorde_model.load_state_dict(torch.load(CONCORDE_WEIGHT_PATH, map_location=device))
    concorde_model.eval()

    print(f"Loading normalizers...")
    gem_norm, con_norm = get_normalizers()

    print(f"Loading precomputed features...")
    dse_engine = DseEngine(list(PRECOMPUTE_DIRS.values()))

    # ARM Neoverse N1 configuration (real processor baseline)
    ARM_N1_CONFIG = {
        "fetchWidth": 4,
        "decodeWidth": 4,
        "dispatchWidth": 4,     # = renameWidth = issueWidth
        "commitWidth": 8,
        "wbWidth": 8,
        "numROBEntries": 128,
        "numIQEntries": 32,     # typical for N1-class
        "LQEntries": 12, "SQEntries": 18, "fetchBufferSize": 1,
        "SQEntries": 18,
        "fetchBufferSize": 1,
        "IntALU": 3,
        "IntMultDiv": 1,
        "FP_ALU": 2,
        "FP_MultDiv": 1,
        "RdWrPort": 2,
        "l1_size": 64,
        "l2_size": 1024,
        "l1d_assoc": 4,         # typical for 64KB L1
        "l1i_assoc": 4,
        "l2_assoc": 8,
        "bp_type": 0,           # BiMode (closest to TAGE in our design space)
    }

    config = ARM_N1_CONFIG
    config_label = "ARM_N1"

    # ---- MSCS prediction ----
    print(f"\nPredicting MSCS for {config_label} config...")
    mscs = predict_mscs(gem_model, gem_norm, dse_engine, benchmark, config)
    w_min = compute_w_min(config)
    cpi = 1.0 / (mscs[12] * w_min + 1e-8)

    print(f"  CPI = {cpi:.4f} (W_min={w_min})")
    print(f"  MSCS:")
    for s in range(4):
        vals = [f"{mscs[s*4+t]:.4f}" for t in range(4)]
        print(f"    {STAGE_NAMES[s]:>10s}: Base={vals[0]} FP={vals[1]} BP={vals[2]} SP={vals[3]}")
    print(f"  Penetration: FP_commit={mscs[13]:.4f}, BP_decode={mscs[2]:.4f}, "
          f"SP_commit={mscs[15]:.4f}")

    # ---- Plot MSCS bars ----
    plot_mscs_bars(mscs, benchmark, output_dir, config_label)

    # ---- Compute MSCS Jacobian (all slices) ----
    total_slices = len(dse_engine.get_features(config, workload=benchmark))

    print(f"\nComputing MSCS Jacobian — ALL {total_slices} slices (backprop)...")
    t0 = time.time()
    param_names_mscs, jacobian, param_ranges_mscs, mscs_base = compute_mscs_jacobian(
        gem_model, gem_norm, dse_engine, benchmark, config, n_grad_slices=None)
    t_all = time.time() - t0

    print(f"\nComputing MSCS Jacobian — 50 slices (backprop)...")
    t0 = time.time()
    _, jacobian_50, _, _ = compute_mscs_jacobian(
        gem_model, gem_norm, dse_engine, benchmark, config, n_grad_slices=50)
    t_50 = time.time() - t0

    # Filter out weak-signal params (total effect < 5% of max)
    total_effects = np.array([
        (np.maximum(0, -jacobian[i, FP_INDICES]).sum() +
         np.maximum(0, -jacobian[i, BP_INDICES]).sum() +
         np.maximum(0, -jacobian[i, SP_INDICES]).sum()) * param_ranges_mscs[i]
        for i in range(len(param_names_mscs))
    ])
    max_effect = total_effects.max() + 1e-10
    WEAK_THRESHOLD = 0.05  # 5% of max
    keep = [i for i in range(len(param_names_mscs))
            if total_effects[i] >= max_effect * WEAK_THRESHOLD]
    excluded = [param_names_mscs[i] for i in range(len(param_names_mscs)) if i not in keep]
    param_names_mscs = [param_names_mscs[i] for i in keep]
    jacobian = jacobian[keep]
    jacobian_50 = jacobian_50[keep]
    param_ranges_mscs = param_ranges_mscs[keep]
    if excluded:
        print(f"  Excluded weak-signal params (<5% of max): {excluded}")

    print(f"  {len(param_names_mscs)} params × 16 MSCS | "
          f"all={total_slices} slices ({t_all:.1f}s), 50 slices ({t_50:.1f}s)")

    # ---- Compare all vs 50 slices ----
    print(f"\n  All slices vs 50 slices comparison (reduction effect × range):")
    print(f"  {'Param':<20s} {'FP_all':>8s} {'FP_50':>8s} {'BP_all':>8s} {'BP_50':>8s} "
          f"{'SP_all':>8s} {'SP_50':>8s} {'Δmax':>8s}")
    print(f"  {'-'*82}")
    max_deltas = []
    for i, name in enumerate(param_names_mscs):
        r = param_ranges_mscs[i]
        fp_a = np.maximum(0, -jacobian[i, FP_INDICES]).sum() * r
        bp_a = np.maximum(0, -jacobian[i, BP_INDICES]).sum() * r
        sp_a = np.maximum(0, -jacobian[i, SP_INDICES]).sum() * r
        fp_5 = np.maximum(0, -jacobian_50[i, FP_INDICES]).sum() * r
        bp_5 = np.maximum(0, -jacobian_50[i, BP_INDICES]).sum() * r
        sp_5 = np.maximum(0, -jacobian_50[i, SP_INDICES]).sum() * r
        total_a = fp_a + bp_a + sp_a + 1e-10
        total_5 = fp_5 + bp_5 + sp_5 + 1e-10
        # Max pct-point difference in decomposition
        d_fp = abs(fp_a/total_a - fp_5/total_5) * 100
        d_bp = abs(bp_a/total_a - bp_5/total_5) * 100
        d_sp = abs(sp_a/total_a - sp_5/total_5) * 100
        d_max = max(d_fp, d_bp, d_sp)
        max_deltas.append(d_max)
        print(f"  {name:<20s} {fp_a:>8.4f} {fp_5:>8.4f} {bp_a:>8.4f} {bp_5:>8.4f} "
              f"{sp_a:>8.4f} {sp_5:>8.4f} {d_max:>7.1f}pp")
    print(f"  Average max Δ: {np.mean(max_deltas):.1f}pp | "
          f"Max Δ: {np.max(max_deltas):.1f}pp | "
          f"Speedup: {t_all/t_50:.1f}x slower for all slices")

    # Print Jacobian summary (all slices, used for plots)
    print(f"\n  Per-parameter pressure reduction — ALL slices (negative gradient × range):")
    print(f"  {'Param':<20s} {'Range':>8s} {'FP_effect':>10s} {'BP_effect':>10s} {'SP_effect':>10s} {'Total':>10s}")
    print(f"  {'-'*70}")
    for i, name in enumerate(param_names_mscs):
        r = param_ranges_mscs[i]
        fp = np.maximum(0, -jacobian[i, FP_INDICES]).sum() * r
        bp = np.maximum(0, -jacobian[i, BP_INDICES]).sum() * r
        sp = np.maximum(0, -jacobian[i, SP_INDICES]).sum() * r
        total = fp + bp + sp
        print(f"  {name:<20s} {r:>8.0f} {fp:>10.4f} {bp:>10.4f} {sp:>10.4f} {total:>10.4f}")

    # ---- Compute scalar CPI gradient ----
    print(f"\nComputing Concorde CPI gradient (backprop)...")
    t0 = time.time()
    param_names_cpi, cpi_grads, param_ranges_cpi, cpi_base = compute_cpi_gradient(
        concorde_model, con_norm, dse_engine, benchmark, config)
    print(f"  {len(param_names_cpi)} params, CPI_base={cpi_base:.4f}, {time.time()-t0:.1f}s")

    print(f"\n  Scalar CPI gradient (full-range effect):")
    print(f"  {'Param':<20s} {'Range':>8s} {'∂CPI/∂θ':>12s} {'Full-range':>12s}")
    print(f"  {'-'*54}")
    sorted_cpi = sorted(zip(param_names_cpi, cpi_grads, param_ranges_cpi),
                         key=lambda x: abs(x[1]) * x[2], reverse=True)
    for name, g, r in sorted_cpi:
        print(f"  {name:<20s} {r:>8.0f} {g:>+12.6f} {abs(g)*r:>12.4f}")

    # ---- Generate plots ----
    print(f"\nGenerating plots...")

    # 1. MSCS stacked bar attribution
    plot_mscs_attribution(param_names_mscs, jacobian, param_ranges_mscs,
                           mscs_base, benchmark, output_dir)

    # 2. Stage-wise profiles for top parameters
    plot_stage_profile(param_names_mscs, jacobian, param_ranges_mscs,
                        benchmark, output_dir)

    # 3. Scalar vs MSCS comparison
    plot_scalar_vs_mscs(param_names_mscs, jacobian, param_ranges_mscs,
                         param_names_cpi, cpi_grads, param_ranges_cpi,
                         benchmark, output_dir)

    # 4. Percentage decomposition (FP/BP/SP ratio per parameter)
    plot_percentage_decomposition(param_names_mscs, jacobian, param_ranges_mscs,
                                   benchmark, output_dir)

    # 5. Jacobian heatmap (full picture)
    plot_jacobian_heatmap(param_names_mscs, jacobian, param_ranges_mscs,
                           benchmark, output_dir)

    # ---- Save JSON ----
    data = {
        "benchmark": benchmark,
        "config": config,
        "cpi": float(cpi),
        "mscs": mscs.tolist(),
        "penetrations": {
            "FP": float(mscs[13]),
            "BP": float(mscs[2]),
            "SP": float(mscs[15]),
        },
        "param_names": param_names_mscs,
        "jacobian": jacobian.tolist(),
        "cpi_gradient": {n: {"grad": float(g), "range": float(r),
                              "full_range_effect": float(abs(g) * r)}
                         for n, g, r in zip(param_names_cpi, cpi_grads, param_ranges_cpi)},
        "attribution": {
            name: {
                "param_range": float(param_ranges_mscs[i]),
                "FP_effect": float(np.maximum(0, -jacobian[i, FP_INDICES]).sum() * param_ranges_mscs[i]),
                "BP_effect": float(np.maximum(0, -jacobian[i, BP_INDICES]).sum() * param_ranges_mscs[i]),
                "SP_effect": float(np.maximum(0, -jacobian[i, SP_INDICES]).sum() * param_ranges_mscs[i]),
            }
            for i, name in enumerate(param_names_mscs)
        },
    }
    json_path = os.path.join(output_dir, f"{benchmark}_attribution.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\n{'='*60}")
    print(f"Done! Outputs in {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()