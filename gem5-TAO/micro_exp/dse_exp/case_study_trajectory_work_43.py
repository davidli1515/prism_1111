"""
Case Study: Scalar CPI Gradient vs MSCS Dominant-Type Reduction

Two trajectories from ARM N1 baseline:
  Blue:  Each step picks argmax ΔCPI (scalar gradient — any bottleneck type)
  Green: Each step picks argmax dominant-type reduction (MSCS-guided)

Shows that MSCS avoids wasting steps on parameters that don't address
the dominant bottleneck, leading to faster CPI convergence.

Usage:
    python case_study_trajectory.py --benchmark 605.mcf_s
"""

import argparse
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
from config_dse import (
    GRADIENT_PARAMS, PARAM_TO_GD_INDEX, ALL_PARAMS,
    compute_w_min, get_param_values,
    GEM_DATA_ARCH_START, GEM_DATA_ARCH_END,
)

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

FP_INDICES = [1, 5, 9, 13]
BP_INDICES = [2, 6, 10, 14]
SP_INDICES = [3, 7, 11, 15]
BOTTLENECK_INDICES = {"FP": FP_INDICES, "BP": BP_INDICES, "SP": SP_INDICES}

ARM_N1_CONFIG = {
    "fetchWidth": 4, "decodeWidth": 4, "dispatchWidth": 4,
    "commitWidth": 8, "wbWidth": 8,
    "numROBEntries": 128, "numIQEntries": 32,
    "LQEntries": 12, "SQEntries": 18, "fetchBufferSize": 1,
    "IntALU": 3, "IntMultDiv": 1, "FP_ALU": 2, "FP_MultDiv": 1, "RdWrPort": 2,
    "l1_size": 64, "l2_size": 1024,
    "l1d_assoc": 4, "l1i_assoc": 4, "l2_assoc": 8, "bp_type": 0,
}


def predict_cpi(model, normalizer, dse_engine, workload, config):
    all_feats = dse_engine.get_features(config, workload=workload)
    batch = np.stack([f[:GEM_INPUT_DIM] for f in all_feats])
    x = torch.from_numpy(normalizer.transform(batch)).float()
    with torch.no_grad():
        mscs = model(x)
        w_min = float(compute_w_min(config))
        cpis = 1.0 / (mscs[:, 12] * w_min + 1e-8)
        return cpis.mean().item()


def predict_mscs(model, normalizer, dse_engine, workload, config):
    all_feats = dse_engine.get_features(config, workload=workload)
    batch = np.stack([f[:GEM_INPUT_DIM] for f in all_feats])
    x = torch.from_numpy(normalizer.transform(batch)).float()
    with torch.no_grad():
        return model(x).mean(dim=0).cpu().numpy()


def compute_jacobian(model, normalizer, dse_engine, workload, config, n_slices=50):
    """Backprop Jacobian over n_slices random slices."""
    all_feats = dse_engine.get_features(config, workload=workload)
    a_s, a_e = GEM_DATA_ARCH_START, GEM_DATA_ARCH_END
    norm_std = normalizer.std[a_s:a_e]

    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_feats), size=min(n_slices, len(all_feats)), replace=False)

    avg_jac = np.zeros((a_e - a_s, 16))
    for si in indices:
        feat = all_feats[si][:GEM_INPUT_DIM].copy()
        normalized = normalizer.transform(feat.reshape(1, -1)).flatten()
        x = torch.from_numpy(normalized).float()
        for oi in range(16):
            x_arch = x[a_s:a_e].detach().requires_grad_(True)
            x_full = torch.cat([x[:a_s].detach(), x_arch, x[a_e:].detach()])
            out = model(x_full.unsqueeze(0))
            out[0, oi].backward()
            avg_jac[:, oi] += x_arch.grad.detach().cpu().numpy()
    avg_jac /= len(indices)
    for j in range(16):
        avg_jac[:, j] /= (norm_std + 1e-8)

    result = {}
    for name in GRADIENT_PARAMS:
        if name not in PARAM_TO_GD_INDEX:
            continue
        local_idx = PARAM_TO_GD_INDEX[name] - GEM_DATA_ARCH_START
        result[name] = avg_jac[local_idx, :]
    return result


PARAM_STRIDE = {
    "fetchWidth": 1, "decodeWidth": 1, "dispatchWidth": 1, "commitWidth": 1,
    "numROBEntries": 16, "numIQEntries": 8, "LQEntries": 8, "SQEntries": 8,
    "IntALU": 1, "IntMultDiv": 1, "FP_ALU": 1, "FP_MultDiv": 1, "RdWrPort": 1,
}


def increase_param(config, name):
    """Increase param by 1 stride. Returns (new_config, new_val) or (None, None)."""
    c = config.copy()
    vals = get_param_values(name)
    current = c[name]
    if isinstance(ALL_PARAMS.get(name), list):
        idx = vals.index(current) if current in vals else 0
        if idx >= len(vals) - 1:
            return None, None
        c[name] = vals[idx + 1]
    else:
        stride = PARAM_STRIDE.get(name, 1)
        hi = max(vals)
        new_val = min(hi, current + stride)
        if new_val == current:
            return None, None
        c[name] = new_val
    return c, c[name]


def run_scalar_trajectory(model, normalizer, dse_engine, workload, config, n_steps=8):
    """Each step: pick param with largest ΔCPI (finite diff)."""
    trajectory = []
    cpi = predict_cpi(model, normalizer, dse_engine, workload, config)
    trajectory.append({"step": 0, "cpi": cpi, "param": "baseline", "delta_cpi": 0,
                        "config": config.copy()})

    print(f"\n  Scalar CPI Gradient Trajectory:")
    print(f"    Step 0: CPI={cpi:.4f} (baseline)")

    for step in range(1, n_steps + 1):
        best_param, best_dcpi, best_config = None, 0, None
        best_old, best_new = None, None

        for name in GRADIENT_PARAMS:
            if name not in PARAM_TO_GD_INDEX:
                continue
            c_new, new_val = increase_param(config, name)
            if c_new is None:
                continue
            cpi_new = predict_cpi(model, normalizer, dse_engine, workload, c_new)
            dcpi = cpi - cpi_new  # positive = improvement
            if dcpi > best_dcpi:
                best_dcpi = dcpi
                best_param = name
                best_config = c_new
                best_old = config[name]
                best_new = new_val

        if best_param is None:
            break

        config = best_config
        cpi = cpi - best_dcpi
        trajectory.append({
            "step": step, "cpi": cpi, "param": best_param,
            "delta_cpi": best_dcpi, "config": config.copy(),
            "old_val": best_old, "new_val": best_new,
        })
        print(f"    Step {step}: +{best_param} ({best_old}→{best_new}) "
              f"→ CPI={cpi:.4f} (ΔCPI={best_dcpi:+.4f})")

    return trajectory


def run_mscs_trajectory(model, normalizer, dse_engine, workload, config, n_steps=8):
    """Each step: find dominant type, pick param with largest reduction of that type."""
    trajectory = []
    cpi = predict_cpi(model, normalizer, dse_engine, workload, config)
    trajectory.append({"step": 0, "cpi": cpi, "param": "baseline", "delta_cpi": 0,
                        "dominant": "-", "reduction": 0, "config": config.copy()})

    print(f"\n  MSCS Dominant-Type Trajectory:")
    print(f"    Step 0: CPI={cpi:.4f} (baseline)")

    for step in range(1, n_steps + 1):
        # Find dominant bottleneck
        mscs = predict_mscs(model, normalizer, dse_engine, workload, config)
        fp_avg = np.mean([mscs[i] for i in FP_INDICES])
        bp_avg = np.mean([mscs[i] for i in BP_INDICES])
        sp_avg = np.mean([mscs[i] for i in SP_INDICES])
        pressures = {"FP": fp_avg, "BP": bp_avg, "SP": sp_avg}
        dominant = max(pressures, key=pressures.get)
        dom_indices = BOTTLENECK_INDICES[dominant]

        # Compute Jacobian
        jacobian = compute_jacobian(model, normalizer, dse_engine, workload, config)

        # Find best param for dominant type reduction
        best_param, best_red, best_config, best_dcpi = None, -1, None, 0
        best_old, best_new = None, None
        for name, jac_row in jacobian.items():
            # Compute stride for this param
            if isinstance(ALL_PARAMS.get(name), list):
                vals = get_param_values(name)
                current = config[name]
                idx = vals.index(current) if current in vals else 0
                if idx >= len(vals) - 1:
                    continue
                stride = vals[idx + 1] - current
            else:
                stride = PARAM_STRIDE.get(name, 1)

            reduction = np.maximum(0, -jac_row[dom_indices]).sum() * stride

            c_new, new_val = increase_param(config, name)
            if c_new is None:
                continue

            if reduction > best_red:
                best_red = reduction
                best_param = name
                best_config = c_new
                best_old = config[name]
                best_new = new_val

        if best_param is None:
            break

        cpi_new = predict_cpi(model, normalizer, dse_engine, workload, best_config)
        dcpi = cpi - cpi_new
        config = best_config
        cpi = cpi_new

        trajectory.append({
            "step": step, "cpi": cpi, "param": best_param,
            "delta_cpi": dcpi, "dominant": dominant,
            "reduction": best_red, "config": config.copy(),
            "old_val": best_old, "new_val": best_new,
        })
        print(f"    Step {step}: dom={dominant}, +{best_param} ({best_old}→{best_new}, "
              f"red={best_red:.4f}) → CPI={cpi:.4f} (ΔCPI={dcpi:+.4f})")

    return trajectory


def plot_trajectories(scalar_traj, mscs_traj, benchmark, output_dir):
    """Side-by-side CPI trajectory comparison — clean style."""
    fig, ax = plt.subplots(figsize=(15, 7))

    s_steps = [t["step"] for t in scalar_traj]
    s_cpis = [t["cpi"] for t in scalar_traj]
    m_steps = [t["step"] for t in mscs_traj]
    m_cpis = [t["cpi"] for t in mscs_traj]

    ax.plot(s_steps, s_cpis, 'o-', color='#3498db', linewidth=2.5,
            markersize=10, label='Scalar ∂CPI/∂θ', zorder=4)
    ax.plot(m_steps, m_cpis, 'D-', color='#2ecc71', linewidth=2.5,
            markersize=10, label='MSCS dominant reduction', zorder=4)

    # Annotate scalar (above line, blue)
    for t in scalar_traj[1:]:
        val_str = f"{t.get('old_val','')}→{t.get('new_val','')}"
        label = f"+{t['param']}({val_str})  ΔCPI={t['delta_cpi']:+.4f}"
        ax.annotate(label, (t["step"], t["cpi"]),
                    textcoords="offset points", xytext=(0, 14),
                    fontsize=8.5, color='#2471a3', ha='center', va='bottom')

    # Annotate MSCS (below line, green)
    for t in mscs_traj[1:]:
        val_str = f"{t.get('old_val','')}→{t.get('new_val','')}"
        label = f"+{t['param']}({val_str}) [{t['dominant']}]  ΔCPI={t['delta_cpi']:+.4f}"
        ax.annotate(label, (t["step"], t["cpi"]),
                    textcoords="offset points", xytext=(0, -14),
                    fontsize=8.5, color='#1e8449', ha='center', va='top')

    ax.set_xlabel('Step', fontsize=13)
    ax.set_ylabel('CPI', fontsize=13)
    ax.set_title(f'Scalar CPI Gradient vs MSCS Dominant-Type Reduction — {benchmark}',
                 fontsize=14, pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(max(len(s_steps), len(m_steps))))
    ax.tick_params(labelsize=11)

    # Extra y margin for annotations
    ymin, ymax = ax.get_ylim()
    margin = (ymax - ymin) * 0.15
    ax.set_ylim(ymin - margin, ymax + margin)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{benchmark}_trajectory_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="605.mcf_s",
                        choices=list(PRECOMPUTE_DIRS.keys()))
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    benchmark = args.benchmark
    bname = {"605.mcf_s": "mcf", "623.xalancbmk_s": "xalan"}[benchmark]
    output_dir = os.path.join(args.output_dir, benchmark)
    os.makedirs(output_dir, exist_ok=True)

    # Load
    model = GEMModel(input_dim=GEM_INPUT_DIM, hidden_dims=GEM_HIDDEN_DIMS, dropout=0.1)
    model.load_state_dict(torch.load(GEM_WEIGHT_PATH, map_location="cpu"))
    model.eval()

    gem_norm = FeatureNormalizer()
    gem_norm.mean = np.load(os.path.join(NORMALIZER_CACHE, "gem_mean.npy"))
    gem_norm.std  = np.load(os.path.join(NORMALIZER_CACHE, "gem_std.npy"))

    dse_engine = DseEngine(list(PRECOMPUTE_DIRS.values()))

    print(f"\n{'='*60}")
    print(f"Case Study: Trajectory Comparison — {benchmark}")
    print(f"{'='*60}")

    # Run both trajectories from same baseline
    scalar_traj = run_scalar_trajectory(
        model, gem_norm, dse_engine, benchmark, ARM_N1_CONFIG.copy(), args.n_steps)

    mscs_traj = run_mscs_trajectory(
        model, gem_norm, dse_engine, benchmark, ARM_N1_CONFIG.copy(), args.n_steps)

    # Summary
    s_final = scalar_traj[-1]["cpi"]
    m_final = mscs_traj[-1]["cpi"]
    s_base = scalar_traj[0]["cpi"]

    print(f"\n  {'='*50}")
    print(f"  Summary after {args.n_steps} steps:")
    print(f"    Baseline:         CPI = {s_base:.4f}")
    print(f"    Scalar gradient:  CPI = {s_final:.4f} ({(s_final-s_base)/s_base*100:+.1f}%)")
    print(f"    MSCS reduction:   CPI = {m_final:.4f} ({(m_final-s_base)/s_base*100:+.1f}%)")

    divergent = sum(1 for i in range(1, min(len(scalar_traj), len(mscs_traj)))
                    if scalar_traj[i]["param"] != mscs_traj[i]["param"])
    print(f"    Divergent steps:  {divergent}/{args.n_steps}")
    print(f"  {'='*50}")

    # Plot
    plot_trajectories(scalar_traj, mscs_traj, bname, output_dir)

    print(f"\nDone!")


if __name__ == "__main__":
    main()