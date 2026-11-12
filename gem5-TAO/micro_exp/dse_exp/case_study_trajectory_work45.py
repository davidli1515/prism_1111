"""
Case Study: CPI vs Area trajectories

Scalar:  Each step picks argmax ΔCPI
MSCS:    dominant-type reduction / Δarea + reclaim (same as DSE that beat BO)

Plot: CPI vs Area (Pareto-style)
"""

import argparse, os, sys, numpy as np, torch

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
from area_model import compute_area_cached

import matplotlib; matplotlib.use('Agg')
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

DEFAULT_CONFIG = {
    "fetchWidth": 8, "decodeWidth": 8, "dispatchWidth": 8,
    "commitWidth": 8, "wbWidth": 8,
    "numROBEntries": 192, "numIQEntries": 64,
    "LQEntries": 32, "SQEntries": 32, "fetchBufferSize": 32,
    "IntALU": 2, "IntMultDiv": 1, "FP_ALU": 2, "FP_MultDiv": 1, "RdWrPort": 2,
    "l1_size": 64, "l2_size": 1024,
    "l1d_assoc": 2, "l1i_assoc": 2, "l2_assoc": 8, "bp_type": 0,
}

PARAM_STRIDE = {
    "fetchWidth": 1, "decodeWidth": 1, "dispatchWidth": 1, "commitWidth": 1,
    "numROBEntries": 16, "numIQEntries": 8, "LQEntries": 8, "SQEntries": 8,
    "IntALU": 1, "IntMultDiv": 1, "FP_ALU": 1, "FP_MultDiv": 1, "RdWrPort": 1,
}

RECLAIM_THRESHOLD = 0.05


# ============================================================
# Helpers
# ============================================================
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


def safe_area(config):
    try:
        return compute_area_cached(config)
    except Exception:
        return None


def increase_param(config, name):
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


def decrease_param(config, name):
    c = config.copy()
    vals = get_param_values(name)
    current = c[name]
    if isinstance(ALL_PARAMS.get(name), list):
        idx = vals.index(current) if current in vals else 0
        if idx <= 0:
            return None, None
        c[name] = vals[idx - 1]
    else:
        stride = PARAM_STRIDE.get(name, 1)
        lo = min(vals)
        new_val = max(lo, current - stride)
        if new_val == current:
            return None, None
        c[name] = new_val
    return c, c[name]


# ============================================================
# Scalar: argmax ΔCPI each step
# ============================================================
def run_scalar_trajectory(model, normalizer, dse_engine, workload, config, n_steps=15):
    trajectory = []
    cpi = predict_cpi(model, normalizer, dse_engine, workload, config)
    area = safe_area(config)
    trajectory.append({"step": 0, "cpi": cpi, "area": area, "param": "baseline",
                        "config": config.copy()})
    print(f"\n  Scalar CPI Gradient Trajectory (+ reclaim):")
    print(f"    Step 0: CPI={cpi:.4f}, Area={area}")

    for step in range(1, n_steps + 1):
        # --- Increase: argmax ΔCPI ---
        best_param, best_dcpi, best_config = None, 0, None
        best_old, best_new = None, None
        for name in GRADIENT_PARAMS:
            if name not in PARAM_TO_GD_INDEX:
                continue
            c_new, new_val = increase_param(config, name)
            if c_new is None:
                continue
            cpi_new = predict_cpi(model, normalizer, dse_engine, workload, c_new)
            dcpi = cpi - cpi_new
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
        changes = [f"+{best_param}({best_old}→{best_new})"]

        # --- Reclaim: bottom-3 by |ΔCPI_decrease| ---
        config_pre = config.copy()
        cpi_pre = cpi
        reclaim_cands = []
        for name in GRADIENT_PARAMS:
            if name == best_param or name not in PARAM_TO_GD_INDEX:
                continue
            c_dec, dec_val = decrease_param(config_pre, name)
            if c_dec is None:
                continue
            cpi_dec = predict_cpi(model, normalizer, dse_engine, workload, c_dec)
            cost = cpi_dec - cpi_pre  # positive = CPI worsens
            if cost < RECLAIM_THRESHOLD:
                reclaim_cands.append({"name": name, "cost": cost, "config": c_dec, "val": dec_val})

        reclaim_cands.sort(key=lambda x: x["cost"])
        reclaim_changes = []
        for rc in reclaim_cands[:3]:
            config = decrease_param(config, rc["name"])[0] or config
            reclaim_changes.append(f"-{rc['name']}→{rc['val']}")

        if reclaim_changes:
            cpi_after = predict_cpi(model, normalizer, dse_engine, workload, config)
            if cpi_after > cpi_pre * 1.005:
                config = config_pre
            else:
                cpi = cpi_after
                changes.extend(reclaim_changes)

        area = safe_area(config)
        trajectory.append({"step": step, "cpi": cpi, "area": area, "param": best_param,
                            "old_val": best_old, "new_val": best_new, "config": config.copy()})
        ch_str = ", ".join(changes[:4])
        print(f"    Step {step}: {ch_str} → CPI={cpi:.4f} Area={area}")
    return trajectory


# ============================================================
# MSCS: dominant-type reduction / Δarea + reclaim
# ============================================================
def run_mscs_trajectory(model, normalizer, dse_engine, workload, config, n_steps=15):
    trajectory = []
    cpi = predict_cpi(model, normalizer, dse_engine, workload, config)
    area = safe_area(config)
    trajectory.append({"step": 0, "cpi": cpi, "area": area, "param": "baseline",
                        "dominant": "-", "config": config.copy()})
    print(f"\n  MSCS-Guided Trajectory (dominant reduction / Δarea + reclaim):")
    print(f"    Step 0: CPI={cpi:.4f}, Area={area}")

    for step in range(1, n_steps + 1):
        mscs = predict_mscs(model, normalizer, dse_engine, workload, config)
        fp_avg = np.mean([mscs[i] for i in FP_INDICES])
        bp_avg = np.mean([mscs[i] for i in BP_INDICES])
        sp_avg = np.mean([mscs[i] for i in SP_INDICES])
        pressures = {"FP": fp_avg, "BP": bp_avg, "SP": sp_avg}
        dominant = max(pressures, key=pressures.get)
        dom_indices = BOTTLENECK_INDICES[dominant]
        COMMIT_IDX = {"FP": 13, "BP": 14, "SP": 15}
        dom_commit = COMMIT_IDX[dominant]

        jacobian = compute_jacobian(model, normalizer, dse_engine, workload, config)

        # --- Increase: dominant-type reduction / Δarea, with safety valve ---
        area_old = safe_area(config)
        candidates = []
        all_reductions = {}
        for name, jac_row in jacobian.items():
            vals = get_param_values(name)
            param_range = max(vals) - min(vals)
            # Reduction: commit-stage only (no double counting)
            reduction = max(0, -jac_row[dom_commit]) * param_range
            # Total reduction for reclaim: 4-stage sum is OK here (checking if param matters at all)
            total_reduction = (np.maximum(0, -jac_row[FP_INDICES]).sum() +
                               np.maximum(0, -jac_row[BP_INDICES]).sum() +
                               np.maximum(0, -jac_row[SP_INDICES]).sum()) * param_range
            all_reductions[name] = total_reduction

            if reduction < 1e-8:
                continue
            c_new, new_val = increase_param(config, name)
            if c_new is None:
                continue
            area_new = safe_area(c_new)
            if area_new is None or area_old is None:
                continue
            area_cost = max(area_new - area_old, 1e-6)
            efficiency = reduction

            # Safety valve
            cpi_new = predict_cpi(model, normalizer, dse_engine, workload, c_new)
            if cpi_new >= cpi:
                continue

            candidates.append({
                "name": name, "efficiency": efficiency, "reduction": reduction,
                "config": c_new, "old_val": config[name], "new_val": new_val,
                "cpi_new": cpi_new, "area_new": area_new,
            })

        changes = []
        if candidates:
            candidates.sort(key=lambda x: x["efficiency"], reverse=True)
            best = candidates[0]
            config = best["config"]
            cpi = best["cpi_new"]
            area = best["area_new"]
            changes.append(f"+{best['name']}({best['old_val']}→{best['new_val']})")

        # --- Reclaim: bottom-3 by total MSCS reduction ---
        config_pre_reclaim = config.copy()
        cpi_pre_reclaim = cpi
        reclaim_cands = []
        for name, total in all_reductions.items():
            if any(name in c for c in changes):
                continue
            if total < RECLAIM_THRESHOLD:
                c_dec, dec_val = decrease_param(config, name)
                if c_dec is not None:
                    reclaim_cands.append({"name": name, "total": total, "val": dec_val})

        reclaim_cands.sort(key=lambda x: x["total"])
        reclaim_changes = []
        for rc in reclaim_cands[:3]:
            config = decrease_param(config, rc["name"])[0] or config
            reclaim_changes.append(f"-{rc['name']}→{rc['val']}")

        # Safety valve: revert if reclaim worsened CPI
        if reclaim_changes:
            cpi_after = predict_cpi(model, normalizer, dse_engine, workload, config)
            if cpi_after > cpi_pre_reclaim * 1.005:  # allow 0.5% tolerance
                config = config_pre_reclaim
                reclaim_changes = []
            else:
                cpi = cpi_after
                changes.extend(reclaim_changes)

        area = safe_area(config)

        trajectory.append({
            "step": step, "cpi": cpi, "area": area,
            "param": changes[0] if changes else "none",
            "dominant": dominant, "config": config.copy(),
        })
        ch_str = ", ".join(changes[:4])
        if len(changes) > 4:
            ch_str += f" +{len(changes)-4}more"
        print(f"    Step {step}: dom={dominant}, {ch_str} "
              f"→ CPI={cpi:.4f} Area={area}")

    return trajectory


# ============================================================
# Plot: CPI vs Area
# ============================================================
def plot_trajectories(scalar_traj, mscs_traj, benchmark, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))

    s_a = [t["area"] for t in scalar_traj if t["area"] is not None]
    s_c = [t["cpi"]  for t in scalar_traj if t["area"] is not None]
    m_a = [t["area"] for t in mscs_traj if t["area"] is not None]
    m_c = [t["cpi"]  for t in mscs_traj if t["area"] is not None]

    ax.plot(s_a, s_c, 'o-', color='#3498db', linewidth=2.5, markersize=9,
            label='Scalar ∂CPI/∂θ', zorder=4)
    ax.plot(m_a, m_c, 'D-', color='#2ecc71', linewidth=2.5, markersize=9,
            label='MSCS-guided (ours)', zorder=4)

    # Annotate scalar steps
    valid_s = [t for t in scalar_traj if t["area"] is not None and t["param"] != "baseline"]
    for i, t in enumerate(valid_s):
        if i < 6 or i % 3 == 0:
            ax.annotate(f"+{t['param']}", (t["area"], t["cpi"]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=7, color='#2471a3', ha='left')

    # Annotate MSCS steps
    valid_m = [t for t in mscs_traj if t["area"] is not None and t["param"] != "baseline"]
    for i, t in enumerate(valid_m):
        if i < 6 or i % 3 == 0:
            p = t["param"].split("(")[0] if "(" in t["param"] else t["param"]
            ax.annotate(p, (t["area"], t["cpi"]),
                        textcoords="offset points", xytext=(-8, -12),
                        fontsize=7, color='#1e8449', ha='right')

    ax.set_xlabel('Area (mm²)', fontsize=13)
    ax.set_ylabel('CPI', fontsize=13)
    ax.set_title(f'CPI vs Area — {benchmark}', fontsize=14, pad=15)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{benchmark}_trajectory_cpi_area.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="605.mcf_s",
                        choices=list(PRECOMPUTE_DIRS.keys()))
    parser.add_argument("--n-steps", type=int, default=15)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    benchmark = args.benchmark
    bname = {"605.mcf_s": "mcf", "623.xalancbmk_s": "xalan"}[benchmark]
    output_dir = os.path.join(args.output_dir, benchmark)
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(DSE_DIR)

    model = GEMModel(input_dim=GEM_INPUT_DIM, hidden_dims=GEM_HIDDEN_DIMS, dropout=0.1)
    model.load_state_dict(torch.load(GEM_WEIGHT_PATH, map_location="cpu"))
    model.eval()

    gem_norm = FeatureNormalizer()
    gem_norm.mean = np.load(os.path.join(NORMALIZER_CACHE, "gem_mean.npy"))
    gem_norm.std  = np.load(os.path.join(NORMALIZER_CACHE, "gem_std.npy"))

    dse_engine = DseEngine(list(PRECOMPUTE_DIRS.values()))

    print(f"\n{'='*60}")
    print(f"CPI vs Area Trajectory — {benchmark}")
    print(f"{'='*60}")

    scalar_traj = run_scalar_trajectory(
        model, gem_norm, dse_engine, benchmark, DEFAULT_CONFIG.copy(), args.n_steps)
    mscs_traj = run_mscs_trajectory(
        model, gem_norm, dse_engine, benchmark, DEFAULT_CONFIG.copy(), args.n_steps)

    # Summary
    bc, ba = scalar_traj[0]["cpi"], scalar_traj[0]["area"]
    sc, sa = scalar_traj[-1]["cpi"], scalar_traj[-1]["area"]
    mc, ma = mscs_traj[-1]["cpi"], mscs_traj[-1]["area"]

    print(f"\n  {'='*60}")
    print(f"  Baseline:  CPI={bc:.4f}, Area={ba}")
    print(f"  Scalar:    CPI={sc:.4f} ({(sc-bc)/bc*100:+.1f}%), Area={sa}")
    print(f"  MSCS:      CPI={mc:.4f} ({(mc-bc)/bc*100:+.1f}%), Area={ma}")
    print(f"  {'='*60}")

    plot_trajectories(scalar_traj, mscs_traj, bname, output_dir)


if __name__ == "__main__":
    main()