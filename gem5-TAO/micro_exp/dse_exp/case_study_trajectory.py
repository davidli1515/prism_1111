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

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 600,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def _style_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
    ax.xaxis.grid(False)


def _save_fig(fig, output_dir, name):
    png_path = os.path.join(output_dir, f"{name}.png")
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")

GEM_INPUT_DIM   = 4681
GEM_HIDDEN_DIMS = [512, 256]
GEM_WEIGHT_PATH = "/scratch/gem5-TAO/micro_exp/exp1/results/exp1_v2/exp1_512_256/gem_full.pt"
# GEM_WEIGHT_PATH = "/scratch/gem5-TAO/micro_exp/exp4/results_77/exp4_18k_extra_unlabeled/checkpoints/extra_100pct.pt"

NORMALIZER_CACHE = os.path.join(DSE_DIR, "normalizer_cache")
OUTPUT_DIR = os.path.join(DSE_DIR, "results")

PRECOMPUTE_BASE = os.path.join(DSE_DIR, "precompute")

BENCHMARKS = [
    "508.namd_r", "526.blender_r", "600.perlbench_s", "602.gcc_s",
    "605.mcf_s", "607.cactuBSSN_s", "619.lbm_s", "620.omnetpp_s",
    "621.wrf_s", "623.xalancbmk_s", "631.deepsjeng_s", "638.imagick_s",
    "641.leela_s", "644.nab_s", "648.exchange2_s", "654.roms_s", "657.xz_s",
]

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

RECLAIM_THRESHOLD = 0.05       # for scalar: ΔCPI cost of decreasing
MSCS_RECLAIM_THRESHOLD = 0.02   # for MSCS: 4-stage total < threshold


# ============================================================
# Helpers
# ============================================================
MAX_SLICES = 200  # cap slices for speed


def predict_cpi(model, normalizer, dse_engine, workload, config):
    all_feats = dse_engine.get_features(config, workload=workload)[:MAX_SLICES]
    batch = np.stack([f[:GEM_INPUT_DIM] for f in all_feats])
    x = torch.from_numpy(normalizer.transform(batch)).float()
    with torch.no_grad():
        mscs = model(x)
        w_min = float(compute_w_min(config))
        cpis = 1.0 / (mscs[:, 12] * w_min + 1e-8)
        return cpis.mean().item()


def predict_mscs(model, normalizer, dse_engine, workload, config):
    all_feats = dse_engine.get_features(config, workload=workload)[:MAX_SLICES]
    batch = np.stack([f[:GEM_INPUT_DIM] for f in all_feats])
    x = torch.from_numpy(normalizer.transform(batch)).float()
    with torch.no_grad():
        return model(x).mean(dim=0).cpu().numpy()


def compute_jacobian(model, normalizer, dse_engine, workload, config, n_slices=50):
    all_feats = dse_engine.get_features(config, workload=workload)[:MAX_SLICES]
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
        # --- Increase: argmax ΔCPI (no anti-oscillation — this IS the problem) ---
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

        # --- Reclaim: ΔCPI cost < threshold, no MSCS protection ---
        config_pre = config.copy()
        cpi_pre = cpi
        reclaim_cands = []
        all_costs = {}
        for name in GRADIENT_PARAMS:
            if name == best_param or name not in PARAM_TO_GD_INDEX:
                continue
            c_dec, dec_val = decrease_param(config_pre, name)
            if c_dec is None:
                continue
            cpi_dec = predict_cpi(model, normalizer, dse_engine, workload, c_dec)
            cost = cpi_dec - cpi_pre
            all_costs[name] = cost
            if cost < RECLAIM_THRESHOLD:
                reclaim_cands.append({"name": name, "cost": cost, "val": dec_val})

        if step <= 2:
            print(f"      [SCALAR reclaim] ΔCPI cost (threshold={RECLAIM_THRESHOLD}):")
            print(f"        {'param':<20} {'ΔCPI_dec':>10} {'action':>10}")
            for name, cost in sorted(all_costs.items(), key=lambda x: x[1]):
                action = "RECLAIM" if cost < RECLAIM_THRESHOLD else "KEEP"
                print(f"        {name:<20} {cost:>+10.4f} {action:>10}")

        reclaim_cands.sort(key=lambda x: x["cost"])
        for rc in reclaim_cands:
            c_dec, _ = decrease_param(config, rc["name"])
            if c_dec is None:
                continue
            cpi_after = predict_cpi(model, normalizer, dse_engine, workload, c_dec)
            if cpi_after <= cpi * 1.005:
                config = c_dec
                cpi = cpi_after
                changes.append(f"-{rc['name']}→{rc['val']}")

        # Global safety valve
        if cpi > cpi_pre * 1.005:
            config = config_pre
            cpi = cpi_pre
            changes = [c for c in changes if not c.startswith("-")]

        area = safe_area(config)
        trajectory.append({"step": step, "cpi": cpi, "area": area, "param": best_param,
                            "old_val": best_old, "new_val": best_new, "config": config.copy()})
        ch_str = ", ".join(changes[:4])
        if len(changes) > 4:
            ch_str += f" +{len(changes)-4}more"
        print(f"    Step {step}: {ch_str} → CPI={cpi:.4f} Area={area}")
    return trajectory


# ============================================================
# MSCS: finite-diff ΔMSCS for both increase and reclaim
#
# Increase: overlap-free score = Σ_type max_stage(max(0, ΔMSCS))
#   - Each type measured independently (orthogonal, no cross-type pollution)
#   - Max over stages within type (no double counting)
#   - Sum across types (captures full contribution, not just dominant)
#
# Reclaim: ΔCPI proposes, ΔMSCS protects
#   - ΔCPI < threshold → candidate for reclaim
#   - But if ΔMSCS shows large per-type effects that cancel → PROTECT
#   - Only reclaim when ΔMSCS also shows no effect on any type/stage
# ============================================================
def compute_mscs_finite_diff_increase(model, normalizer, dse_engine, workload, config, name):
    """Finite-diff ΔMSCS for increasing a parameter by one stride."""
    c_new, new_val = increase_param(config, name)
    if c_new is None:
        return None, None, None
    mscs_old = predict_mscs(model, normalizer, dse_engine, workload, config)
    mscs_new = predict_mscs(model, normalizer, dse_engine, workload, c_new)
    delta = mscs_old - mscs_new  # positive = type pressure reduced = improvement
    return c_new, new_val, delta


def compute_mscs_finite_diff_decrease(model, normalizer, dse_engine, workload, config, name):
    """Finite-diff ΔMSCS for decreasing a parameter by one stride."""
    c_dec, dec_val = decrease_param(config, name)
    if c_dec is None:
        return None, None, None
    mscs_old = predict_mscs(model, normalizer, dse_engine, workload, config)
    mscs_dec = predict_mscs(model, normalizer, dse_engine, workload, c_dec)
    delta = mscs_dec - mscs_old  # positive = type pressure increased = worsening
    return c_dec, dec_val, delta


def mscs_total_effect(delta_mscs):
    """Total effect across all 16 dims (L1 norm). For reclaim detection."""
    return np.sum(np.abs(delta_mscs))


def cancel_score(delta_mscs):
    """
    Cross-type cancellation score: how much do different types cancel each other?
    High score = parameter changes multiple bottleneck types in opposite directions
    → removing overlap, exposing clean gradient landscape.
    """
    # Per-type net effect at max stage (captures pipeline-masked effects too)
    type_effects = []
    for type_indices in [FP_INDICES, BP_INDICES, SP_INDICES]:
        # Use the stage with largest absolute effect (may be masked at commit)
        vals = delta_mscs[type_indices]
        max_abs_idx = np.argmax(np.abs(vals))
        type_effects.append(vals[max_abs_idx])

    # Cancel score: sum of absolute per-type effects minus absolute of their sum
    # High when types have large effects that cancel each other
    abs_sum = sum(abs(e) for e in type_effects)
    sum_abs = abs(sum(type_effects))
    return abs_sum - sum_abs, type_effects


DECOUPLE_THRESHOLD = 0.01  # ΔCPI below this = plateau → try decouple


def run_mscs_trajectory(model, normalizer, dse_engine, workload, config, n_steps=15):
    trajectory = []
    cpi = predict_cpi(model, normalizer, dse_engine, workload, config)
    area = safe_area(config)
    trajectory.append({"step": 0, "cpi": cpi, "area": area, "param": "baseline",
                        "config": config.copy()})
    print(f"\n  MSCS-Guided Trajectory (ΔCPI primary + MSCS decouple/protect):")
    print(f"    Step 0: CPI={cpi:.4f}, Area={area}")

    last_reclaimed = set()  # params reclaimed in previous step (anti-oscillation)

    for step in range(1, n_steps + 1):
        # --- Phase 1: Increase by argmax ΔCPI (same as scalar) ---
        cpi_candidates = []
        for name in GRADIENT_PARAMS:
            if name not in PARAM_TO_GD_INDEX:
                continue
            if name in last_reclaimed:
                continue  # don't re-increase what was just reclaimed last step
            c_new, new_val = increase_param(config, name)
            if c_new is None:
                continue
            cpi_new = predict_cpi(model, normalizer, dse_engine, workload, c_new)
            dcpi = cpi - cpi_new
            if dcpi <= 0:
                continue
            cpi_candidates.append({
                "name": name, "delta_cpi": dcpi,
                "config": c_new, "old_val": config[name], "new_val": new_val,
                "cpi_new": cpi_new,
            })

        # Check if we have good ΔCPI candidates
        best_dcpi = max((c["delta_cpi"] for c in cpi_candidates), default=0)
        mode = "ΔCPI"

        if best_dcpi >= DECOUPLE_THRESHOLD:
            # Normal mode: pick argmax ΔCPI
            cpi_candidates.sort(key=lambda x: x["delta_cpi"], reverse=True)
            best = cpi_candidates[0]
        elif cpi_candidates:
            # Plateau: all ΔCPI < threshold → MSCS decouple mode
            mode = "DECOUPLE"
            decouple_candidates = []
            for c in cpi_candidates:
                _, _, delta = compute_mscs_finite_diff_increase(
                    model, normalizer, dse_engine, workload, config, c["name"])
                if delta is None:
                    continue
                cscore, type_effs = cancel_score(delta)
                c["cancel_score"] = cscore
                c["type_effects"] = type_effs
                decouple_candidates.append(c)

            if not decouple_candidates:
                break
            decouple_candidates.sort(key=lambda x: x["cancel_score"], reverse=True)
            best = decouple_candidates[0]

            if step <= 5:
                print(f"      [MSCS DECOUPLE] plateau detected (best ΔCPI={best_dcpi:.4f} < {DECOUPLE_THRESHOLD})")
                print(f"        {'param':<20} {'cancel':>10} {'ΔCPI':>10} {'ΔFP':>8} {'ΔBP':>8} {'ΔSP':>8}")
                for c in decouple_candidates[:5]:
                    te = c["type_effects"]
                    marker = " ←" if c["name"] == best["name"] else ""
                    print(f"        {c['name']:<20} {c['cancel_score']:>10.4f} {c['delta_cpi']:>+10.4f} "
                          f"{te[0]:>+8.4f} {te[1]:>+8.4f} {te[2]:>+8.4f}{marker}")
        else:
            break

        if step <= 2 and mode == "ΔCPI":
            print(f"      [ΔCPI increase] (normal mode, best ΔCPI={best_dcpi:.4f})")
            print(f"        {'param':<20} {'ΔCPI':>10}")
            for c in cpi_candidates[:5]:
                marker = " ←" if c["name"] == best["name"] else ""
                print(f"        {c['name']:<20} {c['delta_cpi']:>+10.4f}{marker}")

        config = best["config"]
        cpi = best["cpi_new"]
        increased_name = best["name"]
        changes = [f"+{best['name']}({best['old_val']}→{best['new_val']})"]
        if mode == "DECOUPLE":
            changes[0] += " [decouple]"

        # --- Phase 2: Reclaim ---
        # In DECOUPLE mode: skip reclaim (investment phase, don't undo landscape changes)
        # Exception: RECLAIM* (ΔCPI < 0, removing actively improves CPI)
        # In ΔCPI mode: full reclaim with MSCS PROTECT
        config_pre_reclaim = config.copy()
        cpi_pre_reclaim = cpi
        this_reclaimed = set()

        if mode == "DECOUPLE":
            # Only reclaim actively harmful params (ΔCPI < 0)
            for name in GRADIENT_PARAMS:
                if name == increased_name or name not in PARAM_TO_GD_INDEX:
                    continue
                c_dec, dec_val = decrease_param(config, name)
                if c_dec is None:
                    continue
                cpi_dec = predict_cpi(model, normalizer, dse_engine, workload, c_dec)
                dcpi_cost = cpi_dec - cpi
                if dcpi_cost < 0:
                    cpi_after = predict_cpi(model, normalizer, dse_engine, workload, c_dec)
                    if cpi_after <= cpi * 1.005:
                        config = c_dec
                        cpi = cpi_after
                        changes.append(f"-{name}→{dec_val}")
                        this_reclaimed.add(name)
        else:
            # Normal ΔCPI mode: full reclaim with MSCS PROTECT
            if step <= 2:
                print(f"      [MSCS reclaim]")
                print(f"        {'param':<20} {'ΔCPI_dec':>10} {'MSCS_effect':>12} {'action':>10}")
            for name in GRADIENT_PARAMS:
                if name == increased_name or name not in PARAM_TO_GD_INDEX:
                    continue

                c_dec, dec_val = decrease_param(config, name)
                if c_dec is None:
                    continue
                cpi_dec = predict_cpi(model, normalizer, dse_engine, workload, c_dec)
                dcpi_cost = cpi_dec - cpi

                if dcpi_cost >= RECLAIM_THRESHOLD:
                    if step <= 2:
                        print(f"        {name:<20} {dcpi_cost:>+10.4f} {'':>12} {'KEEP':>10}")
                    continue

                # ΔCPI < 0: removing IMPROVES CPI → always reclaim
                if dcpi_cost < 0:
                    cpi_after = predict_cpi(model, normalizer, dse_engine, workload, c_dec)
                    if cpi_after <= cpi * 1.005:
                        if step <= 2:
                            print(f"        {name:<20} {dcpi_cost:>+10.4f} {'':>12} {'RECLAIM*':>10}")
                        config = c_dec
                        cpi = cpi_after
                        changes.append(f"-{name}→{dec_val}")
                        this_reclaimed.add(name)
                    continue

                # 0 <= ΔCPI < threshold: MSCS decides
                _, _, delta_dec = compute_mscs_finite_diff_decrease(
                    model, normalizer, dse_engine, workload, config, name)
                if delta_dec is None:
                    continue
                total_eff = mscs_total_effect(delta_dec)

                if total_eff >= MSCS_RECLAIM_THRESHOLD:
                    if step <= 2:
                        print(f"        {name:<20} {dcpi_cost:>+10.4f} {total_eff:>12.6f} {'PROTECT':>10}")
                    continue

                # Both agree: truly useless
                cpi_after = predict_cpi(model, normalizer, dse_engine, workload, c_dec)
                if cpi_after <= cpi * 1.005:
                    if step <= 2:
                        print(f"        {name:<20} {dcpi_cost:>+10.4f} {total_eff:>12.6f} {'RECLAIM':>10}")
                    config = c_dec
                    cpi = cpi_after
                    changes.append(f"-{name}→{dec_val}")
                    this_reclaimed.add(name)
                elif step <= 2:
                    print(f"        {name:<20} {dcpi_cost:>+10.4f} {total_eff:>12.6f} {'BLOCKED':>10}")

        # Global safety valve
        if cpi > cpi_pre_reclaim * 1.005:
            config = config_pre_reclaim
            cpi = cpi_pre_reclaim
            changes = [c for c in changes if not c.startswith("-")]
            this_reclaimed = set()

        area = safe_area(config)

        trajectory.append({
            "step": step, "cpi": cpi, "area": area,
            "param": best["name"], "mode": mode, "config": config.copy(),
        })
        ch_str = ", ".join(changes[:4])
        if len(changes) > 4:
            ch_str += f" +{len(changes)-4}more"
        print(f"    Step {step}: [{mode}] {ch_str} → CPI={cpi:.4f} Area={area}")

        last_reclaimed = this_reclaimed  # anti-oscillation for next step

    return trajectory


# ============================================================
# Plot: CPI vs Area
# ============================================================
def plot_scalar_only(scalar_traj, benchmark, output_dir):
    """Standalone scalar CPI trajectory."""
    fig, ax = plt.subplots(figsize=(8, 6))
    valid = [t for t in scalar_traj if t["area"] is not None]
    areas = [t["area"] for t in valid]
    cpis = [t["cpi"] for t in valid]

    ax.plot(areas, cpis, 'o-', color='#3498db', linewidth=2.5, markersize=8, zorder=4)

    for i, t in enumerate(valid):
        if t["param"] == "baseline":
            ax.annotate("baseline", (t["area"], t["cpi"]),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=8, color='#2471a3', fontweight='bold')
        else:
            y_off = 10 if i % 2 == 0 else -12
            ha = 'left' if i % 2 == 0 else 'right'
            x_off = 6 if i % 2 == 0 else -6
            ax.annotate(f"S{i}: +{t['param']}", (t["area"], t["cpi"]),
                        textcoords="offset points", xytext=(x_off, y_off),
                        fontsize=7, color='#2471a3', ha=ha)

    ax.set_xlabel('Area (mm²)', fontweight='bold')
    ax.set_ylabel('CPI', fontweight='bold')
    ax.set_title(f'Scalar ΔCPI — {benchmark}', fontweight='bold', pad=12)

    _style_axes(ax)

    plt.tight_layout()
    _save_fig(fig, output_dir, f"{benchmark}_scalar_cpi_area")


def plot_trajectories(scalar_traj, mscs_traj, benchmark, output_dir):
    """Combined CPI vs Area trajectory comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    s_valid = [t for t in scalar_traj if t["area"] is not None]
    m_valid = [t for t in mscs_traj if t["area"] is not None]
    s_a = [t["area"] for t in s_valid]
    s_c = [t["cpi"] for t in s_valid]
    m_a = [t["area"] for t in m_valid]
    m_c = [t["cpi"] for t in m_valid]

    ax.plot(s_a, s_c, 'o-', color='#3498db', linewidth=2.5, markersize=9,
            label='Scalar ΔCPI', zorder=4, markeredgecolor='white', markeredgewidth=1.0)
    ax.plot(m_a, m_c, 'D-', color='#2ecc71', linewidth=2.5, markersize=9,
            label='MSCS-guided', zorder=5, markeredgecolor='white', markeredgewidth=1.0)

    # Step numbers inside markers
    for i in range(len(s_valid)):
        ax.annotate(str(i), (s_a[i], s_c[i]), fontsize=5.5, color='white',
                    ha='center', va='center', fontweight='bold', zorder=6)
    for i in range(len(m_valid)):
        ax.annotate(str(i), (m_a[i], m_c[i]), fontsize=5.5, color='white',
                    ha='center', va='center', fontweight='bold', zorder=7)

    ax.set_xlabel('Area (mm²)', fontweight='bold')
    ax.set_ylabel('CPI', fontweight='bold')
    ax.set_title(f'CPI vs Area — {benchmark}', fontweight='bold', pad=12)

    _style_axes(ax)

    legend = ax.legend(loc='upper right', frameon=True, edgecolor='black',
                        facecolor='white', fancybox=False, framealpha=1.0,
                        fontsize=12, borderpad=1.0)
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    _save_fig(fig, output_dir, f"{benchmark}_trajectory_cpi_area")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", nargs="+", default=["all"],
                        help="Benchmark(s) to run, or 'all' for all 17")
    parser.add_argument("--n-steps", type=int, default=15)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.benchmark == ["all"]:
        benchmarks = BENCHMARKS
    else:
        benchmarks = args.benchmark

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(DSE_DIR)

    model = GEMModel(input_dim=GEM_INPUT_DIM, hidden_dims=GEM_HIDDEN_DIMS, dropout=0.1)
    model.load_state_dict(torch.load(GEM_WEIGHT_PATH, map_location="cpu"))
    model.eval()

    gem_norm = FeatureNormalizer()
    gem_norm.mean = np.load(os.path.join(NORMALIZER_CACHE, "gem_mean.npy"))
    gem_norm.std  = np.load(os.path.join(NORMALIZER_CACHE, "gem_std.npy"))

    dse_engine = DseEngine(PRECOMPUTE_BASE)

    for benchmark in benchmarks:
        # Short name for plot titles
        bname = benchmark.split(".")[1].replace("_s", "").replace("_r", "")

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

        plot_scalar_only(scalar_traj, bname, output_dir)
        plot_trajectories(scalar_traj, mscs_traj, bname, output_dir)

        # Save trajectory data as JSON for standalone plotting
        import json
        traj_data = {
            "benchmark": benchmark,
            "bname": bname,
            "n_steps": args.n_steps,
            "baseline": {"cpi": bc, "area": ba},
            "scalar": [{"step": t["step"], "cpi": t["cpi"], "area": t["area"],
                         "param": t["param"], "config": t.get("config", {})}
                        for t in scalar_traj],
            "mscs": [{"step": t["step"], "cpi": t["cpi"], "area": t["area"],
                       "param": t["param"], "mode": t.get("mode", "ΔCPI"),
                       "config": t.get("config", {})}
                      for t in mscs_traj],
        }
        json_path = os.path.join(output_dir, f"{bname}_trajectory.json")
        with open(json_path, "w") as f:
            json.dump(traj_data, f, indent=2)
        print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()