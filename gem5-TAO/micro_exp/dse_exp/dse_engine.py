"""
Exp 5c: Penetration-Guided DSE via Multi-Stage Bottleneck Analysis

Core idea: Fix the most PENETRATING bottleneck, not the largest one.
Penetration = bottleneck value at the sink stage (farthest from source).
High penetration = pipeline failed to absorb it = worth fixing.
Low penetration = pipeline already hid it = don't waste resources.

Methods:
  1. Concorde + BO                (baseline: scalar CPI, global search)
  2. Concorde scalar gradient     (baseline: scalar CPI, greedy gradient)
  3. GEM scalar gradient          (better model, same greedy strategy)
  4. GEM Penetration-Guided       (our method: MSCS-aware bottleneck targeting)
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
np.seterr(divide='ignore', invalid='ignore')

from config_dse import (
    ALL_PARAMS, GRADIENT_PARAMS, NON_GRADIENT_PARAMS,
    PARAM_TO_GD_INDEX, GEM_DATA_ARCH_START, GEM_DATA_ARCH_END,
    is_valid_config, sample_random_config,
    config_to_arch_vector, compute_w_min, get_param_values,
)
from area_model import compute_area_cached

GEM_INPUT_DIM = 4681
CONCORDE_INPUT_DIM = 3873


# ============================================================
# MSCS Layout
# ============================================================
# 16 values: 4 stages × 4 types
# Stage order: decode(0), dispatch(1), issue(2), commit(3)
# Type order:  Base(0), FP(1), BP(2), SP(3)
# Index = stage * 4 + type

MSCS_FP_DECODE  = 1;  MSCS_FP_COMMIT  = 13
MSCS_BP_DECODE  = 2;  MSCS_BP_COMMIT  = 14
MSCS_SP_DECODE  = 3;  MSCS_SP_COMMIT  = 15

# Penetration: value at sink stage (lower bound, after full pipeline absorption)
# Source: where the pressure originates (upper bound)
PRESSURE_TYPES = {
    "FP": {
        "source_idx": MSCS_FP_DECODE,   # FP originates at decode (fetch stalls)
        "sink_idx":   MSCS_FP_COMMIT,   # penetration = how much survives to commit
        "params": ["fetchWidth", "decodeWidth", "dispatchWidth",
                   "fetchBufferSize", "l1_size", "l1i_assoc"],
    },
    "BP": {
        "source_idx": MSCS_BP_COMMIT,   # BP originates at commit (retirement stalls)
        "sink_idx":   MSCS_BP_DECODE,   # penetration = how much propagates back to decode
        "params": ["numROBEntries", "numIQEntries", "LQEntries", "SQEntries",
                   "l1_size", "l1d_assoc", "l2_size", "l2_assoc"],
    },
    "SP": {
        "source_idx": MSCS_SP_DECODE,   # SP originates at decode (branch mispred)
        "sink_idx":   MSCS_SP_COMMIT,   # penetration = how much survives to commit
        "params": ["IntALU", "IntMultDiv", "FP_ALU", "FP_MultDiv", "RdWrPort"],
    },
}


# ============================================================
# Pareto Utilities
# ============================================================

def compute_pareto_front(points):
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if (points[j][0] <= points[i][0] and points[j][1] <= points[i][1] and
                    (points[j][0] < points[i][0] or points[j][1] < points[i][1])):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def compute_hypervolume_2d(pareto_points, ref_point):
    if not pareto_points:
        return 0.0
    valid = [(c, a) for c, a in pareto_points if c < ref_point[0] and a < ref_point[1]]
    if not valid:
        return 0.0
    valid.sort(key=lambda x: x[0])
    hv, prev_area = 0.0, ref_point[1]
    for cpi, area in valid:
        hv += (ref_point[0] - cpi) * (prev_area - area)
        prev_area = area
    return hv


def compute_hypervolume_curve(cpis, areas, ref_point):
    curve = []
    pts = []
    for i in range(len(cpis)):
        pts.append((cpis[i], areas[i]))
        pidx = compute_pareto_front(pts)
        hv = compute_hypervolume_2d([pts[j] for j in pidx], ref_point)
        curve.append((i + 1, hv))
    return curve


# ============================================================
# Projection to Valid Config
# ============================================================

def _project_to_valid(config):
    c = config.copy()
    for name in ALL_PARAMS:
        vals = get_param_values(name)
        lo, hi = min(vals), max(vals)
        c[name] = max(lo, min(hi, c[name]))
        if isinstance(ALL_PARAMS[name], list):
            c[name] = min(vals, key=lambda v: abs(v - c[name]))
    if c["decodeWidth"] > c["fetchWidth"]:
        c["decodeWidth"] = c["fetchWidth"]
    if c["dispatchWidth"] > c["decodeWidth"]:
        c["dispatchWidth"] = c["decodeWidth"]
    for p in ["wbWidth", "commitWidth"]:
        if c[p] < c["dispatchWidth"]:
            c[p] = c["dispatchWidth"]
    DIQ = 2 * c["dispatchWidth"]
    min_rob = DIQ + c["numIQEntries"] + c["LQEntries"] + c["SQEntries"]
    if c["numROBEntries"] < min_rob:
        c["numROBEntries"] = min(min_rob, 1024)
    l1_bytes = c["l1_size"] * 1024
    max_l1_assoc = max(1, l1_bytes // (16 * 64))
    if c["l1d_assoc"] > max_l1_assoc:
        vals = get_param_values("l1d_assoc")
        c["l1d_assoc"] = max(v for v in vals if v <= max_l1_assoc)
    if c["l1i_assoc"] > max_l1_assoc:
        vals = get_param_values("l1i_assoc")
        c["l1i_assoc"] = max(v for v in vals if v <= max_l1_assoc)
    return c


# ============================================================
# GEM Predictor
# ============================================================

class GEMPredictor:
    def __init__(self, model, normalizer, dse_engine, workload,
                 n_cpi=200, n_grad=50, device="cpu", max_slices=200):
        self.model = model
        self.model.eval()
        self.normalizer = normalizer
        self.dse_engine = dse_engine
        self.workload = workload
        self.device = device
        self.max_slices = max_slices

        from config_dse import DEFAULT_CONFIG
        total = len(dse_engine.get_features(DEFAULT_CONFIG, workload=workload))
        # Subsample slices: take every Nth to cover full range
        if total > max_slices:
            step = total // max_slices
            self.slice_indices = list(range(0, total, step))[:max_slices]
        else:
            self.slice_indices = list(range(total))
        actual = len(self.slice_indices)

        rng = np.random.RandomState(42)
        self.cpi_indices = rng.choice(actual, size=min(n_cpi, actual), replace=False)
        self.grad_indices = rng.choice(actual, size=min(n_grad, actual), replace=False)
        print(f"  GEMPredictor [{workload}]: {total} total, "
              f"subsampled={actual}, CPI={len(self.cpi_indices)}, grad={len(self.grad_indices)}")

    def _get_features(self, config, indices):
        all_feats = self.dse_engine.get_features(config, workload=self.workload)
        # Only use subsampled slices, then select from those
        subsampled = [all_feats[si] for si in self.slice_indices]
        valid_indices = [i for i in indices if i < len(subsampled)]
        batch = np.stack([subsampled[i][:GEM_INPUT_DIM] for i in valid_indices])
        normalized = self.normalizer.transform(batch)
        return torch.from_numpy(normalized).float().to(self.device)

    def predict_cpi(self, config):
        with torch.no_grad():
            x = self._get_features(config, self.cpi_indices)
            mscs = self.model(x)
            w_min = float(compute_w_min(config))
            cpis = 1.0 / (mscs[:, 12] * w_min + 1e-8)
            return cpis.mean().item()

    def predict_mscs(self, config):
        with torch.no_grad():
            x = self._get_features(config, self.cpi_indices)
            return self.model(x).mean(dim=0).cpu().numpy()

    def _predict_cpi_grad(self, config):
        with torch.no_grad():
            x = self._get_features(config, self.grad_indices)
            mscs = self.model(x)
            w_min = float(compute_w_min(config))
            cpis = 1.0 / (mscs[:, 12] * w_min + 1e-8)
            return cpis.mean().item()

    def compute_cpi_gradient(self, config):
        """Finite difference CPI gradient using grad slices."""
        cpi_base = self._predict_cpi_grad(config)
        grad = {}
        for name in GRADIENT_PARAMS:
            if name not in PARAM_TO_GD_INDEX:
                continue
            vals = get_param_values(name)
            current = config[name]
            if isinstance(ALL_PARAMS.get(name), list):
                idx = vals.index(current) if current in vals else 0
                if idx < len(vals) - 1:
                    c_plus = config.copy(); c_plus[name] = vals[idx + 1]
                    delta = vals[idx + 1] - current
                else:
                    c_plus = config.copy(); c_plus[name] = vals[max(0, idx - 1)]
                    delta = vals[max(0, idx - 1)] - current
            else:
                hi = max(vals)
                if current < hi:
                    c_plus = config.copy(); c_plus[name] = current + 1
                    delta = 1.0
                else:
                    c_plus = config.copy(); c_plus[name] = current - 1
                    delta = -1.0
            if abs(delta) < 1e-10:
                grad[name] = 0.0
                continue
            cpi_plus = self._predict_cpi_grad(c_plus)
            grad[name] = (cpi_plus - cpi_base) / delta
        return grad, cpi_base

    def compute_mscs_jacobian(self, config):
        """
        Backprop Jacobian: ∂MSCS/∂(arch_params) averaged over grad slices.
        Returns dict: param_name → jacobian_row (16,)
        """
        all_feats = self.dse_engine.get_features(config, workload=self.workload)
        a_s, a_e = GEM_DATA_ARCH_START, GEM_DATA_ARCH_END
        norm_std = self.normalizer.std[a_s:a_e]
        n_arch = a_e - a_s

        avg_jac = np.zeros((n_arch, 16))
        n_slices = len(self.grad_indices)

        for si in self.grad_indices:
            feat = all_feats[si][:GEM_INPUT_DIM].copy()
            normalized = self.normalizer.transform(feat.reshape(1, -1)).flatten()
            x = torch.from_numpy(normalized).float().to(self.device)

            for oi in range(16):
                x_arch = x[a_s:a_e].detach().requires_grad_(True)
                x_full = torch.cat([x[:a_s].detach(), x_arch, x[a_e:].detach()])
                out = self.model(x_full.unsqueeze(0))
                out[0, oi].backward()
                avg_jac[:, oi] += x_arch.grad.detach().cpu().numpy()

        avg_jac /= n_slices
        for j in range(16):
            avg_jac[:, j] /= (norm_std + 1e-8)

        # Map to named params
        result = {}
        for name in GRADIENT_PARAMS:
            if name not in PARAM_TO_GD_INDEX:
                continue
            local_idx = PARAM_TO_GD_INDEX[name] - GEM_DATA_ARCH_START
            result[name] = avg_jac[local_idx, :]
        return result


# ============================================================
# Concorde Predictor
# ============================================================

class ConcordePredictor:
    def __init__(self, model, normalizer, dse_engine, workload,
                 n_cpi=200, n_grad=50, device="cpu"):
        self.model = model
        self.model.eval()
        self.normalizer = normalizer
        self.dse_engine = dse_engine
        self.workload = workload
        self.device = device

        from config_dse import DEFAULT_CONFIG
        total = len(dse_engine.get_features(DEFAULT_CONFIG, workload=workload))
        rng = np.random.RandomState(42)
        self.cpi_indices = rng.choice(total, size=min(n_cpi, total), replace=False)
        self.grad_indices = rng.choice(total, size=min(n_grad, total), replace=False)
        print(f"  ConcordePredictor [{workload}]: {total} total, "
              f"CPI={len(self.cpi_indices)}, grad={len(self.grad_indices)}")

    def _get_features(self, config, indices):
        all_feats = self.dse_engine.get_features(config, workload=self.workload)
        batch = np.stack([all_feats[i][:CONCORDE_INPUT_DIM] for i in indices])
        normalized = self.normalizer.transform(batch)
        return torch.from_numpy(normalized).float().to(self.device)

    def predict_cpi(self, config):
        with torch.no_grad():
            x = self._get_features(config, self.cpi_indices)
            return self.model(x).mean().item()

    def _predict_cpi_grad(self, config):
        with torch.no_grad():
            x = self._get_features(config, self.grad_indices)
            return self.model(x).mean().item()

    def compute_cpi_gradient(self, config):
        cpi_base = self._predict_cpi_grad(config)
        grad = {}
        for name in GRADIENT_PARAMS:
            if name not in PARAM_TO_GD_INDEX:
                continue
            vals = get_param_values(name)
            current = config[name]
            if isinstance(ALL_PARAMS.get(name), list):
                idx = vals.index(current) if current in vals else 0
                if idx < len(vals) - 1:
                    c_plus = config.copy(); c_plus[name] = vals[idx + 1]
                    delta = vals[idx + 1] - current
                else:
                    c_plus = config.copy(); c_plus[name] = vals[max(0, idx - 1)]
                    delta = vals[max(0, idx - 1)] - current
            else:
                hi = max(vals)
                if current < hi:
                    c_plus = config.copy(); c_plus[name] = current + 1
                    delta = 1.0
                else:
                    c_plus = config.copy(); c_plus[name] = current - 1
                    delta = -1.0
            if abs(delta) < 1e-10:
                grad[name] = 0.0
                continue
            cpi_plus = self._predict_cpi_grad(c_plus)
            grad[name] = (cpi_plus - cpi_base) / delta
        return grad, cpi_base


# ============================================================
# Multi-Workload Predictor (joint optimization across benchmarks)
# ============================================================
class MultiWorkloadPredictor:
    """
    Wraps multiple GemPredictor instances (one per workload).
    predict_cpi → geometric mean CPI across workloads
    predict_mscs → average MSCS across workloads
    Same interface as GemPredictor → _attribution_step works unchanged.
    """
    def __init__(self, predictors):
        """predictors: list of GemPredictor, one per workload."""
        self.predictors = predictors
        print(f"  MultiWorkloadPredictor: {len(predictors)} workloads")

    def predict_cpi(self, config):
        """Geometric mean CPI across all workloads."""
        cpis = [p.predict_cpi(config) for p in self.predictors]
        return float(np.exp(np.mean(np.log(cpis))))

    def predict_mscs(self, config):
        """Average MSCS across all workloads."""
        mscs_list = [p.predict_mscs(config) for p in self.predictors]
        return np.mean(mscs_list, axis=0)

    def compute_cpi_gradient(self, config):
        """
        Gradient of geometric mean CPI.
        ∂(gmean)/∂θ = gmean × mean(1/cpi_i × ∂cpi_i/∂θ)
        Approximated by finite diff of geometric mean.
        """
        cpi_base = self.predict_cpi(config)
        grad = {}
        for name in GRADIENT_PARAMS:
            if name not in PARAM_TO_GD_INDEX:
                continue
            vals = get_param_values(name)
            current = config[name]
            if isinstance(ALL_PARAMS.get(name), list):
                idx = vals.index(current) if current in vals else 0
                if idx < len(vals) - 1:
                    c_plus = config.copy(); c_plus[name] = vals[idx + 1]
                    delta = vals[idx + 1] - current
                else:
                    c_plus = config.copy(); c_plus[name] = vals[max(0, idx - 1)]
                    delta = vals[max(0, idx - 1)] - current
            else:
                hi = max(vals)
                if current < hi:
                    c_plus = config.copy(); c_plus[name] = current + 1
                    delta = 1.0
                else:
                    c_plus = config.copy(); c_plus[name] = current - 1
                    delta = -1.0
            if abs(delta) < 1e-10:
                grad[name] = 0.0
                continue
            cpi_plus = self.predict_cpi(c_plus)
            grad[name] = (cpi_plus - cpi_base) / delta
        return grad, cpi_base


# ============================================================
# Scalar Gradient Step (for baseline methods 2 & 3)
# ============================================================

def _scalar_gradient_step(config, cpi_grad, threshold, decrease_threshold=0.1,
                           top_k_increase=1):
    new_config = config.copy()
    changes = {}
    abs_grads = np.array([abs(cpi_grad.get(n, 0.0)) for n in GRADIENT_PARAMS
                          if n in PARAM_TO_GD_INDEX])
    max_abs = abs_grads.max() + 1e-10

    increase_candidates = []
    for name in GRADIENT_PARAMS:
        if name not in cpi_grad:
            continue
        if cpi_grad[name] < -threshold:
            increase_candidates.append((name, cpi_grad[name]))
    increase_candidates.sort(key=lambda x: x[1])
    top_k_names = set(name for name, _ in increase_candidates[:top_k_increase])

    for name in GRADIENT_PARAMS:
        if name not in cpi_grad:
            continue
        grad = cpi_grad[name]
        norm_abs = abs(grad) / max_abs
        vals = get_param_values(name)
        lo, hi = min(vals), max(vals)
        current = config[name]

        if name in top_k_names:
            if isinstance(ALL_PARAMS.get(name), list):
                idx = vals.index(current) if current in vals else 0
                if idx < len(vals) - 1:
                    new_config[name] = vals[idx + 1]
                    changes[name] = {"from": current, "to": vals[idx + 1],
                                     "grad": grad, "action": "increase"}
            else:
                new_val = min(hi, current + 1)
                if new_val != current:
                    new_config[name] = new_val
                    changes[name] = {"from": current, "to": new_val,
                                     "grad": grad, "action": "increase"}
        elif norm_abs < decrease_threshold and current > lo:
            if isinstance(ALL_PARAMS.get(name), list):
                idx = vals.index(current) if current in vals else 0
                if idx > 0:
                    new_config[name] = vals[idx - 1]
                    changes[name] = {"from": current, "to": vals[idx - 1],
                                     "grad": grad, "action": "decrease"}
            else:
                new_config[name] = current - 1
                changes[name] = {"from": current, "to": current - 1,
                                 "grad": grad, "action": "decrease"}
    return _project_to_valid(new_config), changes


# ============================================================
# Penetration-Guided Step (Method 4: our contribution)
# ============================================================

def _increase_param(config, name):
    """Try to increase param by one stride. Returns (new_val, old_val) or None."""
    vals = get_param_values(name)
    current = config[name]
    if isinstance(ALL_PARAMS.get(name), list):
        idx = vals.index(current) if current in vals else 0
        if idx < len(vals) - 1:
            return vals[idx + 1], current
    else:
        hi = max(vals)
        if current < hi:
            return current + 1, current
    return None


def _decrease_param(config, name):
    """Try to decrease param by one stride. Returns (new_val, old_val) or None."""
    vals = get_param_values(name)
    current = config[name]
    if isinstance(ALL_PARAMS.get(name), list):
        idx = vals.index(current) if current in vals else 0
        if idx > 0:
            return vals[idx - 1], current
    else:
        lo = min(vals)
        if current > lo:
            return current - 1, current
    return None


def _penetration_step(config, gem_predictor, reclaim_threshold=0.02,
                       verbose=False):
    """
    Penetration-guided DSE step.

    1. Predict MSCS → compute penetration for FP, BP, SP
    2. Target the most penetrating bottleneck → increase best resource
    3. Reclaim resources for bottlenecks with raw ≈ 0 at source
    """
    mscs = gem_predictor.predict_mscs(config)
    changes = {}

    # --- Step 1: Compute penetration and raw ---
    penetrations = {}
    raws = {}
    for ptype, info in PRESSURE_TYPES.items():
        penetrations[ptype] = float(mscs[info["sink_idx"]])
        raws[ptype] = float(mscs[info["source_idx"]])

    # --- Step 2: Increase resources for most penetrating bottleneck ---
    target_type = max(penetrations, key=penetrations.get)
    target_info = PRESSURE_TYPES[target_type]
    target_params = [p for p in target_info["params"] if p in config]

    # Pick best resource: the one whose increase reduces penetration most
    # Use finite diff of the sink-stage pressure
    best_param = None
    best_reduction = 0.0

    for pname in target_params:
        result = _increase_param(config, pname)
        if result is None:
            continue
        new_val, old_val = result
        c_test = config.copy()
        c_test[pname] = new_val
        c_test = _project_to_valid(c_test)
        mscs_test = gem_predictor.predict_mscs(c_test)
        new_pen = float(mscs_test[target_info["sink_idx"]])
        reduction = penetrations[target_type] - new_pen
        if reduction > best_reduction:
            best_reduction = reduction
            best_param = pname

    new_config = config.copy()

    if best_param is not None:
        result = _increase_param(config, best_param)
        if result:
            new_config[best_param] = result[0]
            changes[best_param] = {
                "from": result[1], "to": result[0],
                "action": "increase",
                "reason": f"{target_type}_penetration={penetrations[target_type]:.4f}",
                "reduction": best_reduction,
            }

    # --- Step 3: Reclaim over-provisioned resources ---
    for ptype, info in PRESSURE_TYPES.items():
        if raws[ptype] < reclaim_threshold:
            # Source stage says ≈ 0 → this bottleneck type is over-provisioned
            # Pick the resource with the largest current area contribution
            # (simplification: pick the one with largest current value relative to max)
            reclaim_params = [p for p in info["params"]
                              if p in config and p not in changes]
            best_reclaim = None
            best_ratio = 0.0
            for pname in reclaim_params:
                vals = get_param_values(pname)
                lo, hi = min(vals), max(vals)
                current = config[pname]
                if current <= lo:
                    continue
                ratio = (current - lo) / (hi - lo + 1e-10)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_reclaim = pname

            if best_reclaim is not None:
                result = _decrease_param(new_config, best_reclaim)
                if result:
                    new_config[best_reclaim] = result[0]
                    changes[best_reclaim] = {
                        "from": result[1], "to": result[0],
                        "action": "reclaim",
                        "reason": f"{ptype}_raw={raws[ptype]:.4f}<{reclaim_threshold}",
                    }

    info = {
        "penetrations": penetrations,
        "raws": raws,
        "target": target_type,
        "mscs": mscs.tolist(),
    }

    if verbose:
        print(f"      Pen: FP={penetrations['FP']:.4f} BP={penetrations['BP']:.4f} "
              f"SP={penetrations['SP']:.4f} → target={target_type}")
        print(f"      Raw: FP={raws['FP']:.4f} BP={raws['BP']:.4f} SP={raws['SP']:.4f}")
        inc = [f"{k}({v['from']}→{v['to']})" for k, v in changes.items()
               if v['action'] == 'increase']
        rec = [f"{k}({v['from']}→{v['to']})" for k, v in changes.items()
               if v['action'] == 'reclaim']
        if inc: print(f"      ↑ {', '.join(inc)}")
        if rec: print(f"      ↓ {', '.join(rec)}")

    return _project_to_valid(new_config), changes, info


# ============================================================
# Search Loops
# ============================================================

def _run_gradient_search(step_fn, predictor, area_fn, method_name,
                          n_starts=30, max_steps=30, seed=42, verbose=True):
    """Search loop for gradient-based methods."""
    rng = np.random.RandomState(seed)
    configs, cpis, areas = [], [], []
    trajectories = []
    t0 = time.time()

    budget_multipliers = np.linspace(1.0, 4.0, n_starts)

    if verbose:
        print(f"\n{'='*60}\n{method_name} (starts={n_starts}, steps={max_steps})\n{'='*60}")

    for si in range(n_starts):
        config = sample_random_config(rng)
        cpi = predictor.predict_cpi(config)
        try:
            area = area_fn(config)
        except Exception:
            continue
        configs.append(config.copy()); cpis.append(cpi); areas.append(area)

        area_budget = area * budget_multipliers[si]
        trajectory = [{"step": 0, "cpi": cpi, "area": area, "config": config.copy()}]

        if verbose:
            print(f"\n  Start {si+1}/{n_starts}: CPI={cpi:.4f}, Area={area:.2f}")

        prev_cpi, stagnant = cpi, 0
        for step in range(1, max_steps + 1):
            over_budget = area > area_budget
            new_config, changes, info = step_fn(config, predictor,
                                                 override_no_increase=over_budget)
            actual = {k: v for k, v in changes.items() if v.get("action") != "hold"}
            if not actual:
                break

            new_cpi = predictor.predict_cpi(new_config)
            try:
                new_area = area_fn(new_config)
            except Exception:
                break
            configs.append(new_config.copy()); cpis.append(new_cpi); areas.append(new_area)

            n_inc = sum(1 for v in changes.values() if v.get("action", "").startswith("increase"))
            n_dec = sum(1 for v in changes.values() if v["action"] in ("decrease", "reclaim"))

            trajectory.append({"step": step, "cpi": new_cpi, "area": new_area,
                                "config": new_config.copy(), "changes": changes})

            imp = abs(new_cpi - prev_cpi) / (abs(prev_cpi) + 1e-8)
            if imp < 0.001:
                stagnant += 1
                if stagnant >= 3: break
            else:
                stagnant = 0

            if verbose and step % 5 == 0:
                print(f"    Step {step}: CPI={new_cpi:.4f}, Area={new_area:.2f}, "
                      f"↑{n_inc} ↓{n_dec} [{time.time()-t0:.0f}s]")

            config, prev_cpi, area = new_config, new_cpi, new_area
        trajectories.append(trajectory)

    elapsed = time.time() - t0
    if not cpis:
        return {"method": method_name, "configs": [], "cpis": [], "areas": [],
                "pareto_indices": [], "hv_curve": [], "ref_point": (1, 1),
                "trajectories": [], "wall_time": elapsed}
    pts = list(zip(cpis, areas))
    pidx = compute_pareto_front(pts)
    ref = (max(cpis) * 1.1, max(areas) * 1.1)
    if verbose:
        from area_model import compute_area as _ca
        _nf = getattr(_ca, "_fail_count", 0); _ca._fail_count = 0
        print(f"\n  Total: {len(cpis)} evals in {elapsed:.0f}s | "
              f"Pareto={len(pidx)} | McPAT_fails={_nf}")
    return {"method": method_name, "configs": configs, "cpis": cpis, "areas": areas,
            "pareto_indices": pidx.tolist(),
            "hv_curve": compute_hypervolume_curve(cpis, areas, ref),
            "ref_point": ref, "trajectories": trajectories, "wall_time": elapsed}


def _run_penetration_search(gem_predictor, area_fn, reclaim_threshold=0.02,
                              n_starts=30, max_steps=30, seed=42, verbose=True):
    """Search loop for penetration-guided method."""
    rng = np.random.RandomState(seed)
    configs, cpis, areas = [], [], []
    trajectories = []
    t0 = time.time()

    if verbose:
        print(f"\n{'='*60}\ngem_penetration (starts={n_starts}, steps={max_steps})\n{'='*60}")

    for si in range(n_starts):
        config = sample_random_config(rng)
        cpi = gem_predictor.predict_cpi(config)
        try:
            area = area_fn(config)
        except Exception:
            continue
        configs.append(config.copy()); cpis.append(cpi); areas.append(area)

        trajectory = [{"step": 0, "cpi": cpi, "area": area, "config": config.copy()}]

        if verbose:
            print(f"\n  Start {si+1}/{n_starts}: CPI={cpi:.4f}, Area={area:.2f}")

        prev_cpi, stagnant = cpi, 0
        for step in range(1, max_steps + 1):
            new_config, changes, info = _penetration_step(
                config, gem_predictor, reclaim_threshold,
                verbose=(verbose and step <= 3))

            if not changes:
                break

            new_cpi = gem_predictor.predict_cpi(new_config)
            try:
                new_area = area_fn(new_config)
            except Exception:
                break
            configs.append(new_config.copy()); cpis.append(new_cpi); areas.append(new_area)

            n_inc = sum(1 for v in changes.values() if v["action"] == "increase")
            n_rec = sum(1 for v in changes.values() if v["action"] == "reclaim")

            traj_entry = {"step": step, "cpi": new_cpi, "area": new_area,
                          "config": new_config.copy(), "changes": changes}
            traj_entry["mscs"] = info.get("mscs")
            traj_entry["penetrations"] = info.get("penetrations")
            traj_entry["target"] = info.get("target")
            trajectory.append(traj_entry)

            imp = abs(new_cpi - prev_cpi) / (abs(prev_cpi) + 1e-8)
            if imp < 0.001:
                stagnant += 1
                if stagnant >= 3: break
            else:
                stagnant = 0

            if verbose and step % 5 == 0:
                pen = info["penetrations"]
                print(f"    Step {step}: CPI={new_cpi:.4f}, Area={new_area:.2f}, "
                      f"↑{n_inc} ↓{n_rec}, target={info['target']} "
                      f"pen=[{pen['FP']:.3f},{pen['BP']:.3f},{pen['SP']:.3f}] "
                      f"[{time.time()-t0:.0f}s]")

            config, prev_cpi = new_config, new_cpi
        trajectories.append(trajectory)

    elapsed = time.time() - t0
    if not cpis:
        return {"method": "gem_penetration", "configs": [], "cpis": [], "areas": [],
                "pareto_indices": [], "hv_curve": [], "ref_point": (1, 1),
                "trajectories": [], "wall_time": elapsed}
    pts = list(zip(cpis, areas))
    pidx = compute_pareto_front(pts)
    ref = (max(cpis) * 1.1, max(areas) * 1.1)
    if verbose:
        from area_model import compute_area as _ca
        _nf = getattr(_ca, "_fail_count", 0); _ca._fail_count = 0
        print(f"\n  Total: {len(cpis)} evals in {elapsed:.0f}s | "
              f"Pareto={len(pidx)} | McPAT_fails={_nf}")
    return {"method": "gem_penetration", "configs": configs, "cpis": cpis, "areas": areas,
            "pareto_indices": pidx.tolist(),
            "hv_curve": compute_hypervolume_curve(cpis, areas, ref),
            "ref_point": ref, "trajectories": trajectories, "wall_time": elapsed}


# ============================================================
# Method Runners
# ============================================================

def run_bo_concorde(concorde_predictor, area_fn, budget=500, n_initial=30,
                     seed=42, verbose=True):
    """Method 1: Concorde + Bayesian Optimization."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel

    rng = np.random.RandomState(seed)
    param_names = list(ALL_PARAMS.keys())
    param_ranges = {n: (min(get_param_values(n)), max(get_param_values(n)))
                    for n in param_names}
    def to_vec(c):
        return np.array([(c[n] - param_ranges[n][0]) /
                         (param_ranges[n][1] - param_ranges[n][0] + 1e-8)
                         for n in param_names])
    def safe_area(c):
        try: return area_fn(c)
        except Exception: return None

    configs, cpis, areas, vecs = [], [], [], []
    t0 = time.time()
    if verbose:
        print(f"\n{'='*60}\nConcorde + BO (budget={budget})\n{'='*60}")

    attempts = 0
    while len(configs) < n_initial and attempts < n_initial * 3:
        attempts += 1
        c = sample_random_config(rng)
        area = safe_area(c)
        if area is None: continue
        cpi = concorde_predictor.predict_cpi(c)
        configs.append(c); cpis.append(cpi); areas.append(area); vecs.append(to_vec(c))
        if verbose and len(configs) % 10 == 0:
            print(f"  Init {len(configs)}/{n_initial} [{time.time()-t0:.0f}s]")

    for it in range(n_initial, budget):
        X = np.array(vecs)
        gp_c = GaussianProcessRegressor(kernel=ConstantKernel()*Matern(nu=2.5),
                                         n_restarts_optimizer=2, normalize_y=True,
                                         random_state=seed).fit(X, np.array(cpis))
        gp_a = GaussianProcessRegressor(kernel=ConstantKernel()*Matern(nu=2.5),
                                         n_restarts_optimizer=2, normalize_y=True,
                                         random_state=seed).fit(X, np.array(areas))
        pts = list(zip(cpis, areas))
        pidx = compute_pareto_front(pts)
        ref = (max(cpis)*1.1, max(areas)*1.1)
        cur_hv = compute_hypervolume_2d([pts[j] for j in pidx], ref)

        best_score, best_c = -np.inf, None
        for _ in range(500):
            cand = sample_random_config(rng)
            cv = to_vec(cand).reshape(1, -1)
            pc, sc = gp_c.predict(cv, return_std=True)
            pa, sa = gp_a.predict(cv, return_std=True)
            test_pts = [pts[j] for j in pidx] + [(pc[0]-1.5*sc[0], pa[0]-1.5*sa[0])]
            ti = compute_pareto_front(test_pts)
            score = compute_hypervolume_2d([test_pts[j] for j in ti], ref) - cur_hv + 0.1*(sc[0]+sa[0])
            if score > best_score:
                best_score, best_c = score, cand

        area = safe_area(best_c)
        if area is None: continue
        cpi = concorde_predictor.predict_cpi(best_c)
        configs.append(best_c); cpis.append(cpi); areas.append(area); vecs.append(to_vec(best_c))
        if verbose and (it+1) % 50 == 0:
            pts = list(zip(cpis, areas))
            pidx = compute_pareto_front(pts)
            hv = compute_hypervolume_2d([pts[j] for j in pidx], ref)
            print(f"  Iter {it+1}/{budget} | Pareto={len(pidx)} | HV={hv:.4f} [{time.time()-t0:.0f}s]")

    elapsed = time.time() - t0
    pts = list(zip(cpis, areas))
    pidx = compute_pareto_front(pts)
    ref = (max(cpis)*1.1, max(areas)*1.1)
    if verbose:
        print(f"\n  Total: {len(cpis)} evals in {elapsed:.0f}s | Pareto={len(pidx)}")
    return {"method": "concorde_bo", "configs": configs, "cpis": cpis, "areas": areas,
            "pareto_indices": pidx.tolist(),
            "hv_curve": compute_hypervolume_curve(cpis, areas, ref),
            "ref_point": ref, "wall_time": elapsed}


def run_bo_gem(gem_predictor, area_fn, budget=500, n_initial=30,
                seed=42, verbose=True, refine_fn=None, method_name="gem_bo"):
    """
    GEM + Bayesian Optimization.
    refine_fn: optional local search after each BO pick.
      signature: refine_fn(config, gem_predictor, area_fn) -> refined_config
      If None, pure BO (no local refinement).
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel

    rng = np.random.RandomState(seed)
    param_names = list(ALL_PARAMS.keys())
    param_ranges = {n: (min(get_param_values(n)), max(get_param_values(n)))
                    for n in param_names}
    def to_vec(c):
        return np.array([(c[n] - param_ranges[n][0]) /
                         (param_ranges[n][1] - param_ranges[n][0] + 1e-8)
                         for n in param_names])
    def safe_area(c):
        try: return area_fn(c)
        except Exception: return None

    configs, cpis, areas, vecs = [], [], [], []
    t0 = time.time()
    label = method_name.replace("_", " ").upper()
    if verbose:
        print(f"\n{'='*60}\n{label} (budget={budget})\n{'='*60}")

    attempts = 0
    while len(configs) < n_initial and attempts < n_initial * 3:
        attempts += 1
        c = sample_random_config(rng)
        # No refine on initial points — too expensive, BO will explore
        area = safe_area(c)
        if area is None: continue
        cpi = gem_predictor.predict_cpi(c)
        configs.append(c); cpis.append(cpi); areas.append(area); vecs.append(to_vec(c))
        if verbose and len(configs) % 10 == 0:
            print(f"  Init {len(configs)}/{n_initial} [{time.time()-t0:.0f}s]")

    for it in range(n_initial, budget):
        X = np.array(vecs)
        gp_c = GaussianProcessRegressor(kernel=ConstantKernel()*Matern(nu=2.5),
                                         n_restarts_optimizer=2, normalize_y=True,
                                         random_state=seed).fit(X, np.array(cpis))
        gp_a = GaussianProcessRegressor(kernel=ConstantKernel()*Matern(nu=2.5),
                                         n_restarts_optimizer=2, normalize_y=True,
                                         random_state=seed).fit(X, np.array(areas))
        pts = list(zip(cpis, areas))
        pidx = compute_pareto_front(pts)
        ref = (max(cpis)*1.1, max(areas)*1.1)
        cur_hv = compute_hypervolume_2d([pts[j] for j in pidx], ref)

        best_score, best_c = -np.inf, None
        for _ in range(500):
            cand = sample_random_config(rng)
            cv = to_vec(cand).reshape(1, -1)
            pc, sc = gp_c.predict(cv, return_std=True)
            pa, sa = gp_a.predict(cv, return_std=True)
            test_pts = [pts[j] for j in pidx] + [(pc[0]-1.5*sc[0], pa[0]-1.5*sa[0])]
            ti = compute_pareto_front(test_pts)
            score = compute_hypervolume_2d([test_pts[j] for j in ti], ref) - cur_hv + 0.1*(sc[0]+sa[0])
            if score > best_score:
                best_score, best_c = score, cand

        # Local refinement
        if refine_fn is not None:
            best_c = refine_fn(best_c, gem_predictor, area_fn)

        area = safe_area(best_c)
        if area is None: continue
        cpi = gem_predictor.predict_cpi(best_c)
        configs.append(best_c); cpis.append(cpi); areas.append(area); vecs.append(to_vec(best_c))
        if verbose and (it+1) % 10 == 0:
            pts = list(zip(cpis, areas))
            pidx = compute_pareto_front(pts)
            hv = compute_hypervolume_2d([pts[j] for j in pidx], ref)
            print(f"  Iter {it+1}/{budget} | Pareto={len(pidx)} | HV={hv:.4f} [{time.time()-t0:.0f}s]")

    elapsed = time.time() - t0
    pts = list(zip(cpis, areas))
    pidx = compute_pareto_front(pts)
    ref = (max(cpis)*1.1, max(areas)*1.1)
    if verbose:
        print(f"\n  Total: {len(cpis)} evals in {elapsed:.0f}s | Pareto={len(pidx)}")
    return {"method": method_name, "configs": configs, "cpis": cpis, "areas": areas,
            "pareto_indices": pidx.tolist(),
            "hv_curve": compute_hypervolume_curve(cpis, areas, ref),
            "ref_point": ref, "wall_time": elapsed}


def _scalar_refine(config, gem_predictor, area_fn, n_steps=15):
    """Local refinement: scalar ΔCPI greedy + ΔCPI reclaim."""
    for _ in range(n_steps):
        cpi_grad, _ = gem_predictor.compute_cpi_gradient(config)
        new_config, changes = _scalar_gradient_step(config, cpi_grad, 0.01)
        if not changes:
            break
        config = new_config
    return config


def _mscs_refine(config, gem_predictor, area_fn, n_steps=15):
    """Local refinement: coordinated budget swap with MSCS PROTECT."""
    for _ in range(n_steps):
        new_config, changes, info = _coordinated_step(
            config, gem_predictor, area_fn)
        if not changes:
            break
        config = new_config
    return config


def _coordinated_step(config, gem_predictor, area_fn, verbose=False):
    """
    Coordinated budget reassignment with MSCS PROTECT.
    
    1. Identify dominant bottleneck type via MSCS (FP/BP/SP)
    2. Increase param that reduces dominant bottleneck most (CPI safety valve)
    3. Decrease param with least dominant impact, BUT respect MSCS PROTECT
    4. One step, approximately area-neutral
    """
    mscs = gem_predictor.predict_mscs(config)
    cpi_base = gem_predictor.predict_cpi(config)

    # Step 1: Identify dominant bottleneck type (4-stage average)
    fp_pressure = np.mean([mscs[i] for i in FP_INDICES])
    bp_pressure = np.mean([mscs[i] for i in BP_INDICES])
    sp_pressure = np.mean([mscs[i] for i in SP_INDICES])
    pressures = {"FP": fp_pressure, "BP": bp_pressure, "SP": sp_pressure}
    dominant = max(pressures, key=pressures.get)
    dom_indices = BOTTLENECK_INDICES[dominant]

    # Step 2: For each param, measure dominant bottleneck effect via MSCS only
    # Skip predict_cpi here — only call it for top candidates (expensive with 17 benchmarks)
    increase_candidates = []
    decrease_candidates = []

    mscs_cache = {}  # cache mscs predictions

    for name in GRADIENT_PARAMS:
        # Increase evaluation — need mscs for dominant bottleneck ranking
        new_val_inc, old_val_inc = _increase_by_stride(config, name)
        if new_val_inc is not None:
            c_inc = config.copy()
            c_inc[name] = new_val_inc
            c_inc = _project_to_valid(c_inc)
            mscs_inc = gem_predictor.predict_mscs(c_inc)
            mscs_cache[(name, "inc")] = (c_inc, mscs_inc)
            dom_reduction = np.mean([mscs[i] - mscs_inc[i] for i in dom_indices])
            increase_candidates.append({
                "name": name, "dom_reduction": dom_reduction,
                "new_val": new_val_inc, "old_val": old_val_inc,
            })

        # Decrease evaluation — only CPI cost (cheap), MSCS checked later for PROTECT
        new_val_dec, old_val_dec = _decrease_by_stride(config, name)
        if new_val_dec is not None:
            c_dec = config.copy()
            c_dec[name] = new_val_dec
            c_dec = _project_to_valid(c_dec)
            cpi_dec = gem_predictor.predict_cpi(c_dec)
            dcpi_cost = cpi_dec - cpi_base  # positive = CPI worsens
            decrease_candidates.append({
                "name": name, "dcpi_cost": dcpi_cost,
                "new_val": new_val_dec, "old_val": old_val_dec,
                "config_dec": c_dec,
            })

    if not increase_candidates:
        return config.copy(), {}, {"mode": "no_candidates"}

    # Step 3: Pick increase — sort by dom_reduction, CPI check only top-3
    increase_candidates.sort(key=lambda x: x["dom_reduction"], reverse=True)
    best_inc = None
    for c in increase_candidates[:3]:
        c_inc, _ = mscs_cache[(c["name"], "inc")]
        cpi_inc = gem_predictor.predict_cpi(c_inc)
        c["cpi_new"] = cpi_inc
        if cpi_inc < cpi_base:
            best_inc = c
            break
    if best_inc is None:
        return config.copy(), {}, {"mode": "no_improvement"}

    # Apply increase
    new_config = config.copy()
    new_config[best_inc["name"]] = best_inc["new_val"]
    new_config = _project_to_valid(new_config)
    changes = {
        best_inc["name"]: {
            "from": best_inc["old_val"], "to": best_inc["new_val"],
            "action": "increase",
            "reason": f"top {dominant} reducer, Δ={best_inc['dom_reduction']:.4f}",
        }
    }

    # Step 4: Pick decrease — lowest ΔCPI cost + MSCS PROTECT
    dec_cands = [d for d in decrease_candidates if d["name"] != best_inc["name"]]
    dec_cands.sort(key=lambda x: x["dcpi_cost"])  # smallest CPI cost first

    for best_dec in dec_cands:
        # Skip if removing hurts CPI too much
        if best_dec["dcpi_cost"] >= 0.05:
            break  # sorted, rest are worse

        # PROTECT check: compute MSCS only for this candidate (lazy)
        mscs_dec = gem_predictor.predict_mscs(best_dec["config_dec"])
        delta_dec = mscs_dec - mscs
        total_eff = _mscs_total_effect(delta_dec)
        if best_dec["dcpi_cost"] >= 0 and total_eff >= MSCS_RECLAIM_THRESHOLD:
            continue  # PROTECT: overlap-hidden value

        # CPI safety valve after increase
        c_test = new_config.copy()
        c_test[best_dec["name"]] = best_dec["new_val"]
        c_test = _project_to_valid(c_test)
        cpi_test = gem_predictor.predict_cpi(c_test)
        cpi_after_inc = gem_predictor.predict_cpi(new_config)
        if cpi_test <= cpi_after_inc * 1.005:
            new_config = c_test
            changes[best_dec["name"]] = {
                "from": best_dec["old_val"], "to": best_dec["new_val"],
                "action": "decrease",
                "reason": f"least CPI cost={best_dec['dcpi_cost']:.4f}, "
                          f"mscs_eff={total_eff:.4f}",
            }
            break  # one swap per step

    new_config = _project_to_valid(new_config)

    info = {
        "mode": "coordinated",
        "dominant": dominant,
        "pressures": pressures,
        "inc_param": best_inc["name"],
        "inc_reduction": best_inc["dom_reduction"],
    }

    if verbose:
        print(f"      [COORD] dom={dominant} "
              f"(FP={fp_pressure:.3f} BP={bp_pressure:.3f} SP={sp_pressure:.3f})")
        inc_str = f"+{best_inc['name']}({best_inc['old_val']}→{best_inc['new_val']})"
        dec_strs = [f"-{k}({v['from']}→{v['to']})" for k, v in changes.items()
                    if v['action'] == 'decrease']
        dec_str = ", ".join(dec_strs) if dec_strs else "none (all PROTECTED)"
        print(f"      Swap: {inc_str} | {dec_str}")

    return new_config, changes, info


def _coordinated_refine(config, gem_predictor, area_fn, n_steps=15):
    """Local refinement: coordinated budget reassignment."""
    for _ in range(n_steps):
        new_config, changes, info = _coordinated_step(
            config, gem_predictor, area_fn)
        if not changes:
            break
        config = new_config
    return config


def run_bo_then_refine(gem_predictor, area_fn, budget=500, n_initial=30,
                        seed=42, verbose=True, refine_fn=None, method_name="gem_bo_refine",
                        refine_steps=15):
    """
    Two-phase DSE:
      Phase 1: Pure BO exploration (GP not disturbed)
      Phase 2: Refine each Pareto front point with scalar/MSCS local search

    This avoids corrupting the GP model with refined points.
    """
    # Phase 1: Pure BO
    if verbose:
        print(f"\n  Phase 1: Pure BO exploration (budget={budget})...")
    bo_results = run_bo_gem(gem_predictor, area_fn, budget=budget,
                             n_initial=n_initial, seed=seed, verbose=verbose,
                             refine_fn=None, method_name="gem_bo")

    all_configs = list(bo_results["configs"])
    all_cpis = list(bo_results["cpis"])
    all_areas = list(bo_results["areas"])

    if refine_fn is None:
        bo_results["method"] = method_name
        return bo_results

    # Phase 2: Refine best points (Pareto + near-Pareto)
    # Sort all points by CPI, take top 50 (covers diverse area range)
    sorted_idx = np.argsort(all_cpis)
    refine_idx = sorted_idx[:min(50, len(sorted_idx))]
    # Also include all Pareto points
    pts = list(zip(all_cpis, all_areas))
    pidx = set(compute_pareto_front(pts).tolist())
    refine_idx = list(set(refine_idx.tolist()) | pidx)
    n_refine = len(refine_idx)

    if verbose:
        print(f"\n  Phase 2: Refining {n_refine} points ({refine_steps} steps each)...")

    def safe_area(c):
        try: return area_fn(c)
        except Exception: return None

    t0 = time.time()
    n_refined = 0
    for i, pi in enumerate(refine_idx):
        config = all_configs[pi]
        refined = refine_fn(config, gem_predictor, area_fn, n_steps=refine_steps)
        area = safe_area(refined)
        if area is None:
            continue
        cpi = gem_predictor.predict_cpi(refined)
        all_configs.append(refined)
        all_cpis.append(cpi)
        all_areas.append(area)
        n_refined += 1

        if verbose and (i + 1) % 10 == 0:
            print(f"    Refined {i+1}/{n_refine} [{time.time()-t0:.0f}s]")

    if verbose:
        print(f"    Done: {n_refined} refined points added [{time.time()-t0:.0f}s]")

    # Recompute Pareto
    pts = list(zip(all_cpis, all_areas))
    pidx = compute_pareto_front(pts)
    ref = (max(all_cpis) * 1.1, max(all_areas) * 1.1)

    elapsed = bo_results.get("wall_time", 0) + (time.time() - t0)

    return {
        "method": method_name,
        "configs": all_configs,
        "cpis": all_cpis,
        "areas": all_areas,
        "pareto_indices": pidx.tolist(),
        "hv_curve": compute_hypervolume_curve(all_cpis, all_areas, ref),
        "ref_point": ref,
        "wall_time": elapsed,
        "n_bo_evals": len(bo_results["cpis"]),
        "n_refined": n_refined,
    }


def run_concorde_scalar(concorde_predictor, area_fn, threshold=0.01,
                         n_starts=30, max_steps=30, seed=42, verbose=True):
    """Method 2: Concorde scalar gradient."""
    def step_fn(config, predictor, override_no_increase=False):
        cpi_grad, _ = predictor.compute_cpi_gradient(config)
        k = 0 if override_no_increase else 1
        new_config, changes = _scalar_gradient_step(config, cpi_grad, threshold,
                                                     top_k_increase=k)
        return new_config, changes, {}
    return _run_gradient_search(step_fn, concorde_predictor, area_fn,
                                "concorde_scalar_grad", n_starts, max_steps, seed, verbose)


def run_gem_scalar(gem_predictor, area_fn, threshold=0.01,
                    n_starts=30, max_steps=30, seed=42, no_reclaim=False, verbose=True):
    """Method 3: GEM scalar gradient. no_reclaim=True disables decrease."""
    dec_thresh = 0.0 if no_reclaim else 0.1
    method_name = "gem_scalar_noreclaim" if no_reclaim else "gem_scalar_grad"
    def step_fn(config, predictor, override_no_increase=False):
        cpi_grad, _ = predictor.compute_cpi_gradient(config)
        k = 0 if override_no_increase else 1
        new_config, changes = _scalar_gradient_step(config, cpi_grad, threshold,
                                                     decrease_threshold=dec_thresh,
                                                     top_k_increase=k)
        return new_config, changes, {}
    return _run_gradient_search(step_fn, gem_predictor, area_fn,
                                method_name, n_starts, max_steps, seed, verbose)


def run_gem_penetration(gem_predictor, area_fn, reclaim_threshold=0.02,
                          n_starts=30, max_steps=30, seed=42, verbose=True):
    """Method 4: GEM Penetration-Guided DSE (our contribution)."""
    return _run_penetration_search(gem_predictor, area_fn, reclaim_threshold,
                                    n_starts, max_steps, seed, verbose)


# ============================================================
# Method 5: MSCS-Attributed Greedy Search (core contribution)
# ============================================================

FP_INDICES = [1, 5, 9, 13]
BP_INDICES = [2, 6, 10, 14]
SP_INDICES = [3, 7, 11, 15]

BOTTLENECK_INDICES = {"FP": FP_INDICES, "BP": BP_INDICES, "SP": SP_INDICES}


def _compute_area_cost(config, name, area_fn):
    """Compute area increase for one step of parameter increase."""
    vals = get_param_values(name)
    current = config[name]
    if isinstance(ALL_PARAMS.get(name), list):
        idx = vals.index(current) if current in vals else 0
        if idx >= len(vals) - 1:
            return None, None  # already at max
        new_val = vals[idx + 1]
    else:
        hi = max(vals)
        if current >= hi:
            return None, None
        new_val = current + 1

    c_new = config.copy()
    c_new[name] = new_val
    c_new = _project_to_valid(c_new)
    try:
        area_new = area_fn(c_new)
        area_old = area_fn(config)
        return new_val, max(area_new - area_old, 1e-6)  # avoid div by zero
    except Exception:
        return None, None


# Per-parameter DSE stride (how much to increase per step)
PARAM_STRIDE = {
    "fetchWidth": 1, "decodeWidth": 1, "dispatchWidth": 1, "commitWidth": 1,
    "numROBEntries": 16, "numIQEntries": 8, "LQEntries": 8, "SQEntries": 8,
    "IntALU": 1, "IntMultDiv": 1, "FP_ALU": 1, "FP_MultDiv": 1, "RdWrPort": 1,
    # List params: stride = jump to next value (handled in _increase_by_stride)
    "l1_size": None, "l1d_assoc": None, "l1i_assoc": None,
}


def _increase_by_stride(config, name):
    """Increase param by one stride. Returns (new_val, old_val) or (None, None)."""
    vals = get_param_values(name)
    current = config[name]
    stride = PARAM_STRIDE.get(name)

    if isinstance(ALL_PARAMS.get(name), list):
        # List param: jump to next value
        idx = vals.index(current) if current in vals else 0
        if idx < len(vals) - 1:
            return vals[idx + 1], current
        return None, None
    else:
        hi = max(vals)
        new_val = min(hi, current + (stride or 1))
        if new_val == current:
            return None, None
        return new_val, current


def _decrease_by_stride(config, name):
    """Decrease param by one stride. Returns (new_val, old_val) or (None, None)."""
    vals = get_param_values(name)
    current = config[name]
    stride = PARAM_STRIDE.get(name)

    if isinstance(ALL_PARAMS.get(name), list):
        idx = vals.index(current) if current in vals else 0
        if idx > 0:
            return vals[idx - 1], current
        return None, None
    else:
        lo = min(vals)
        new_val = max(lo, current - (stride or 1))
        if new_val == current:
            return None, None
        return new_val, current


SINGLE_STAGE_INDICES = {
    "FP": [13],   # commit stage only
    "BP": [14],
    "SP": [15],
}


MSCS_12_INDICES = FP_INDICES + BP_INDICES + SP_INDICES  # 12-dim bottleneck vector
DECOUPLE_THRESHOLD = 0.01  # ΔCPI below this = plateau → try decouple
MSCS_RECLAIM_THRESHOLD = 0.02  # MSCS total effect threshold for PROTECT


def _cancel_score(delta_mscs):
    """
    Cross-type cancellation score from 16-dim ΔMSCS.
    High score = parameter changes multiple bottleneck types in opposite directions
    → decoupling overlapping bottlenecks.
    """
    type_effects = []
    for type_indices in [FP_INDICES, BP_INDICES, SP_INDICES]:
        vals = delta_mscs[type_indices]
        max_abs_idx = np.argmax(np.abs(vals))
        type_effects.append(vals[max_abs_idx])
    abs_sum = sum(abs(e) for e in type_effects)
    sum_abs = abs(sum(type_effects))
    return abs_sum - sum_abs, type_effects


def _mscs_total_effect(delta_mscs):
    """Total effect across all 16 dims (L1 norm). For PROTECT detection."""
    return np.sum(np.abs(delta_mscs))


def _attribution_step(config, gem_predictor, area_fn, reclaim_threshold=0.05,
                       single_stage=False, verbose=False):
    """
    MSCS-Guided DSE Step (finite-diff, decouple/protect):

    INCREASE:
      Normal mode (∃ ΔCPI > DECOUPLE_THRESHOLD): argmax ΔCPI (same as scalar)
      Decouple mode (all ΔCPI < threshold): argmax cancel_score
        → break overlapping bottlenecks to expose clean gradient landscape

    DECREASE:
      Normal mode: ΔCPI proposes, MSCS PROTECT guards
        - ΔCPI < 0 → RECLAIM* (removing improves CPI, always reclaim)
        - 0 ≤ ΔCPI < threshold, MSCS effect high → PROTECT
        - 0 ≤ ΔCPI < threshold, MSCS effect low → RECLAIM
      Decouple mode: only RECLAIM* (investment phase, no regular reclaim)
    """

    cpi_base = gem_predictor.predict_cpi(config)

    # --- Phase 1: Increase ---
    # Evaluate all params by ΔCPI (finite diff)
    cpi_candidates = []
    for name in GRADIENT_PARAMS:
        new_val, old_val = _increase_by_stride(config, name)
        if new_val is None:
            continue
        c_new = config.copy()
        c_new[name] = new_val
        c_new = _project_to_valid(c_new)
        cpi_new = gem_predictor.predict_cpi(c_new)
        dcpi = cpi_base - cpi_new
        if dcpi <= 0:
            continue
        cpi_candidates.append({
            "name": name, "delta_cpi": dcpi,
            "new_val": new_val, "old_val": old_val,
            "cpi_new": cpi_new,
        })

    best_dcpi = max((c["delta_cpi"] for c in cpi_candidates), default=0)
    mode = "normal"

    if best_dcpi >= DECOUPLE_THRESHOLD:
        # Normal mode: argmax ΔCPI
        cpi_candidates.sort(key=lambda x: x["delta_cpi"], reverse=True)
        best = cpi_candidates[0]
    elif cpi_candidates:
        # Decouple mode: argmax cancel_score
        mode = "decouple"
        decouple_candidates = []
        for c in cpi_candidates:
            c_new = config.copy()
            c_new[c["name"]] = c["new_val"]
            c_new = _project_to_valid(c_new)
            mscs_old = gem_predictor.predict_mscs(config)
            mscs_new = gem_predictor.predict_mscs(c_new)
            delta = mscs_old - mscs_new  # positive = pressure reduced
            cscore, type_effs = _cancel_score(delta)
            c["cancel_score"] = cscore
            decouple_candidates.append(c)

        if not decouple_candidates:
            return config.copy(), {}, {"mode": "no_candidates"}
        decouple_candidates.sort(key=lambda x: x["cancel_score"], reverse=True)
        best = decouple_candidates[0]
    else:
        return config.copy(), {}, {"mode": "no_candidates"}

    # Apply increase
    new_config = config.copy()
    new_config[best["name"]] = best["new_val"]
    new_config = _project_to_valid(new_config)
    changes = {
        best["name"]: {
            "from": best["old_val"], "to": best["new_val"],
            "action": "increase",
            "reason": f"{'decouple' if mode == 'decouple' else 'argmax_dcpi'}, ΔCPI={best['delta_cpi']:.4f}",
        }
    }

    # --- Phase 2: Reclaim ---
    if mode == "decouple":
        # Decouple mode: only reclaim actively harmful params (ΔCPI < 0)
        cpi_after_inc = gem_predictor.predict_cpi(new_config)
        for name in GRADIENT_PARAMS:
            if name in changes:
                continue
            dec_val, old_val = _decrease_by_stride(new_config, name)
            if dec_val is None:
                continue
            c_dec = new_config.copy()
            c_dec[name] = dec_val
            c_dec = _project_to_valid(c_dec)
            cpi_dec = gem_predictor.predict_cpi(c_dec)
            if cpi_dec < cpi_after_inc:  # removing improves CPI
                if cpi_dec <= cpi_after_inc * 1.005:
                    new_config = c_dec
                    cpi_after_inc = cpi_dec
                    changes[name] = {
                        "from": old_val, "to": dec_val,
                        "action": "reclaim",
                        "reason": "harmful (ΔCPI<0)",
                    }
    else:
        # Normal mode: ΔCPI proposes, MSCS PROTECT guards
        cpi_after_inc = gem_predictor.predict_cpi(new_config)
        config_pre_reclaim = new_config.copy()

        for name in GRADIENT_PARAMS:
            if name in changes:
                continue
            dec_val, old_val = _decrease_by_stride(new_config, name)
            if dec_val is None:
                continue
            c_dec = new_config.copy()
            c_dec[name] = dec_val
            c_dec = _project_to_valid(c_dec)
            cpi_dec = gem_predictor.predict_cpi(c_dec)
            dcpi_cost = cpi_dec - cpi_after_inc

            if dcpi_cost >= reclaim_threshold:
                continue  # KEEP

            # ΔCPI < 0: always reclaim
            if dcpi_cost < 0:
                if cpi_dec <= cpi_after_inc * 1.005:
                    new_config = c_dec
                    cpi_after_inc = cpi_dec
                    changes[name] = {
                        "from": old_val, "to": dec_val,
                        "action": "reclaim",
                        "reason": "harmful (ΔCPI<0)",
                    }
                continue

            # 0 <= ΔCPI < threshold: MSCS decides
            mscs_old = gem_predictor.predict_mscs(new_config)
            mscs_dec = gem_predictor.predict_mscs(c_dec)
            delta_dec = mscs_dec - mscs_old
            total_eff = _mscs_total_effect(delta_dec)

            if total_eff >= MSCS_RECLAIM_THRESHOLD:
                continue  # PROTECT

            # Both agree: reclaim
            if cpi_dec <= cpi_after_inc * 1.005:
                new_config = c_dec
                cpi_after_inc = cpi_dec
                changes[name] = {
                    "from": old_val, "to": dec_val,
                    "action": "reclaim",
                    "reason": f"mscs_eff={total_eff:.4f}<{MSCS_RECLAIM_THRESHOLD}",
                }

        # Global safety valve
        cpi_final = gem_predictor.predict_cpi(new_config)
        cpi_pre = gem_predictor.predict_cpi(config_pre_reclaim)
        if cpi_final > cpi_pre * 1.005:
            new_config = config_pre_reclaim
            changes = {k: v for k, v in changes.items() if v["action"] != "reclaim"}

    new_config = _project_to_valid(new_config)

    info = {
        "mode": mode,
        "best_dcpi": best_dcpi,
        "n_candidates": len(cpi_candidates),
    }

    if verbose:
        print(f"      [{mode.upper()}] best={best['name']} ΔCPI={best['delta_cpi']:.4f}")
        recs = [f"{k}({v['from']}→{v['to']})" for k, v in changes.items()
                if v['action'] == 'reclaim']
        if recs:
            print(f"      Reclaim: {', '.join(recs)}")

    return new_config, changes, info


def run_gem_attribution(gem_predictor, area_fn, reclaim_threshold=0.05,
                          n_starts=30, max_steps=30, seed=42, single_stage=False,
                          verbose=True):
    """
    Method 5: MSCS-Attributed Greedy Search.
    single_stage=True: ablation using only commit stage (single-stage CPI stack baseline).
    """
    method_name = "gem_attribution_single" if single_stage else "gem_attribution"
    rng = np.random.RandomState(seed)
    configs, cpis, areas = [], [], []
    trajectories = []
    t0 = time.time()

    if verbose:
        tag = "single-stage" if single_stage else "multi-stage"
        print(f"\n{'='*60}\n{method_name} [{tag}] (starts={n_starts}, steps={max_steps})\n{'='*60}")

    for si in range(n_starts):
        # Structured starts: 1/3 small, 1/3 random, 1/3 large
        # Same algorithm for all — different starting area covers different Pareto regions
        if si < n_starts // 3:
            # Small config: params biased toward lower half
            config = sample_random_config(rng)
            for name in config:
                vals = get_param_values(name)
                lo, hi = min(vals), max(vals)
                mid = (lo + hi) / 2
                if config[name] > mid:
                    if isinstance(ALL_PARAMS.get(name), list):
                        lower_vals = [v for v in vals if v <= mid]
                        if lower_vals:
                            config[name] = rng.choice(lower_vals)
                    else:
                        config[name] = rng.randint(lo, max(lo + 1, int(mid)))
            config = _project_to_valid(config)
        elif si < 2 * n_starts // 3:
            # Random config (as before)
            config = sample_random_config(rng)
        else:
            # Large config: params biased toward upper half
            config = sample_random_config(rng)
            for name in config:
                vals = get_param_values(name)
                lo, hi = min(vals), max(vals)
                mid = (lo + hi) / 2
                if config[name] < mid:
                    if isinstance(ALL_PARAMS.get(name), list):
                        upper_vals = [v for v in vals if v >= mid]
                        if upper_vals:
                            config[name] = rng.choice(upper_vals)
                    else:
                        config[name] = rng.randint(int(mid), hi + 1)
            config = _project_to_valid(config)

        cpi = gem_predictor.predict_cpi(config)
        try:
            area = area_fn(config)
        except Exception:
            continue
        configs.append(config.copy()); cpis.append(cpi); areas.append(area)

        trajectory = [{"step": 0, "cpi": cpi, "area": area, "config": config.copy()}]

        if verbose:
            print(f"\n  Start {si+1}/{n_starts}: CPI={cpi:.4f}, Area={area:.2f}")

        prev_cpi, stagnant = cpi, 0
        decouple_count = 0
        MAX_DECOUPLE = 5  # max consecutive decouple steps before stopping
        for step in range(1, max_steps + 1):
            new_config, changes, info = _attribution_step(
                config, gem_predictor, area_fn, reclaim_threshold,
                single_stage=single_stage,
                verbose=(verbose and step <= 2))

            if not changes:
                if verbose: print(f"    Step {step}: no changes, stopping")
                break

            # Track decouple steps
            if info.get("mode") == "decouple":
                decouple_count += 1
                if decouple_count >= MAX_DECOUPLE:
                    if verbose: print(f"    Step {step}: max decouple reached, stopping")
                    break
            else:
                decouple_count = 0

            new_cpi = gem_predictor.predict_cpi(new_config)
            try:
                new_area = area_fn(new_config)
            except Exception:
                break
            configs.append(new_config.copy()); cpis.append(new_cpi); areas.append(new_area)

            n_inc = sum(1 for v in changes.values() if v["action"] == "increase")
            n_rec = sum(1 for v in changes.values() if v["action"] == "reclaim")

            traj_entry = {"step": step, "cpi": new_cpi, "area": new_area,
                          "config": new_config.copy(), "changes": changes}
            traj_entry.update(info)
            trajectory.append(traj_entry)

            imp = abs(new_cpi - prev_cpi) / (abs(prev_cpi) + 1e-8)
            if imp < 0.001:
                # Don't count decouple steps as stagnant — they're investment steps
                if info.get("mode") == "decouple":
                    stagnant = 0  # reset: decouple is intentional
                else:
                    stagnant += 1
                    if stagnant >= 3:
                        if verbose: print(f"    Step {step}: stagnant")
                        break
            else:
                stagnant = 0

            if verbose and step % 3 == 0:
                mode_tag = info.get("mode", "?")
                print(f"    Step {step}: CPI={new_cpi:.4f}, Area={new_area:.2f}, "
                      f"↑{n_inc} ↓{n_rec}, mode={mode_tag} "
                      f"[{time.time()-t0:.0f}s]")

            config, prev_cpi = new_config, new_cpi
        trajectories.append(trajectory)

    elapsed = time.time() - t0
    if not cpis:
        return {"method": method_name, "configs": [], "cpis": [], "areas": [],
                "pareto_indices": [], "hv_curve": [], "ref_point": (1, 1),
                "trajectories": [], "wall_time": elapsed}
    pts = list(zip(cpis, areas))
    pidx = compute_pareto_front(pts)
    ref = (max(cpis) * 1.1, max(areas) * 1.1)
    if verbose:
        from area_model import compute_area as _ca
        _nf = getattr(_ca, "_fail_count", 0); _ca._fail_count = 0
        print(f"\n  Total: {len(cpis)} evals in {elapsed:.0f}s | "
              f"Pareto={len(pidx)} | McPAT_fails={_nf}")
    return {"method": method_name, "configs": configs, "cpis": cpis, "areas": areas,
            "pareto_indices": pidx.tolist(),
            "hv_curve": compute_hypervolume_curve(cpis, areas, ref),
            "ref_point": ref, "trajectories": trajectories, "wall_time": elapsed}


def run_gem_coordinated(gem_predictor, area_fn,
                         n_starts=30, max_steps=30, seed=42, verbose=True):
    """
    Method 6: MSCS Coordinated Budget Reassignment.
    Each step: increase bottleneck resource + decrease non-bottleneck resource (area-neutral).
    """
    method_name = "gem_coordinated"
    rng = np.random.RandomState(seed)
    configs, cpis, areas = [], [], []
    trajectories = []
    t0 = time.time()

    if verbose:
        print(f"\n{'='*60}\n{method_name} (starts={n_starts}, steps={max_steps})\n{'='*60}")

    for si in range(n_starts):
        config = sample_random_config(rng)
        cpi = gem_predictor.predict_cpi(config)
        try:
            area = area_fn(config)
        except Exception:
            continue
        configs.append(config.copy()); cpis.append(cpi); areas.append(area)

        traj = [{"step": 0, "cpi": cpi, "area": area}]
        prev_cpi, stagnant = cpi, 0

        for step in range(1, max_steps + 1):
            new_config, changes, info = _coordinated_step(
                config, gem_predictor, area_fn,
                verbose=(verbose and si == 0 and step <= 3))

            if not changes:
                break

            try:
                new_area = area_fn(new_config)
            except Exception:
                break

            new_cpi = gem_predictor.predict_cpi(new_config)
            configs.append(new_config.copy())
            cpis.append(new_cpi)
            areas.append(new_area)
            config = new_config

            traj.append({"step": step, "cpi": new_cpi, "area": new_area,
                          "changes": {k: v for k, v in changes.items()}})

            imp = abs(new_cpi - prev_cpi) / (abs(prev_cpi) + 1e-8)
            if imp < 0.001:
                stagnant += 1
                if stagnant >= 5:
                    break
            else:
                stagnant = 0
            prev_cpi = new_cpi

        trajectories.append(traj)
        if verbose and (si + 1) % 5 == 0:
            print(f"  Start {si+1}/{n_starts}: CPI={cpi:.4f}→{prev_cpi:.4f} "
                  f"Area={traj[0]['area']:.1f}→{traj[-1]['area']:.1f} "
                  f"({len(traj)} steps) [{time.time()-t0:.0f}s]")

    elapsed = time.time() - t0
    if not cpis:
        return {"method": method_name, "configs": [], "cpis": [], "areas": [],
                "pareto_indices": [], "hv_curve": [], "ref_point": (1, 1),
                "trajectories": [], "wall_time": elapsed}
    pts = list(zip(cpis, areas))
    pidx = compute_pareto_front(pts)
    ref = (max(cpis) * 1.1, max(areas) * 1.1)
    if verbose:
        print(f"\n  Total: {len(cpis)} evals in {elapsed:.0f}s | Pareto={len(pidx)}")
    return {"method": method_name, "configs": configs, "cpis": cpis, "areas": areas,
            "pareto_indices": pidx.tolist(),
            "hv_curve": compute_hypervolume_curve(cpis, areas, ref),
            "ref_point": ref, "trajectories": trajectories, "wall_time": elapsed}


# ============================================================
# Comparison & Saving
# ============================================================

def compare_methods(results_list, ref_point=None):
    if ref_point is None:
        max_cpi = max(max(r["cpis"]) for r in results_list if r["cpis"])
        max_area = max(max(r["areas"]) for r in results_list if r["areas"])
        ref_point = (max_cpi * 1.1, max_area * 1.1)
    comparison = {"ref_point": ref_point, "methods": []}
    for r in results_list:
        if not r["cpis"]: continue
        hv_curve = compute_hypervolume_curve(r["cpis"], r["areas"], ref_point)
        pts = list(zip(r["cpis"], r["areas"]))
        pidx = compute_pareto_front(pts)
        comparison["methods"].append({
            "name": r["method"], "n_evals": len(r["cpis"]),
            "pareto_size": len(pidx),
            "final_hv": compute_hypervolume_2d([pts[j] for j in pidx], ref_point),
            "hv_curve": hv_curve,
            "pareto_cpis": [pts[j][0] for j in pidx],
            "pareto_areas": [pts[j][1] for j in pidx],
            "wall_time": r.get("wall_time")})
    return comparison


def save_results(results, output_dir, prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{prefix}{results['method']}_results.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {filepath}")