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
        print(f"  GEMPredictor [{workload}]: {total} total, "
              f"CPI={len(self.cpi_indices)}, grad={len(self.grad_indices)}")

    def _get_features(self, config, indices):
        all_feats = self.dse_engine.get_features(config, workload=self.workload)
        batch = np.stack([all_feats[i][:GEM_INPUT_DIM] for i in indices])
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
                seed=42, verbose=True):
    """GEM + Bayesian Optimization (same BO, better evaluator)."""
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
        print(f"\n{'='*60}\nGEM + BO (budget={budget})\n{'='*60}")

    attempts = 0
    while len(configs) < n_initial and attempts < n_initial * 3:
        attempts += 1
        c = sample_random_config(rng)
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

        area = safe_area(best_c)
        if area is None: continue
        cpi = gem_predictor.predict_cpi(best_c)
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
    return {"method": "gem_bo", "configs": configs, "cpis": cpis, "areas": areas,
            "pareto_indices": pidx.tolist(),
            "hv_curve": compute_hypervolume_curve(cpis, areas, ref),
            "ref_point": ref, "wall_time": elapsed}


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
                    n_starts=30, max_steps=30, seed=42, verbose=True):
    """Method 3: GEM scalar gradient."""
    def step_fn(config, predictor, override_no_increase=False):
        cpi_grad, _ = predictor.compute_cpi_gradient(config)
        k = 0 if override_no_increase else 1
        new_config, changes = _scalar_gradient_step(config, cpi_grad, threshold,
                                                     top_k_increase=k)
        return new_config, changes, {}
    return _run_gradient_search(step_fn, gem_predictor, area_fn,
                                "gem_scalar_grad", n_starts, max_steps, seed, verbose)


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


def _attribution_step(config, gem_predictor, area_fn, reclaim_threshold=0.05,
                       verbose=False):
    """
    MSCS-Attributed Greedy Step:
    1. predict_mscs → find dominant bottleneck (avg across stages)
    2. compute Jacobian → per-param reduction of dominant type
    3. cost-effectiveness = reduction / Δarea → pick best
    4. reclaim: shrink params whose total effect ≈ 0
    """
    mscs = gem_predictor.predict_mscs(config)

    # Step 1: Find dominant bottleneck type (avg across 4 stages)
    fp_avg = np.mean([mscs[i] for i in FP_INDICES])
    bp_avg = np.mean([mscs[i] for i in BP_INDICES])
    sp_avg = np.mean([mscs[i] for i in SP_INDICES])
    pressures = {"FP": fp_avg, "BP": bp_avg, "SP": sp_avg}
    dominant = max(pressures, key=pressures.get)
    dom_indices = BOTTLENECK_INDICES[dominant]

    # Step 2: Compute Jacobian (backprop)
    jacobian = gem_predictor.compute_mscs_jacobian(config)

    # Step 3: For each param, compute reduction of dominant type + cost-effectiveness
    candidates = []
    all_reductions = {}
    for name, jac_row in jacobian.items():
        # Reduction of dominant pressure (negative gradient = reduces pressure)
        vals = get_param_values(name)
        param_range = max(vals) - min(vals)
        reduction = np.maximum(0, -jac_row[dom_indices]).sum() * param_range
        total_reduction = (np.maximum(0, -jac_row[FP_INDICES]).sum() +
                           np.maximum(0, -jac_row[BP_INDICES]).sum() +
                           np.maximum(0, -jac_row[SP_INDICES]).sum()) * param_range
        all_reductions[name] = {"reduction": reduction, "total": total_reduction}

        if reduction < 1e-8:
            continue  # no effect on dominant bottleneck

        # Area cost for one step increase
        new_val, area_cost = _compute_area_cost(config, name, area_fn)
        if new_val is None:
            continue

        efficiency = reduction / area_cost
        candidates.append({
            "name": name, "reduction": reduction, "area_cost": area_cost,
            "efficiency": efficiency, "new_val": new_val,
        })

    changes = {}
    new_config = config.copy()

    # Step 4: Pick most cost-effective parameter to increase
    if candidates:
        candidates.sort(key=lambda x: x["efficiency"], reverse=True)
        best = candidates[0]
        new_config[best["name"]] = best["new_val"]
        changes[best["name"]] = {
            "from": config[best["name"]], "to": best["new_val"],
            "action": "increase",
            "reason": f"top-1 {dominant} reducer, eff={best['efficiency']:.4f}",
            "reduction": best["reduction"],
            "area_cost": best["area_cost"],
        }

    # Step 5: Reclaim over-provisioned params (total effect ≈ 0)
    for name, red in all_reductions.items():
        if name in changes:
            continue
        if red["total"] < reclaim_threshold:
            # This param has negligible effect on any bottleneck → shrink it
            vals = get_param_values(name)
            current = new_config[name]
            lo = min(vals)
            if current <= lo:
                continue
            if isinstance(ALL_PARAMS.get(name), list):
                idx = vals.index(current) if current in vals else 0
                if idx > 0:
                    new_config[name] = vals[idx - 1]
                    changes[name] = {
                        "from": current, "to": vals[idx - 1],
                        "action": "reclaim",
                        "reason": f"total_effect={red['total']:.4f}<{reclaim_threshold}",
                    }
            else:
                new_config[name] = current - 1
                changes[name] = {
                    "from": current, "to": current - 1,
                    "action": "reclaim",
                    "reason": f"total_effect={red['total']:.4f}<{reclaim_threshold}",
                }

    new_config = _project_to_valid(new_config)

    info = {
        "pressures": pressures,
        "dominant": dominant,
        "mscs": mscs.tolist(),
        "n_candidates": len(candidates),
        "top_candidate": candidates[0]["name"] if candidates else None,
    }

    if verbose:
        print(f"      Pressure: FP={fp_avg:.3f} BP={bp_avg:.3f} SP={sp_avg:.3f} "
              f"→ dominant={dominant}")
        if candidates:
            b = candidates[0]
            print(f"      Best: {b['name']} (red={b['reduction']:.4f}, "
                  f"Δarea={b['area_cost']:.2f}, eff={b['efficiency']:.4f})")
        recs = [f"{k}({v['from']}→{v['to']})" for k, v in changes.items()
                if v['action'] == 'reclaim']
        if recs:
            print(f"      Reclaim: {', '.join(recs)}")

    return new_config, changes, info


def run_gem_attribution(gem_predictor, area_fn, reclaim_threshold=0.05,
                          n_starts=30, max_steps=30, seed=42, verbose=True):
    """
    Method 5: MSCS-Attributed Greedy Search.

    Each step:
      1. predict_mscs → dominant bottleneck type
      2. backprop Jacobian → per-param reduction of dominant type
      3. cost-effectiveness ranking → increase top-1
      4. reclaim weak-effect params → save area
    """
    rng = np.random.RandomState(seed)
    configs, cpis, areas = [], [], []
    trajectories = []
    t0 = time.time()

    if verbose:
        print(f"\n{'='*60}\ngem_attribution (starts={n_starts}, steps={max_steps})\n{'='*60}")

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
            new_config, changes, info = _attribution_step(
                config, gem_predictor, area_fn, reclaim_threshold,
                verbose=(verbose and step <= 2))

            if not changes:
                if verbose: print(f"    Step {step}: no changes, stopping")
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
            traj_entry.update(info)
            trajectory.append(traj_entry)

            imp = abs(new_cpi - prev_cpi) / (abs(prev_cpi) + 1e-8)
            if imp < 0.001:
                stagnant += 1
                if stagnant >= 3:
                    if verbose: print(f"    Step {step}: stagnant")
                    break
            else:
                stagnant = 0

            if verbose and step % 3 == 0:
                pen = info["pressures"]
                print(f"    Step {step}: CPI={new_cpi:.4f}, Area={new_area:.2f}, "
                      f"↑{n_inc} ↓{n_rec}, dom={info['dominant']} "
                      f"[{pen['FP']:.2f}/{pen['BP']:.2f}/{pen['SP']:.2f}] "
                      f"[{time.time()-t0:.0f}s]")

            config, prev_cpi = new_config, new_cpi
        trajectories.append(trajectory)

    elapsed = time.time() - t0
    if not cpis:
        return {"method": "gem_attribution", "configs": [], "cpis": [], "areas": [],
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
    return {"method": "gem_attribution", "configs": configs, "cpis": cpis, "areas": areas,
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