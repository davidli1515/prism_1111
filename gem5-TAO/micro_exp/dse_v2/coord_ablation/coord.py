"""
2x2 ablation of the paper's BO + MSCS refine (dse_exp/run_gem_dse.py, method gem_bo_mscs =
dse_engine._mscs_refine -> _coordinated_step), under fair conditions.
Original code is only IMPORTED, never modified; nothing is written outside coord_ablation/.

Arms (identical swap structure: one +1 increase and at most one -1 decrease per step):
                      increase choice                         decrease guard
  mscs_protect        dominant-type MSCS reducer (ORIGINAL)   MSCS PROTECT (ORIGINAL)  <- calls _coordinated_step itself
  mscs_noprotect      dominant-type MSCS reducer              none
  scalar_protect      largest Delta-CPI                       MSCS PROTECT
  scalar_noprotect    largest Delta-CPI                       none
coord_step() mirrors _coordinated_step line by line; with (inc="mscs", protect=True) it must give
exactly the original result (checked by selfcheck() in run_coord.py --selfcheck).

Predictor: the ORIGINAL pipeline (exp1_v2 gem_full.pt, dse_exp normalizer cache, GEMPredictor slice
sampling, CPI = mean_slices 1/(Base_commit * W_min)). CachedPredictor evaluates CPI and MSCS in one
forward pass (same numbers as GEMPredictor.predict_cpi / predict_mscs) and counts every new
(config, workload) evaluation; 17 of them = 1 joint query.
"""
import os, sys, json, glob
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MICRO = os.path.dirname(os.path.dirname(HERE))
DSE_EXP = os.path.join(MICRO, "dse_exp")
EXP1 = os.path.join(MICRO, "exp1")
for p in (EXP1, DSE_EXP):
    if p not in sys.path:
        sys.path.insert(0, p)

import dse_engine as DE                                           # original, read-only
from dse_engine import (GEMPredictor, _increase_by_stride, _decrease_by_stride, _project_to_valid,
                        FP_INDICES, BP_INDICES, SP_INDICES, BOTTLENECK_INDICES,
                        MSCS_RECLAIM_THRESHOLD, _mscs_total_effect, GRADIENT_PARAMS)
from dse_infer import DseEngine, load_precomputed
from config_dse import compute_w_min, ALL_PARAMS

ARMS = {"mscs_protect": ("mscs", True), "mscs_noprotect": ("mscs", False),
        "scalar_protect": ("scalar", True), "scalar_noprotect": ("scalar", False)}


def ckey(c):
    return tuple(sorted((k, c[k]) for k in ALL_PARAMS))


# ---------------------------------------------------------------- model / data (original pipeline)
def load_original_model():
    import run_gem_dse as R                                       # constants + cached normalizer (read-only)
    m = R.GEMModel(input_dim=R.GEM_INPUT_DIM, hidden_dims=R.GEM_HIDDEN_DIMS, dropout=R.DROPOUT)
    m.load_state_dict(torch.load(R.GEM_WEIGHT_PATH, map_location="cpu")); m.eval()
    gem_norm, _ = R.get_normalizers()                             # cache exists -> loads only
    return m, gem_norm, R


def single_workload_engine(precompute_base, workload, max_slices=200):
    """DseEngine restricted to one workload: same file order / even-slice rule / cap as
    DseEngine.__init__ (dse_infer.py), so get_features() returns identical slices."""
    eng = DseEngine.__new__(DseEngine)
    eng.max_slices = max_slices
    eng.data, eng.workload_names = {}, set()
    for l1_dir in sorted(glob.glob(os.path.join(precompute_base, "l1_*"))):
        l1 = int(os.path.basename(l1_dir).replace("l1_", ""))
        eng.data[l1] = {}
        bdir = os.path.join(l1_dir, workload)
        if not os.path.isdir(bdir):
            continue
        slices = []
        for f in sorted(glob.glob(os.path.join(bdir, "slice_*", "dse_precomputed.npy"))):
            sid = int(os.path.basename(os.path.dirname(f)).replace("slice_", ""))
            if sid % 2 != 0:
                continue
            if len(slices) >= max_slices:
                break
            slices.append((sid, load_precomputed(f)))
        if slices:
            eng.data[l1][workload] = slices
            eng.workload_names.add(workload)
    return eng


class CachedPredictor:
    """Same interface as GEMPredictor (predict_cpi / predict_mscs), one forward for both, cached."""
    def __init__(self, gp):
        self.gp = gp
        self.cache = {}
        self.n_eval = 0

    @torch.no_grad()
    def _eval(self, config):
        k = ckey(config)
        if k not in self.cache:
            x = self.gp._get_features(config, self.gp.cpi_indices)
            m = self.gp.model(x)
            w = float(compute_w_min(config))
            cpi = (1.0 / (m[:, 12] * w + 1e-8)).mean().item()
            self.cache[k] = (cpi, m.mean(dim=0).cpu().numpy())
            self.n_eval += 1
        return self.cache[k]

    def predict_cpi(self, config):
        return self._eval(config)[0]

    def predict_mscs(self, config):
        return self._eval(config)[1]


# ---------------------------------------------------------------- the step (mirror of _coordinated_step)
def coord_step(config, P, inc_rule="mscs", protect=True):
    mscs = P.predict_mscs(config)
    cpi_base = P.predict_cpi(config)
    pressures = {"FP": np.mean([mscs[i] for i in FP_INDICES]),
                 "BP": np.mean([mscs[i] for i in BP_INDICES]),
                 "SP": np.mean([mscs[i] for i in SP_INDICES])}
    dominant = max(pressures, key=pressures.get)
    dom_indices = BOTTLENECK_INDICES[dominant]

    inc_c, dec_c, cache = [], [], {}
    for name in GRADIENT_PARAMS:
        nv, ov = _increase_by_stride(config, name)
        if nv is not None:
            ci = config.copy(); ci[name] = nv; ci = _project_to_valid(ci)
            mi = P.predict_mscs(ci); cache[name] = ci
            inc_c.append({"name": name, "new_val": nv, "old_val": ov,
                          "dom_reduction": np.mean([mscs[i] - mi[i] for i in dom_indices]),
                          "dcpi": cpi_base - P.predict_cpi(ci) if inc_rule == "scalar" else None})
        nv, ov = _decrease_by_stride(config, name)
        if nv is not None:
            cd = config.copy(); cd[name] = nv; cd = _project_to_valid(cd)
            dec_c.append({"name": name, "new_val": nv, "old_val": ov,
                          "dcpi_cost": P.predict_cpi(cd) - cpi_base, "config_dec": cd})
    if not inc_c:
        return config.copy(), {}, {"mode": "no_candidates"}

    key = "dom_reduction" if inc_rule == "mscs" else "dcpi"
    inc_c.sort(key=lambda x: x[key], reverse=True)
    best = None
    for c in inc_c[:3]:
        if P.predict_cpi(cache[c["name"]]) < cpi_base:
            best = c; break
    if best is None:
        return config.copy(), {}, {"mode": "no_improvement"}

    new = config.copy(); new[best["name"]] = best["new_val"]; new = _project_to_valid(new)
    changes = {best["name"]: {"from": best["old_val"], "to": best["new_val"], "action": "increase"}}
    n_protected = 0
    dec = sorted([d for d in dec_c if d["name"] != best["name"]], key=lambda x: x["dcpi_cost"])
    for d in dec:
        if d["dcpi_cost"] >= 0.05:
            break
        if protect:
            total = _mscs_total_effect(P.predict_mscs(d["config_dec"]) - mscs)
            if d["dcpi_cost"] >= 0 and total >= MSCS_RECLAIM_THRESHOLD:
                n_protected += 1
                continue
        t = new.copy(); t[d["name"]] = d["new_val"]; t = _project_to_valid(t)
        if P.predict_cpi(t) <= P.predict_cpi(new) * 1.005:
            new = t
            changes[d["name"]] = {"from": d["old_val"], "to": d["new_val"], "action": "decrease"}
            break
    new = _project_to_valid(new)
    return new, changes, {"mode": "coordinated", "dominant": dominant, "inc_param": best["name"],
                          "n_protected": n_protected}


def make_step(arm):
    if arm == "mscs_protect":                                    # the original function itself
        return lambda c, P: DE._coordinated_step(c, P, None)
    inc, prot = ARMS[arm]
    return lambda c, P: coord_step(c, P, inc, prot)


def refine(config, P, step, n_steps=15):
    """= dse_engine._mscs_refine, plus a log of the decisions."""
    log = []
    for _ in range(n_steps):
        new, changes, info = step(config, P)
        if not changes:
            break
        log.append({"changes": {k: v["action"] for k, v in changes.items()},
                    "dominant": info.get("dominant"), "n_protected": info.get("n_protected")})
        config = new
    return config, log
