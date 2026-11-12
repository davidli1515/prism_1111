"""
Refine v3 (finite differences only, no Jacobian) for the joint / averaged setting.

Fixes over v2, applied identically to the scalar and MSCS variants:
  1. all 21 parameters are searched (l2_size, l2_assoc, fetchBufferSize, wbWidth, bp_type too)
  2. multi-scale decreases (1 and 4 strides) so large over-provisioning is reclaimed quickly
  3. moves are scored for the CPI-area front: increase by ΔCPI_rel / Δarea_rel,
     reclaim by area saved; every exactly-evaluated neighbour enters the archive
MSCS-specific (the only difference between selectors):
  - per-workload MSCS (17 x 16) instead of the 17-workload average
  - PROTECT: a reclaim with small geomean ΔCPI is refused if ANY workload's stack moves
    (max_w ||ΔMSCS_w||_1 >= tau_m): a resource that is idle on average but carries a
    hidden contribution for some workloads
  - DECOUPLE (plateau, max ΔCPI_rel < eps): pick the increase with the largest
    per-workload cancellation  Σ_w (Σ_t|e_wt| - |Σ_t e_wt|)
Selectors: scalar | mscs | mscs_avg (ablation: same rules on the averaged MSCS)
"""
import time
import numpy as np

from common import Surrogate, ALL_PARAMS, get_param_values, config_key, GEM_INPUT_DIM, BASE_IDX
from dse_engine import _project_to_valid, PARAM_STRIDE
import torch

PRESSURE = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
P3 = dict(eps=0.003, tau=0.003, tau_m=0.02, dec_scales=(1, 4))


class SurrogateWL(Surrogate):
    """Surrogate that also keeps the per-workload MSCS (common.Surrogate is unchanged)."""
    @torch.no_grad()
    def _eval(self, c):
        from common import compute_w_min
        t0 = time.time()
        w = float(compute_w_min(c))
        cpis, mscs = [], []
        for wl in self.workloads:
            X = np.stack([f[:GEM_INPUT_DIM] for f in self.engine.get_features(c, workload=wl)])
            x = torch.from_numpy(self.norm.transform(X)).float().to(self.device)
            m = self.model(x)
            base = torch.clamp(m[:, BASE_IDX], min=0.01)
            cs = torch.clamp(1.0 / (base * w + 1e-8), max=100.0)
            per = cs.mean(1) if self.cpi_stage == "mean" else cs[:, 3]
            cpis.append(per.mean().item()); mscs.append(m.mean(0).cpu().numpy())
        self.n_queries += 1
        self.t_query += time.time() - t0
        M = np.stack(mscs)
        return {"cpi": float(np.exp(np.mean(np.log(cpis)))), "per_wl_cpi": cpis,
                "mscs": M.mean(0), "mscs_wl": M}

    def mscs_wl(self, c):
        return self.query(c)["mscs_wl"]


def _shift(c, name, k):
    """Move parameter `name` by k strides (k<0: decrease). None if out of range / no-op."""
    vals = get_param_values(name)
    cur = c[name]
    if isinstance(ALL_PARAMS[name], list):
        i = vals.index(cur) if cur in vals else 0
        j = i + k
        if not 0 <= j < len(vals):
            return None
        new = vals[j]
    else:
        s = PARAM_STRIDE.get(name) or 1
        new = min(max(cur + k * s, min(vals)), max(vals))
    if new == cur:
        return None
    cn = dict(c); cn[name] = new; cn = _project_to_valid(cn)
    return None if config_key(cn) == config_key(c) else cn


def _cancel(e):                       # e: (W,16) or (16,)
    e = np.atleast_2d(e)[:, PRESSURE]
    return float((np.abs(e).sum(1) - np.abs(e.sum(1))).sum())


def step_v3(c, S, archive, selector, area_fn, p=P3, prefetch=None):
    """One refine step from c. Returns (new_config or None, info)."""
    cpi0, a0 = S.cpi(c), area_fn(c)
    inc = [(n, cn) for n in ALL_PARAMS for cn in [_shift(c, n, +1)] if cn is not None]
    if prefetch: prefetch([cn for _, cn in inc])
    cand = []
    for n, cn in inc:
        archive.add(cn, f"v3_{selector}")
        a = area_fn(cn)
        if a is None:
            continue
        g = (cpi0 - S.cpi(cn)) / cpi0
        if g > 0:
            cand.append((n, cn, g, max(a - a0, 1e-6) / a0))
    if not cand:
        return None, {"mode": "none"}
    best_g = max(t[2] for t in cand)
    mode = "normal"
    if selector != "scalar" and best_g < p["eps"]:
        mode = "decouple"
        m0 = S.mscs_wl(c) if selector == "mscs" else S.mscs(c)
        get = (lambda x: S.mscs_wl(x)) if selector == "mscs" else (lambda x: S.mscs(x))
        name, cnew, g, _ = max(cand, key=lambda t: _cancel(m0 - get(t[1])))
    else:
        name, cnew, g, _ = max(cand, key=lambda t: t[2] / t[3])        # ΔCPI per area

    # ---- reclaim from cnew: one best move (largest area saving among acceptable) ----
    cpi1, a1 = S.cpi(cnew), area_fn(cnew)
    dec = [(n, cd) for n in ALL_PARAMS if n != name for k in p["dec_scales"]
           for cd in [_shift(cnew, n, -k)] if cd is not None]
    if prefetch: prefetch([cd for _, cd in dec])
    ok, protected = [], []
    for n, cd in dec:
        archive.add(cd, f"v3_{selector}")
        a = area_fn(cd)
        if a is None or a >= a1:
            continue
        cost = (S.cpi(cd) - cpi1) / cpi1
        if cost >= p["tau"]:
            continue
        if selector != "scalar" and cost >= 0:
            d = (S.mscs_wl(cd) - S.mscs_wl(cnew)) if selector == "mscs" else (S.mscs(cd) - S.mscs(cnew))
            if np.abs(np.atleast_2d(d)).sum(1).max() >= p["tau_m"]:
                protected.append(n); continue
        ok.append((n, cd, a1 - a))
    out = cnew
    rec = None
    if ok:
        rec, out, _ = max(ok, key=lambda t: t[2])
    return out, {"mode": mode, "inc": name, "gain": g, "reclaim": rec, "protected": protected}


def refine_front_v3(S, archive, selector, area_fn, total_queries, steps=15, max_rounds=10,
                    p=P3, prefetch=None, log=print):
    """Walk every (new) Pareto point of the archive with step_v3 until the budget is used."""
    from common import pareto_mask
    done = set()
    for rnd in range(max_rounds):
        if S.n_queries >= total_queries:
            break
        P = np.c_[archive.cpis, archive.areas]
        todo = sorted((i for i in np.where(pareto_mask(P))[0] if config_key(archive.configs[i]) not in done),
                      key=lambda i: archive.cpis[i])
        if not todo:
            break
        for i in todo:
            if S.n_queries >= total_queries:
                break
            c = archive.configs[i]; done.add(config_key(c))
            prev = S.cpi(c); stag = 0
            for _ in range(steps):
                if S.n_queries >= total_queries:
                    break
                cn, info = step_v3(c, S, archive, selector, area_fn, p, prefetch)
                if cn is None:
                    break
                done.add(config_key(cn))
                cpi = S.cpi(cn)
                stag = stag + 1 if abs(cpi - prev) / prev < 0.001 and info["mode"] != "decouple" else 0
                if stag >= 3:
                    break
                c, prev = cn, cpi
        log(f"    v3 refine round {rnd}: q={S.n_queries} pts={len(archive.cpis)}")
