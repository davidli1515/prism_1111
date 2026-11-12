"""
Unified greedy search: every method shares starts, strides, reclaim rule,
safety valve and stopping rule. Only the *decision* differs:

  selector       increase (when max ΔCPI_rel < eps)        reclaim (0 <= cost_rel < tau)
  ------------   ---------------------------------------   ------------------------------
  scalar         argmax ΔCPI (always)                      reclaim
  mscs           argmax S_cancel  (decouple, Alg. 2)        protect if ||ΔMSCS||_1 >= tau_m
  mscs_decouple  argmax S_cancel                           reclaim            (ablation)
  mscs_protect   argmax ΔCPI                               protect if ...     (ablation)

When max ΔCPI_rel >= eps, every selector takes argmax ΔCPI (Alg. 2 line 4-5).
Thresholds are *relative* to the current CPI so they mean the same thing for
every benchmark / the joint geomean.
"""
import numpy as np

from common import (GRADIENT_PARAMS, ALL_PARAMS, get_param_values, sample_random_config,
                    config_key, arch_delta)
from dse_engine import (_project_to_valid, _increase_by_stride, _decrease_by_stride)  # dse_exp

FP = [1, 5, 9, 13]; BP = [2, 6, 10, 14]; SP = [3, 7, 11, 15]
PRESSURE = FP + BP + SP
SELECTORS = ["scalar", "mscs", "mscs_decouple", "mscs_protect", "mscs_jac", "scalar_jac"]
# *_jac: same decisions as mscs / scalar, but candidates are first ranked with the
# surrogate Jacobian (1 query) and only the top ones are verified with exact queries.
JAC_TOPK_INC = 3      # verified increase candidates per criterion (ΔCPI, S_cancel)
JAC_TOPK_DEC = 4      # verified reclaim candidates (lowest linearized cost)

DEFAULTS = dict(eps=0.003,      # 0.3% CPI: below this the scalar signal is "flat"
                tau=0.003,      # reclaim if removing costs < 0.3% CPI
                tau_m=0.02,     # ||ΔMSCS||_1 threshold for PROTECT
                valve=0.005,    # total reclaim may not worsen CPI by > 0.5%
                max_decouple=5, stagnant=3, stag_eps=0.001)


def cancel_score(e, mode="paper"):
    """e: 16-dim MSCS(θ) - MSCS(θ') (positive = pressure reduced).
    paper : Σ_t |e_t| - |Σ_t e_t| over the 12 pressure components (Lemma 1)
    legacy: dse_exp._cancel_score (per type: largest-|.| stage, then 3 terms)"""
    if mode == "paper":
        v = e[PRESSURE]
        return float(np.abs(v).sum() - abs(v.sum()))
    eff = [e[I][np.argmax(np.abs(e[I]))] for I in (FP, BP, SP)]
    return float(sum(abs(x) for x in eff) - abs(sum(eff)))


def _moves(c, fn):
    out = []
    for n in GRADIENT_PARAMS:
        v, _ = fn(c, n)
        if v is None:
            continue
        cn = dict(c); cn[n] = v; cn = _project_to_valid(cn)
        if config_key(cn) != config_key(c):
            out.append((n, cn))
    return out


def greedy_step(c, S, selector, p=DEFAULTS, cancel="paper"):
    """One step. Returns (new_config, info) or (None, info) if no improving move."""
    use_jac = selector.endswith("_jac")
    base_sel = selector[:-4] if use_jac else selector
    cpi0 = S.cpi(c)
    inc = _moves(c, _increase_by_stride)
    if use_jac:                                   # screen with the linearization
        Jc, Jm = S.jac(c)
        lin = [(n, cn, -(Jc @ arch_delta(c, cn)) / cpi0, -(Jm @ arch_delta(c, cn))) for n, cn in inc]
        keep = {t[0] for t in sorted(lin, key=lambda t: -t[2])[:JAC_TOPK_INC]}
        if base_sel != "scalar":
            keep |= {t[0] for t in sorted(lin, key=lambda t: -cancel_score(t[3], cancel))[:JAC_TOPK_INC]}
        inc = [(n, cn) for n, cn in inc if n in keep]
    cands = []
    for n, cn in inc:
        gain = (cpi0 - S.cpi(cn)) / cpi0
        if gain > 0:
            cands.append((n, cn, gain))
    if not cands:
        return None, {"mode": "none"}

    best_gain = max(g for _, _, g in cands)
    if best_gain >= p["eps"] or base_sel in ("scalar", "mscs_protect"):
        mode = "normal"
        name, cnew, gain = max(cands, key=lambda t: t[2])
    else:
        mode = "decouple"
        m0 = S.mscs(c)
        name, cnew, gain = max(cands, key=lambda t: cancel_score(m0 - S.mscs(t[1]), cancel))

    # ---- reclaim (one parameter at a time, same order for all selectors) ----
    protect = base_sel in ("mscs", "mscs_protect")
    cur, cpi_inc = cnew, S.cpi(cnew)
    reclaimed, protected = [], []
    order = [n for n in GRADIENT_PARAMS if n != name]
    if use_jac:                                   # only verify the cheapest-looking reclaims
        Jc, _ = S.jac(cnew)
        dec = dict(_moves(cnew, _decrease_by_stride))
        lin = sorted(((Jc @ arch_delta(cnew, cd)) / cpi_inc, n) for n, cd in dec.items() if n != name)
        order = [n for cst, n in lin if cst < 2 * p["tau"]][:JAC_TOPK_DEC]
    for n in order:
        dv, _ = _decrease_by_stride(cur, n)
        if dv is None:
            continue
        cd = dict(cur); cd[n] = dv; cd = _project_to_valid(cd)
        cpi_cur = S.cpi(cur)
        cost = (S.cpi(cd) - cpi_cur) / cpi_cur
        if cost < 0:
            cur = cd; reclaimed.append(n); continue
        if cost >= p["tau"]:
            continue
        if protect and np.abs(S.mscs(cd) - S.mscs(cur)).sum() >= p["tau_m"]:
            protected.append(n); continue
        cur = cd; reclaimed.append(n)
    if S.cpi(cur) > cpi_inc * (1 + p["valve"]):                 # global safety valve
        cur, reclaimed = cnew, []
    return cur, {"mode": mode, "inc": name, "gain": gain, "n_inc_verified": len(inc),
                 "reclaimed": reclaimed, "protected": protected}


def make_starts(n, rng):
    """Structured starts (1/3 small, 1/3 random, 1/3 large), identical across selectors."""
    starts = []
    for i in range(n):
        c = sample_random_config(rng)
        bias = "small" if i < n // 3 else ("rand" if i < 2 * n // 3 else "large")
        if bias != "rand":
            for k in c:
                vals = get_param_values(k); lo, hi = min(vals), max(vals); mid = (lo + hi) / 2
                if bias == "small" and c[k] > mid:
                    sub = [v for v in vals if v <= mid]
                elif bias == "large" and c[k] < mid:
                    sub = [v for v in vals if v >= mid]
                else:
                    continue
                if sub:
                    c[k] = sub[rng.randint(len(sub))]
            c = _project_to_valid(c)
        starts.append(c)
    return starts


def walk(c, S, archive, selector, max_steps, p=DEFAULTS, cancel="paper",
         query_budget=None, tag=""):
    """Greedy walk from c; every accepted config goes to the archive. Returns trajectory."""
    traj = [{"step": 0, "cpi": S.cpi(c)}]
    archive.add(c, tag)
    prev, stag, n_dec = S.cpi(c), 0, 0
    for step in range(1, max_steps + 1):
        if query_budget is not None and S.n_queries >= query_budget:
            break
        cn, info = greedy_step(c, S, selector, p, cancel)
        if cn is None:
            break
        archive.add(cn, tag)
        cpi = S.cpi(cn)
        traj.append({"step": step, "cpi": cpi, "queries": S.n_queries, **info})
        if info["mode"] == "decouple":
            n_dec += 1
            if n_dec >= p["max_decouple"]:
                break
        else:
            n_dec = 0
            if abs(cpi - prev) / prev < p["stag_eps"]:
                stag += 1
                if stag >= p["stagnant"]:
                    break
            else:
                stag = 0
        c, prev = cn, cpi
    return traj


def run_greedy(S, archive, selector, seed, n_starts=30, max_steps=30, p=DEFAULTS,
               cancel="paper", query_budget=None):
    rng = np.random.RandomState(seed)
    trajs = []
    for c in make_starts(n_starts, rng):
        if query_budget is not None and S.n_queries >= query_budget:
            break
        trajs.append(walk(c, S, archive, selector, max_steps, p, cancel, query_budget,
                          tag=f"greedy_{selector}"))
    return trajs
