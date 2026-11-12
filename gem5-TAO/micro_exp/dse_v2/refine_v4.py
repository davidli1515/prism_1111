"""
Refine v4 = v3 (finite differences, 21 params, multi-scale reclaim) with the decouple
and protect rules made safe and tunable, plus per-step logging.

Changes vs refine_v3.step_v3 (everything else identical, incl. the scalar path):
  DECOUPLE trigger   v3: best_gain < eps (absolute 0.3%)  -> on gcc (CPI~5, per-step gains
                     0.1-0.5%) this fired on ~50% of steps and picked moves with ~0.5x gain.
                     v4: plateau iff best_gain < max(eps_floor, rel_eps * median(previous
                     best_gains of this walk)); first step uses eps_floor only.
  DECOUPLE pick      v4: argmax cancellation score ONLY among candidates whose gain is
                     >= gamma * best_gain (gamma=0.5); the best-gain move always qualifies,
                     so decouple can never cost more than (1-gamma) of the best gain.
  PROTECT            v4: refuse a reclaim if >= k_wl workloads have ||dMSCS_w||_1 >= tau_m
                     (v3 = k_wl 1, tau_m 0.02). Both are parameters.
  LOGGING            every step returns mode, best_gain, chosen gain, ratio, reclaimed
                     param + area saved, protected params; walks are stored in the result.
Selectors: scalar | mscs (per-workload) | mscs_avg
"""
import numpy as np
from common import ALL_PARAMS, config_key, pareto_mask
from refine_v3 import SurrogateWL, _shift, _cancel, PRESSURE   # reused, not modified

P4 = dict(eps_floor=0.0005,   # absolute plateau floor (0.05%)
          rel_eps=0.2,        # plateau if best_gain < 20% of this walk's median best_gain
          gamma=0.5,          # decouple pick must keep >= 50% of the best gain
          tau=0.003,          # reclaim accepted if CPI rises < 0.3%
          tau_m=0.02, k_wl=1, # protect if >= k_wl workloads move by >= tau_m (L1 of 16-dim)
          dec_scales=(1, 4))


def _protect(S, selector, cur, cd, p):
    if selector == "scalar":
        return False, 0
    if selector == "mscs":
        d = S.mscs_wl(cd) - S.mscs_wl(cur)          # (W,16)
    else:
        d = (S.mscs(cd) - S.mscs(cur))[None, :]
    n_over = int((np.abs(d).sum(1) >= p["tau_m"]).sum())
    return n_over >= p["k_wl"], n_over


def step_v4(c, S, archive, selector, area_fn, hist, p=P4, prefetch=None):
    """One refine step. hist: list of previous best_gains in this walk (mutated)."""
    cpi0, a0 = S.cpi(c), area_fn(c)
    inc = [(n, cn) for n in ALL_PARAMS for cn in [_shift(c, n, +1)] if cn is not None]
    if prefetch: prefetch([cn for _, cn in inc])
    cand = []
    for n, cn in inc:
        archive.add(cn, f"v4_{selector}")
        a = area_fn(cn)
        if a is None:
            continue
        g = (cpi0 - S.cpi(cn)) / cpi0
        if g > 0:
            cand.append((n, cn, g, max(a - a0, 1e-6) / a0))
    if not cand:
        return None, {"mode": "none"}
    best_g = max(t[2] for t in cand)
    thr = max(p["eps_floor"], p["rel_eps"] * float(np.median(hist))) if hist else p["eps_floor"]
    hist.append(best_g)
    mode, cscore = "normal", None
    if selector != "scalar" and best_g < thr:
        mode = "decouple"
        get = S.mscs_wl if selector == "mscs" else S.mscs
        m0 = get(c)
        pool = [t for t in cand if t[2] >= p["gamma"] * best_g]
        scored = [(_cancel(m0 - get(t[1])), t) for t in pool]
        cscore, (name, cnew, g, _) = max(scored, key=lambda x: x[0])
    else:
        name, cnew, g, _ = max(cand, key=lambda t: t[2] / t[3])

    cpi1, a1 = S.cpi(cnew), area_fn(cnew)
    dec = [(n, cd) for n in ALL_PARAMS if n != name for k in p["dec_scales"]
           for cd in [_shift(cnew, n, -k)] if cd is not None]
    if prefetch: prefetch([cd for _, cd in dec])
    ok, protected = [], []
    for n, cd in dec:
        archive.add(cd, f"v4_{selector}")
        a = area_fn(cd)
        if a is None or a >= a1:
            continue
        cost = (S.cpi(cd) - cpi1) / cpi1
        if cost >= p["tau"]:
            continue
        if cost >= 0:
            blocked, n_over = _protect(S, selector, cnew, cd, p)
            if blocked:
                protected.append((n, n_over, a1 - a)); continue
        ok.append((n, cd, a1 - a))
    out, rec, saved = cnew, None, 0.0
    if ok:
        rec, out, saved = max(ok, key=lambda t: t[2])
    return out, {"mode": mode, "threshold": thr, "best_gain": best_g, "inc": name, "gain": g,
                 "gain_ratio": g / best_g, "cancel": cscore, "reclaim": rec, "area_saved": saved,
                 "protected": protected, "cpi": S.cpi(out), "area": area_fn(out), "q": S.n_queries}


def refine_front_v4(S, archive, selector, area_fn, total_queries, steps=15, max_rounds=10,
                    p=P4, prefetch=None, log=print):
    done, walks = set(), []
    for rnd in range(max_rounds):
        if S.n_queries >= total_queries:
            break
        P = np.c_[archive.cpis, archive.areas]
        todo = sorted((i for i in np.where(pareto_mask(P))[0]
                       if config_key(archive.configs[i]) not in done), key=lambda i: archive.cpis[i])
        if not todo:
            break
        for i in todo:
            if S.n_queries >= total_queries:
                break
            c = archive.configs[i]; done.add(config_key(c))
            prev, stag, hist, traj = S.cpi(c), 0, [], []
            for _ in range(steps):
                if S.n_queries >= total_queries:
                    break
                cn, info = step_v4(c, S, archive, selector, area_fn, hist, p, prefetch)
                if cn is None:
                    break
                traj.append(info); done.add(config_key(cn))
                cpi = S.cpi(cn)
                stag = stag + 1 if abs(cpi - prev) / prev < 0.001 and info["mode"] != "decouple" else 0
                if stag >= 3:
                    break
                c, prev = cn, cpi
            walks.append({"round": rnd, "start_idx": int(i), "steps": traj})
        log(f"    v4 refine round {rnd}: q={S.n_queries} pts={len(archive.cpis)}")
    return walks


def walk_stats(walks):
    st = [s for w in walks for s in w["steps"]]
    dec = [s for s in st if s["mode"] == "decouple"]
    return {"steps": len(st), "decouple_steps": len(dec),
            "decouple_gain_ratio_median": float(np.median([s["gain_ratio"] for s in dec])) if dec else None,
            "reclaims": sum(1 for s in st if s["reclaim"]), "area_saved": float(sum(s["area_saved"] for s in st)),
            "protected_events": sum(len(s["protected"]) for s in st)}
