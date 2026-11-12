"""
Coordinated-swap step, generalised (one function for every arm and mode):
  inc_rule  "mscs" (dominant-type MSCS reducer, original) | "scalar" (largest Delta-CPI)
  protect   MSCS PROTECT on decreases (original) or not
  band      None -> original behaviour; (area_fn, a_ref, delta) -> iso-area: a step is accepted
            only if area(new) <= a_ref * (1 + delta) (area_fn = FastArea, a_ref = its value at the
            walk start, so model bias cancels in the ratio); increases are tried in the original
            order (top-3 CPI-improving) until one admits a band-feasible decrease
  strides   decrease strides tried per parameter ((1,) = original)
With band=None and strides=(1,), coord_step17 == coord_ablation/coord.coord_step (checked by
selfcheck17.py), which itself == dse_engine._coordinated_step.
"""
import numpy as np
import space17 as S
C = S.C
from dse_engine import (_increase_by_stride, _decrease_by_stride, FP_INDICES, BP_INDICES, SP_INDICES,
                        BOTTLENECK_INDICES, MSCS_RECLAIM_THRESHOLD, _mscs_total_effect, GRADIENT_PARAMS)
_project_to_valid = C._project_to_valid


def _dec_k(config, name, k):
    c, old = dict(config), config[name]
    for _ in range(k):
        v, _ = _decrease_by_stride(c, name)
        if v is None:
            break
        c[name] = v
    return (c[name], old) if c[name] != old else (None, None)


def coord_step17(config, P, inc_rule="mscs", protect=True, band=None, strides=(1,)):
    mscs = P.predict_mscs(config)
    cpi_base = P.predict_cpi(config)
    pressures = {"FP": np.mean([mscs[i] for i in FP_INDICES]),
                 "BP": np.mean([mscs[i] for i in BP_INDICES]),
                 "SP": np.mean([mscs[i] for i in SP_INDICES])}
    dominant = max(pressures, key=pressures.get)
    dom = BOTTLENECK_INDICES[dominant]
    inc_c, dec_c, cache = [], [], {}
    for name in GRADIENT_PARAMS:
        nv, ov = _increase_by_stride(config, name)
        if nv is not None:
            ci = config.copy(); ci[name] = nv; ci = _project_to_valid(ci)
            mi = P.predict_mscs(ci); cache[name] = ci
            inc_c.append({"name": name, "new_val": nv, "old_val": ov,
                          "dom_reduction": np.mean([mscs[i] - mi[i] for i in dom]),
                          "dcpi": cpi_base - P.predict_cpi(ci) if inc_rule == "scalar" else None})
        for k in strides:
            nv, ov = (_decrease_by_stride(config, name) if k == 1 else _dec_k(config, name, k))
            if nv is not None:
                cd = config.copy(); cd[name] = nv; cd = _project_to_valid(cd)
                dec_c.append({"name": name, "new_val": nv, "old_val": ov, "k": k,
                              "dcpi_cost": P.predict_cpi(cd) - cpi_base, "config_dec": cd})
    if not inc_c:
        return config.copy(), {}, {"mode": "no_candidates"}
    keyf = "dom_reduction" if inc_rule == "mscs" else "dcpi"
    inc_c.sort(key=lambda x: x[keyf], reverse=True)
    dec_sorted = sorted(dec_c, key=lambda x: x["dcpi_cost"])
    in_band = (lambda c: True) if band is None else (lambda c: band[0](c) <= band[1] * (1 + band[2]))

    n_protected, tried = 0, 0
    for best in inc_c[:3]:
        if P.predict_cpi(cache[best["name"]]) >= cpi_base:
            continue
        tried += 1
        new = config.copy(); new[best["name"]] = best["new_val"]; new = _project_to_valid(new)
        changes = {best["name"]: {"from": best["old_val"], "to": best["new_val"], "action": "increase"}}
        chosen = None
        for d in dec_sorted:
            if d["name"] == best["name"]:
                continue
            if d["dcpi_cost"] >= 0.05:
                break
            if protect:
                total = _mscs_total_effect(P.predict_mscs(d["config_dec"]) - mscs)
                if d["dcpi_cost"] >= 0 and total >= MSCS_RECLAIM_THRESHOLD:
                    n_protected += 1
                    continue
            t = new.copy(); t[d["name"]] = d["new_val"]; t = _project_to_valid(t)
            if P.predict_cpi(t) <= P.predict_cpi(new) * 1.005 and in_band(t):
                chosen = (d, t)
                break
        if chosen is not None:
            d, t = chosen
            changes[d["name"]] = {"from": d["old_val"], "to": d["new_val"], "action": "decrease", "k": d["k"]}
            new = t
        elif not in_band(new):
            continue                                  # iso-area: this increase cannot be paid for
        new = _project_to_valid(new)
        return new, changes, {"mode": "coordinated", "dominant": dominant, "inc_param": best["name"],
                              "n_protected": n_protected}
        # band=None: the first CPI-improving increase is always returned here (original behaviour)
    return config.copy(), {}, {"mode": "no_improvement" if tried == 0 else "no_band_feasible"}


def refine17(config, P, inc_rule, protect, n_steps=15, band_fn=None, delta=None, strides=(1,)):
    """Walk; returns (end, path, log). path = every accepted step's config (in order)."""
    band = None if band_fn is None else (band_fn, band_fn(config), delta)
    path, log = [], []
    for _ in range(n_steps):
        new, changes, info = coord_step17(config, P, inc_rule, protect, band, strides)
        if not changes:
            log.append({"stop": info.get("mode")})
            break
        log.append({"changes": {k: v["action"] for k, v in changes.items()},
                    "dominant": info.get("dominant"), "n_protected": info.get("n_protected")})
        config = new; path.append(new)
    return config, path, log
