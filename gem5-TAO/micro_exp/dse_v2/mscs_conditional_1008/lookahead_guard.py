"""MSCS chooses which reclaims to inspect; measured CPI decides protection.

Pure search adapter. A high score NEVER protects a resource by itself.
The caller supplies feasible commuting joint configurations and already observed
future CPI references (as in collect.py). Evaluation is the frozen surrogate or
gem5 oracle chosen by the caller. Results distinguish unverified from safe.
This is not yet wired into the historical DSE drivers, which remain unchanged.
"""
import math


def inspect_reclaims(actions, evaluate_cpi, budget, tau=.003, known_cpi=None):
    """Spend at most budget distinct new joint queries, highest risk score first.

    actions: [{id, score, checks: [{key, config, reference_cpi, partner}]}].
    key must identify the actual joint configuration, not its parameter label.
    known_cpi is copied; all new values are returned for the caller's cache/archive.
    An action is verified_safe only after EVERY provided future check passes.
    Safety refers only to these supplied future directions and this CPI tolerance.
    """
    if budget < 0 or tau < 0:
        raise ValueError("budget and tau must be nonnegative")
    cache = dict(known_cpi or {})
    spent, decisions = 0, []
    for action in sorted(actions, key=lambda a: -a["score"]):
        decision = {"id": action["id"], "status": "unverified", "evidence": []}
        for check in action["checks"]:
            reference = check["reference_cpi"]
            if not math.isfinite(reference) or reference <= 0:
                raise ValueError("reference CPI must be positive and finite")
            key = check["key"]
            if key not in cache:
                if spent >= budget:
                    break
                value = float(evaluate_cpi(check["config"]))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("oracle CPI must be positive and finite")
                cache[key] = value
                spent += 1
            cost = (cache[key]-reference)/reference
            decision["evidence"].append({"partner": check["partner"], "cost": cost,
                                         "cpi": cache[key], "reference_cpi": reference})
            if cost > tau:
                decision["status"] = "protect"
                break
        else:
            if action["checks"]:
                decision["status"] = "verified_safe"
        decisions.append(decision)
    return {"queries": spent, "decisions": decisions, "cpi_cache": cache}
