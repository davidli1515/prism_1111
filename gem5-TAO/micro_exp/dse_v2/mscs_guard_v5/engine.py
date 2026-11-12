"""Frozen-checkpoint DSE: conditional checks choose actual next configurations.

Only scalar_guard versus mscs_guard is a controlled change of query ordering.
The scalar arm is an immediate-reclaim baseline, with the same accounting and
walk scheduler. Every observed configuration can contribute to the archive.
"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mscs_response_1008"))
from probe import SliceOracle, _shift, ALL_PARAMS, config_key, compute_w_min
from response import cpi_samples, config_additive
from common import pareto_mask, hypervolume, sample_random_config

METHODS = ("scalar", "scalar_guard", "mscs_guard")
PARAMS = tuple(ALL_PARAMS)
TAU = .003


class BudgetExhausted(Exception):
    pass


class BudgetOracle:
    """A method's own logical ledger, independent of physical inference reuse."""
    def __init__(self, physical, budget):
        self.physical, self.budget = physical, budget
        self.seen = {}

    @property
    def n_queries(self):
        return len(self.seen)

    def query(self, c):
        key = config_key(c)
        if key not in self.seen:
            if self.n_queries >= self.budget:
                raise BudgetExhausted
            result = self.physical.query(c)
            self.seen[key] = dict(c)
            return result
        return self.physical.query(c)

    def cpi(self, c):
        return self.query(c)[1]


class AtomicArchive:
    """Commit aligned arrays only after both CPI and area have succeeded."""
    def __init__(self, surrogate, area):
        self.S, self.A = surrogate, area
        self.configs, self.cpis, self.areas, self.queries, self.tags = [], [], [], [], []
        self.indices, self.failed_area = {}, []

    def add(self, c, tag=""):
        key = config_key(c)
        if key in self.indices:
            return self.indices[key]
        v = self.S.cpi(c)  # Hard cap is checked before ANY archive mutation.
        a = self.A(c)
        if a is None:
            self.failed_area.append({"config": dict(c), "query": self.S.n_queries, "tag": tag})
            return None
        i = len(self.cpis)
        self.indices[key] = i
        self.configs.append(dict(c)); self.cpis.append(float(v)); self.areas.append(float(a))
        self.queries.append(self.S.n_queries); self.tags.append(tag)
        return i

    def row(self, c, tag=""):
        i = self.add(c, tag)
        return None if i is None else {"config": self.configs[i], "cpi": self.cpis[i],
                                       "area": self.areas[i], "index": i}

    def clone(self, surrogate):
        other = AtomicArchive(surrogate, self.A)
        for name in ("configs", "cpis", "areas", "queries", "tags", "failed_area"):
            setattr(other, name, list(getattr(self, name)))
        other.indices = dict(self.indices)
        surrogate.seen = dict(self.S.seen)
        return other

    def points(self):
        return np.c_[self.cpis, self.areas]

    def result(self):
        return {name: getattr(self, name) for name in
                ("configs", "cpis", "areas", "queries", "tags", "failed_area")}


def pressure_composition(m):
    p = np.asarray(m, dtype=float).reshape(-1, 4, 4)[..., 1:]
    return p / p.sum(axis=-1, keepdims=True)


def risk_score(S, base, dec, method):
    m0, _, ids0 = S.query(base)
    mj, _, idsj = S.query(dec)
    if ids0 != idsj:
        raise ValueError("Per-slice scores require matching slice IDs")
    if method == "scalar_guard":
        v0 = cpi_samples(m0, compute_w_min(base))
        vj = cpi_samples(mj, compute_w_min(dec))
        return float(np.quantile(vj / v0 - 1, .9))
    if method == "mscs_guard":
        return float(np.abs(pressure_composition(mj)-pressure_composition(m0))
                     .sum(axis=(1, 2)).mean())
    raise ValueError(method)


def move_score(base, candidate):
    gain = (base["cpi"]-candidate["cpi"])/base["cpi"]
    area_cost = max(candidate["area"]-base["area"], 1e-6)/base["area"]
    return gain / area_cost


def admissible_joint(base, dec, future):
    """Exclude validity projection artifacts, not just matching parameter names."""
    if dec["param"] == future["param"]:
        return None
    cj, ck = dec["config"], future["config"]
    cjk = _shift(cj, future["param"], future["stride"])
    ckj = _shift(ck, dec["param"], -dec["stride"])
    if (cjk is None or ckj is None or config_key(cjk) != config_key(ckj)
            or not config_additive(base, cj, ck, cjk)):
        return None
    return cjk


def select_verified_options(futures, checks, tau=TAU):
    """A bad check forbids only THAT joint action, never a whole resource."""
    options = [dict(f, choice="future", reclaim=None) for f in futures]
    for check in checks:
        if check["future_cost"] <= tau and check["area"] < check["reference_area"]:
            options.append(dict(check, choice="joint", reclaim=check["j"]))
    return options


def observe_moves(c, specs, arch, tag, prefetch):
    seen, moves = set(), []
    for param, stride in specs:
        cn = _shift(c, param, stride)
        if cn is None or config_key(cn) in seen:
            continue
        seen.add(config_key(cn))
        moves.append((param, stride, cn))
    # Never compute area for unseen configurations beyond the remaining CPI budget.
    remaining = arch.S.budget-arch.S.n_queries
    allowed = []
    for p, k, cn in moves:
        if config_key(cn) not in arch.S.seen:
            if remaining == 0:
                break
            remaining -= 1
        allowed.append((p, k, cn))
    prefetch([cn for _, _, cn in allowed])
    rows = []
    for param, stride, cn in allowed:
        row = arch.row(cn, tag)
        if row is not None:
            rows.append(dict(row, param=param, stride=abs(stride)))
    if len(allowed) != len(moves):
        raise BudgetExhausted
    return rows


def step(c, arch, method, visited, prefetch):
    S = arch.S
    q0 = S.n_queries
    base = arch.row(c)
    inc = observe_moves(c, ((p, 1) for p in PARAMS), arch, "increase", prefetch)
    inc = [r for r in inc if r["cpi"] < base["cpi"] and config_key(r["config"]) not in visited]
    if not inc:
        return None, {"q": S.n_queries, "reason": "no_improvement"}
    y = max(inc, key=lambda r: move_score(base, r))
    cy = y["config"]
    down = observe_moves(cy, ((p, -k) for p in PARAMS if p != y["param"]
                              for k in (1, 4)), arch, "reclaim", prefetch)
    cheap = []
    for d in down:
        cost = d["cpi"]/y["cpi"]-1
        if cost <= TAU and d["area"] < y["area"] and config_key(d["config"]) not in visited:
            cheap.append(dict(d, current_cost=cost))
    immediate = min(cheap, key=lambda d: d["area"]) if cheap else y
    info = {"start": base["index"], "increase": y["param"], "intermediate": y["index"],
            "cheap_reclaims": len(cheap), "q_start": q0, "checks": [], "selected_reclaims": [],
            "future_queries": 0, "joint_queries": 0, "choice": "immediate",
            "reclaim": immediate["param"] if cheap else None}
    out = immediate
    # L1 trace changes remain in scalar search; compositional guards use fixed traces.
    eligible = [d for d in cheap if d["current_cost"] >= 0 and d["param"] != "l1_size"]
    info["eligible_guard_actions"] = len(eligible)
    info["eligible_guard_resources"] = len({d["param"] for d in eligible})
    if method != "scalar" and eligible:
        for d in eligible:
            d["risk_score"] = risk_score(S, cy, d["config"], method)
        selected, used = [], set()
        for d in sorted(eligible, key=lambda r: (-r["risk_score"], r["param"], r["stride"])):
            if d["param"] not in used:
                selected.append(d); used.add(d["param"])
            if len(selected) == 3:
                break
        info["selected_reclaims"] = [{k: d[k] for k in
                                      ("param", "stride", "risk_score", "current_cost", "index")}
                                     for d in selected]
        before = S.n_queries
        up = observe_moves(cy, ((p, k) for p in PARAMS if p != "l1_size"
                                for k in (1, 4, 8)), arch, "future", prefetch)
        info["future_queries"] = S.n_queries-before
        futures, used = [], set()
        # Same observed scalar-gain order in both guarded methods; no joint labels.
        for u in sorted(up, key=lambda r: (r["cpi"], r["param"], r["stride"])):
            if u["cpi"] >= y["cpi"] or u["param"] in used:
                continue
            futures.append(u); used.add(u["param"])
            if len(futures) == 3:
                break
        info["futures"] = [{k: u[k] for k in ("param", "stride", "index")} for u in futures]
        pairs = []
        for d in selected:
            for u in futures:
                joint = admissible_joint(cy, d, u)
                if joint is not None:
                    pairs.append((d, u, joint))
        before = S.n_queries
        # Query order is observable at a hard budget boundary and is intentional.
        for d, u, joint in pairs:
            r = arch.row(joint, "joint")
            if r is None:
                continue
            cost = r["cpi"]/u["cpi"]-1
            info["checks"].append(dict(r, j=d["param"], j_stride=d["stride"],
                                       k=u["param"], k_stride=u["stride"],
                                       reference_index=u["index"], reference_area=u["area"],
                                       current_cost=d["current_cost"], future_cost=cost,
                                       blocked=bool(cost > TAU)))
        info["joint_queries"] = S.n_queries-before
        assert info["joint_queries"] <= 9
        options = select_verified_options(futures, info["checks"])
        # A directly improving reclaim remains usable without speculative protection.
        options += [dict(d, choice="direct_reclaim", reclaim=d["param"])
                    for d in cheap if d["current_cost"] < 0]
        options = [r for r in options if r["cpi"] < y["cpi"]
                   and config_key(r["config"]) not in visited]
        if options:
            out = max(options, key=lambda r: move_score(y, r))
            info["choice"], info["reclaim"] = out["choice"], out["reclaim"]
            info["chosen_future"] = out.get("k", out.get("param")) if out["choice"] != "direct_reclaim" else None
    info.update(end=out["index"], q=S.n_queries, cpi=out["cpi"], area=out["area"])
    return out["config"], info


def refine(arch, method, seed, prefetch, log=print, max_steps=15):
    """No repeated walk states. Exhausted frontiers receive shared random restarts."""
    rng = np.random.RandomState(seed + 100000)
    expanded, walks = set(), []
    last_log, restarts = arch.S.n_queries, 0
    try:
        while arch.S.n_queries < arch.S.budget:
            P = arch.points()
            todo = [i for i in np.flatnonzero(pareto_mask(P))
                    if config_key(arch.configs[i]) not in expanded]
            if not todo:
                # Fresh exploration is charged, with identical rule and seed in all arms.
                before = arch.S.n_queries
                for _ in range(30):
                    arch.add(sample_random_config(rng), "restart")
                    if arch.S.n_queries >= arch.S.budget:
                        break
                restarts += 1
                P = arch.points()
                todo = [i for i in np.flatnonzero(pareto_mask(P))
                        if config_key(arch.configs[i]) not in expanded]
                if not todo and arch.S.n_queries > before:
                    continue
                if not todo:
                    raise RuntimeError("No new configurations after a restart")
            todo.sort(key=lambda i: (arch.cpis[i], arch.areas[i], i))
            for i in todo:
                c, trajectory = arch.configs[i], []
                visited = set(expanded)
                for _ in range(max_steps):
                    key = config_key(c)
                    if key in visited:
                        break
                    visited.add(key); expanded.add(key)
                    cn, info = step(c, arch, method, visited, prefetch)
                    trajectory.append(info)
                    if arch.S.n_queries-last_log >= 250:
                        last_log = arch.S.n_queries
                        log(f"{method} q={last_log} points={len(arch.cpis)} walks={len(walks)}")
                    if cn is None:
                        break
                    c = cn
                walks.append({"start_idx": int(i), "steps": trajectory})
    except BudgetExhausted:
        # Every query made before the boundary is already archived. A partial step
        # cannot execute an unverified move and does not enter completed-step stats.
        if "trajectory" in locals():
            walks.append({"start_idx": int(i), "steps": trajectory, "budget_boundary": True})
    return {"walks": walks, "restarts": restarts, "expanded_states": len(expanded)}
