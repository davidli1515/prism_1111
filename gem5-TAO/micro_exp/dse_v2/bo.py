"""
BO (same acquisition as dse_exp.run_bo_gem: GP(Matern 2.5) on CPI and area,
optimistic HV-improvement + exploration bonus over 500 random candidates),
driven by a *surrogate-query* budget so it can be compared with greedy /
refine methods at equal cost.

  run_bo(..., query_budget=Q)         pure BO until Q queries
  run_bo_refine(..., bo_queries=Q0,   BO until Q0, then refine Pareto points with
                total_queries=Q)      search.walk(selector) until Q (rounds over
                                      newly found Pareto points)
"""
import time
import numpy as np

from common import ALL_PARAMS, get_param_values, sample_random_config, pareto_mask, hypervolume
from search import walk, DEFAULTS

NAMES = list(ALL_PARAMS.keys())
RANGES = {n: (min(get_param_values(n)), max(get_param_values(n))) for n in NAMES}


def to_vec(c):
    return np.array([(c[n] - RANGES[n][0]) / (RANGES[n][1] - RANGES[n][0] + 1e-8) for n in NAMES])


def _gp_subset(archive, cap, rng):
    """Training set for the GPs: all Pareto points + most recent + random rest, <= cap."""
    n = len(archive.cpis)
    if n <= cap:
        return np.arange(n)
    P = np.c_[archive.cpis, archive.areas]
    keep = set(np.where(pareto_mask(P))[0].tolist()) | set(range(n - cap // 4, n))
    rest = [i for i in range(n) if i not in keep]
    k = max(0, cap - len(keep))
    keep |= set(rng.choice(rest, size=min(k, len(rest)), replace=False).tolist())
    return np.array(sorted(keep))


def run_bo(S, archive, seed, query_budget, n_initial=30, n_cand=500, log_every=50,
           batch=10, gp_cap=600):
    """batch: points proposed per GP fit (top-`batch` acquisition scores);
    gp_cap: max GP training points. Both keep BO tractable at thousands of queries
    and are identical for bo / bo_scalar / bo_mscs."""
    from sklearn.gaussian_process import GaussianProcessRegressor   # only BO needs sklearn
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    rng = np.random.RandomState(seed)
    t0 = time.time()
    tries = 0
    while len(archive.cpis) < n_initial and tries < 3 * n_initial:
        tries += 1
        archive.add(sample_random_config(rng), "bo_init")
    while S.n_queries < query_budget:
        idx = _gp_subset(archive, gp_cap, rng)
        X = np.array([to_vec(archive.configs[i]) for i in idx])
        kw = dict(kernel=ConstantKernel() * Matern(nu=2.5), n_restarts_optimizer=2,
                  normalize_y=True, random_state=seed)
        gp_c = GaussianProcessRegressor(**kw).fit(X, np.array(archive.cpis)[idx])
        gp_a = GaussianProcessRegressor(**kw).fit(X, np.array(archive.areas)[idx])
        P = np.c_[archive.cpis, archive.areas]
        F = P[pareto_mask(P)]
        ref = (P[:, 0].max() * 1.1, P[:, 1].max() * 1.1)
        hv0 = hypervolume(F, ref)
        cands = [sample_random_config(rng) for _ in range(n_cand)]
        V = np.array([to_vec(c) for c in cands])
        mc, sc = gp_c.predict(V, return_std=True)
        ma, sa = gp_a.predict(V, return_std=True)
        score = np.array([hypervolume(np.vstack([F, [mc[i] - 1.5 * sc[i], ma[i] - 1.5 * sa[i]]]), ref)
                          - hv0 + 0.1 * (sc[i] + sa[i]) for i in range(n_cand)])
        before = S.n_queries
        for i in np.argsort(-score)[:batch]:
            if S.n_queries >= query_budget:
                break
            archive.add(cands[i], "bo")
        if S.n_queries == before:            # all duplicates / McPAT failures: avoid a stall
            archive.add(sample_random_config(rng), "bo_rand")
        if log_every and S.n_queries % log_every == 0:
            print(f"    BO q={S.n_queries} pts={len(archive.cpis)} [{time.time()-t0:.0f}s]", flush=True)


def run_bo_refine(S, archive, seed, selector, bo_queries, total_queries,
                  refine_steps=15, max_rounds=10, p=DEFAULTS, cancel="paper", **bo_kw):
    run_bo(S, archive, seed, bo_queries, **bo_kw)
    done = set()
    for rnd in range(max_rounds):
        if S.n_queries >= total_queries:
            break
        P = np.c_[archive.cpis, archive.areas]
        todo = [i for i in np.where(pareto_mask(P))[0] if i not in done]
        if not todo:
            break
        todo.sort(key=lambda i: archive.cpis[i])            # deterministic order
        for i in todo:
            if S.n_queries >= total_queries:
                break
            done.add(i)
            walk(archive.configs[i], S, archive, selector, refine_steps, p, cancel,
                 query_budget=total_queries, tag=f"refine_{selector}_r{rnd}")
        print(f"    refine round {rnd}: q={S.n_queries} pts={len(archive.cpis)}", flush=True)
