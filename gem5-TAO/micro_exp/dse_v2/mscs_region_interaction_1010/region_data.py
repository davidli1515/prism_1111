"""Explicit complete resource sets, saved labels, and non-Base features.

This module never instantiates a surrogate or reads a descendant label in features.
"""
from collections import Counter
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parent / 'dse_exp'))
from config_dse import ALL_PARAMS, compute_w_min, get_param_values, is_valid_config

FE = ('fetchBufferSize', 'l1i_assoc')
PATTERNS = ('buffer', 'assoc', 'joint')
OLD = ROOT / 'mscs_reachable_v2_1009/results_s42_01'
CKPT = ROOT.parent / 'exp1/results/lambda_sweep_1004/t10_c1_m0p01_s42.pt'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def key(c):
    return tuple(c[k] for k in ALL_PARAMS)


def legal(c):
    # Exactly the fixed points of the current validity projection on this grid.
    return (set(c) == set(ALL_PARAMS) and all(c[k] in get_param_values(k) for k in ALL_PARAMS)
            and is_valid_config(c) and max(c['l1i_assoc'], c['l1d_assoc']) <= c['l1_size'])


def near(a, b):
    return sum(a[k] != b[k] for k in ALL_PARAMS if k not in FE) <= 1


def axes(root):
    assert legal(root)
    return {p: [v for v in get_param_values(p) if v >= root[p]
                and (p != 'l1i_assoc' or v <= root['l1_size'])] for p in FE}


def enumerate_set(root, domain):
    assert legal(root) and set(domain) == set(FE)
    for p in FE:
        vals = domain[p]
        assert vals == sorted(set(vals)) and vals[0] == root[p]
        assert all(v in axes(root)[p] for v in vals)
    result = []
    for b, a in product(domain[FE[0]], domain[FE[1]]):
        c = dict(root, fetchBufferSize=b, l1i_assoc=a)
        if c != root:
            assert legal(c) and compute_w_min(c) == compute_w_min(root)
            result.append(c)
    return result


def scopes(root):
    full = axes(root)
    out = []
    for kind in PATTERNS:
        domain = {p: list(full[p]) if (kind == 'joint' or p == FE[PATTERNS.index(kind)])
                  else [root[p]] for p in FE}
        if kind == 'joint' and any(len(v) == 1 for v in domain.values()):
            continue  # Do not duplicate the only nonempty single-resource set.
        candidates = enumerate_set(root, domain)
        if candidates:
            out.append((kind, domain, candidates))
    return out


def connected_groups(roots):
    parent = list(range(len(roots)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(roots):
        for j, b in enumerate(roots[:i]):
            if near(a['root'], b['root']):
                parent[find(i)] = find(j)
    ids = [find(i) for i in range(len(roots))]
    remap = {g: i for i, g in enumerate(sorted(set(ids)))}
    return [remap[g] for g in ids]


def expand(roots):
    groups = connected_groups(roots)
    rows = []
    for ri, r in enumerate(roots):
        expected = enumerate_set(r['root'], axes(r['root']))
        assert len(r['candidates']) == len(r['descendant_cpis'])
        labels = {key(c): float(y) for c, y in zip(r['candidates'], r['descendant_cpis'])}
        assert len(labels) == len(r['candidates'])
        assert set(labels) == {key(c) for c in expected}, 'Incomplete source joint set'
        assert all(np.isfinite(v) and v > 0 for v in labels.values())
        for kind, domain, candidates in scopes(r['root']):
            yy = [labels[key(c)] for c in candidates]
            rows.append({'root_id': ri, 'group': groups[ri], 'bench': r['bench'],
                         'pattern': kind, 'domain': domain, 'candidates': candidates,
                         'descendant_cpis': yy, 'gain': max(0., 1-min(yy)/r['root_cpi'])})
    return rows


def load_saved():
    plan = read(OLD / 'plan.json')
    roots = plan['development']
    fresh, labels = read(OLD / 'fresh_roots.json'), read(OLD / 'fresh_labels.json')
    assert len(fresh) == len(labels) == 6
    for r, lab in zip(fresh, labels):
        assert r['root'] == lab['root'] and r['bench'] == lab['bench']
        assert r['candidates'] == [v['config'] for v in lab['records']]
        roots.append(dict(r, descendant_cpis=[v['cpi'] for v in lab['records']]))
    assert len({(r['bench'], key(r['root'])) for r in roots}) == len(roots) == 24
    rows = expand(roots)
    assert Counter(r['pattern'] for r in rows) == {'buffer': 10, 'assoc': 22, 'joint': 8}
    for r in roots:
        cp, _ = root_features(r['root'], r['mscs'])
        assert np.isclose(cp['cpi'], r['root_cpi'], rtol=3e-6)
        assert len(r['slice_ids']) == 50
    return roots, rows


def root_features(c, m):
    m = np.asarray(m, np.float32).reshape(-1, 4, 4)
    assert m.shape == (50, 4, 4) and np.isfinite(m).all()
    assert (m >= 0).all() and np.allclose(m.sum(2), 1., atol=2e-6)
    cs = np.minimum(1/(np.maximum(m[:, :, 0], .01)*compute_w_min(c)+1e-8), 100.).astype(float)
    ci = cs.mean(1)
    compact = np.array([np.log1p(ci.mean()), ci.std()/ci.mean(),
                        np.log(np.quantile(ci, .9)/np.quantile(ci, .1)),
                        np.mean(np.ptp(cs, axis=1)/ci), np.log2(compute_w_min(c)),
                        np.log2(c['l1_size']), np.log2(c[FE[0]]), np.log2(c[FE[1]])])
    scalar = {'compact': compact, 'detail': np.quantile(np.log1p(cs), np.linspace(0, 1, 50), axis=0).ravel(),
              'arch': np.array([get_param_values(p).index(c[p])/(len(get_param_values(p))-1) for p in ALL_PARAMS]),
              'cpi': float(ci.mean())}
    pressure = m[:, :, 1:].astype(float)
    total = pressure.sum(2, keepdims=True)
    composition = np.divide(pressure, total, out=np.zeros_like(pressure), where=total > 0)
    extra = np.array([np.sum(ci*composition[:, s, k])/ci.sum() for s, k in ((0, 0), (3, 0), (0, 2), (3, 1))])
    return scalar, extra


def features(root, m, domain):
    """All predictors receive R explicitly; descendant CPI is not an argument."""
    candidates = enumerate_set(root, domain)
    assert candidates
    scalar, pressure = root_features(root, m)
    db, da = [np.log2(domain[p][-1]/root[p]) for p in FE]
    direction = np.array([float(db > 0), float(da > 0), float(db > 0 and da > 0), db, da, db*da])
    scope = np.r_[direction, np.log1p(len(candidates)), [np.log2(domain[p][-1]) for p in FE]]
    compact = np.r_[scalar['compact'], scope, np.outer(scalar['compact'][:4], direction).ravel()]
    return {'compact': compact, 'arch': scalar['arch'], 'detail': scalar['detail'],
            'pressure': pressure, 'direction': direction,
            'cap': max(0., 1-1/(compute_w_min(root)*scalar['cpi']))}


def make_arrays(roots, rows):
    f = [features(roots[r['root_id']]['root'], roots[r['root_id']]['mscs'], r['domain']) for r in rows]
    return {k: np.array([x[k] for x in f]) for k in f[0]}
