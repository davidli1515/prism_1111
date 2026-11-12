#!/usr/bin/env python3
"""Freeze set-pruning decisions, then exhaustively audit small FE-only sets."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import product
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V5 = ROOT / 'mscs_guard_v5'
sys.path.insert(0, str(V5))
from engine import SliceOracle, config_key, compute_w_min, pareto_mask, hypervolume, cpi_samples
from area_store import AreaStore
from common import DEFAULT_CKPT, NORM_CACHE
from config_dse import get_param_values, is_valid_config
from dse_engine import _project_to_valid

BENCHES = ('623.xalancbmk_s', '641.leela_s')
DATA = V5 / 'results_full_01'
METHODS = ('scalar_width', 'mscs_decode', 'mscs_stage_max')
FE = ('fetchBufferSize', 'l1i_assoc')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def region(root):
    """Enumerate the specified finite set without collateral projection edits."""
    candidates, rejected = [], []
    values = [[v for v in get_param_values(p) if v >= root[p]] for p in FE]
    for vals in product(*values):
        c = dict(root, **dict(zip(FE, vals)))
        if c == root:
            continue
        if not is_valid_config(c) or _project_to_valid(c) != c:
            rejected.append(c)
            continue
        assert all(c[k] == root[k] for k in root if k not in FE)
        assert compute_w_min(c) == compute_w_min(root)
        candidates.append(c)
    assert len({config_key(c) for c in candidates}) == len(candidates)
    return candidates, rejected


def root_bounds(mscs, width):
    m = np.asarray(mscs, dtype=np.float32)
    assert m.ndim == 2 and m.shape[1] == 16 and np.isfinite(m).all()
    s = m.reshape(-1, 4, 4)
    assert np.allclose(s.sum(-1), 1., atol=2e-6)
    ci = cpi_samples(m, width).astype(float)
    fp = s[:, :, 1].astype(float)
    cs = np.minimum(1. / (np.maximum(s[:, :, 0], .01)*width+1e-8), 100.)
    floor = 1. / width
    return {
        'scalar_width': floor,
        'mscs_decode': max(floor, float(np.mean(ci*(1-fp[:, 0])))),
        'mscs_stage_max': max(floor, float(np.mean(ci*(1-fp.max(1))))),
        'root_cpi': float(ci.mean()),
        'fp_decode_cpi_weighted': float(np.mean(ci*fp[:, 0])/ci.mean()),
        'naive_product_bound': float(ci.mean()*(1-fp[:, 0].mean())),
        'fp_stage_order_violation_slices': int((np.diff(fp, axis=1) > 1e-6).any(1).sum()),
        'max_stage_cpi_relative_spread': float(np.max(np.ptp(cs, axis=1)/cs.mean(1))),
        'n_slices': len(m),
    }


def certificate(front, cpi_lower, area_lower):
    """No descendant CPI argument: dominance of the region's optimistic corner."""
    f = np.asarray(front, dtype=float).reshape(-1, 2)
    witnesses = np.flatnonzero((f[:, 0] <= cpi_lower) & (f[:, 1] <= area_lower))
    return {'prune': bool(len(witnesses)),
            'witness': f[witnesses[0]].tolist() if len(witnesses) else None}


def decide(front, bounds, areas):
    complete = bool(areas) and all(a is not None and np.isfinite(a) and a > 0 for a in areas)
    lower = min(areas) if complete else 0.
    return {'area_lower': float(lower), 'area_complete': complete,
            'methods': {k: certificate(front, bounds[k], lower) for k in METHODS}}


def hvi(front, points, reference):
    if not len(points):
        return 0.
    f = np.asarray(front, float).reshape(-1, 2)
    return max(0., float(hypervolume(np.vstack([f, points]), reference)-hypervolume(f, reference)))


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    contexts, sources, archives = [], {}, {}
    for bench in BENCHES:
        path = DATA / bench / 's42/initial.json'
        d = load(path)
        assert d['checkpoint_sha256'] == sha(DEFAULT_CKPT)
        assert d['seed'] == 42 and max(d['queries']) == 500
        p = np.c_[d['cpis'], d['areas']]
        fi = np.flatnonzero(pareto_mask(p))
        fi = fi[np.argsort(p[fi, 1], kind='stable')]
        sources[str(path)] = sha(path)
        archives[bench] = {'source': str(path), 'front': p[fi].tolist(),
                           'front_indices': fi.tolist(), 'reference': d['reference'],
                           'area_sources': d['area_sources']}
        known = {config_key(c) for c in d['configs']}
        for label, rank in zip(('low', 'middle', 'high'), (0, (len(fi)-1)//2, len(fi)-1)):
            i = int(fi[rank]); root = d['configs'][i]
            candidates, rejected = region(root)
            contexts.append({'id': f'{bench}_{label}', 'bench': bench, 'root_index': i,
                             'root': root, 'root_point': p[i].tolist(),
                             'candidates': candidates, 'rejected': rejected,
                             'known_at_snapshot': [config_key(c) in known for c in candidates]})
    paths = [Path(__file__), HERE/'PROTOCOL.txt', HERE/'test_audit.py', Path(DEFAULT_CKPT),
             V5/'area_store.py', V5/'engine.py', ROOT/'common.py',
             ROOT/'mscs_response_1008/probe.py', ROOT/'mscs_response_1008/response.py',
             ROOT/'mscs_response_1008/safe_area.py', ROOT.parent/'dse_exp/config_dse.py',
             ROOT.parent/'dse_exp/dse_engine.py']
    paths += [Path(NORM_CACHE)/f'18k_s42_{x}.npy' for x in ('mean', 'std')]
    sources.update({str(p): sha(p) for p in paths})
    n = sum(len(x['candidates']) for x in contexts)
    assert n <= 120
    plan = {'seed': 42, 'archives': archives, 'contexts': contexts,
            'source_sha256': sources, 'candidate_count': n,
            'max_distinct_requests': n+len(contexts)}
    save(out/'plan.json', plan)
    print(json.dumps({'phase': 'plan', 'candidates': n,
                      'regions': {x['id']: len(x['candidates']) for x in contexts}}), flush=True)
    return plan


def run(out):
    started = time.monotonic()
    plan = prepare(out)
    area = AreaStore(out/'area_store')
    assert all(a['area_sources'] == area.sources for a in plan['archives'].values())
    unique = {config_key(c): c for ctx in plan['contexts'] for c in [ctx['root']]+ctx['candidates']}
    counters = Counter()
    # Existing safe caches are strictly read-only; only the NEW store is written.
    caches = [V5/'area_store_v1/areas.sqlite',
              ROOT/'mscs_branch_1009/results_s42_02/area_store/areas.sqlite']
    for dbpath in caches:
        with sqlite3.connect(f'file:{dbpath}?mode=ro', uri=True) as db:
            for c in unique.values():
                k = area.key(c)
                if area.lookup(k) is not None:
                    continue
                hit = db.execute('SELECT value FROM areas WHERE key=?', (k,)).fetchone()
                if hit:
                    area._record((k, hit[0], None)); counters['area_safe_cache_hits'] += 1
    missing = [c for c in unique.values() if area.lookup(area.key(c)) is None]
    counters['area_new_configs'] = len(missing)
    def get_areas():
        t = time.monotonic()
        area.prefetch(missing, workers=4)
        return time.monotonic()-t
    oracles, root_records = {}, {}
    # Independent private McPAT jobs overlap with root feature extraction.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(get_areas)
        for ctx in plan['contexts']:
            b = ctx['bench']
            if b not in oracles:
                oracles[b] = SliceOracle(b)
            t = time.monotonic()
            m, cpi, ids = oracles[b].query(ctx['root'])
            bounds = root_bounds(m, compute_w_min(ctx['root']))
            assert np.isclose(cpi, ctx['root_point'][0], rtol=3e-6)
            assert np.isclose(cpi, bounds['root_cpi'], rtol=3e-6)
            root_records[ctx['id']] = {'bounds': bounds, 'mscs': m.tolist(),
                                       'slice_ids': list(ids), 'seconds': time.monotonic()-t}
            print(json.dumps({'phase': 'root', 'id': ctx['id'], 'bounds': bounds}), flush=True)
        area_seconds = future.result()
    decisions = []
    for ctx in plan['contexts']:
        a = [area.lookup(area.key(c)) for c in ctx['candidates']]
        root_area = area.lookup(area.key(ctx['root']))
        assert root_area is not None and abs(root_area-ctx['root_point'][1]) < 1e-6
        record = root_records[ctx['id']]
        decision = decide(plan['archives'][ctx['bench']]['front'], record['bounds'], a)
        decisions.append(dict(ctx, **record, areas=a, **decision))
    save(out/'decisions.json', {'phase': 'frozen_before_descendant_CPI', 'contexts': decisions})
    frozen_sha = sha(out/'decisions.json')
    print(json.dumps({'phase': 'decisions_frozen', 'area_seconds': area_seconds,
                      'area_counts': dict(counters)}), flush=True)

    # Only after the immutable decision file exists may saved future CPI enter.
    index, reuse_sources = {}, {}
    for bench in BENCHES:
        index[bench] = {}
        for name in ('initial', 'scalar', 'scalar_guard', 'mscs_guard'):
            p = DATA/bench/'s42'/f'{name}.json'; d = load(p)
            assert d['checkpoint_sha256'] == sha(DEFAULT_CKPT)
            reuse_sources[str(p)] = sha(p)
            for c, cpi in zip(d['configs'], d['cpis']):
                k = config_key(c)
                if k in index[bench]:
                    assert np.isclose(index[bench][k][0], cpi, rtol=3e-6)
                else:
                    index[bench][k] = (cpi, str(p))
    results = []
    for d in decisions:
        b = d['bench']; labels = []
        f = plan['archives'][b]['front']; ref = plan['archives'][b]['reference']
        for c, a, known in zip(d['candidates'], d['areas'], d['known_at_snapshot']):
            old = index[b].get(config_key(c))
            if old is not None:
                cpi, source = old; counters['candidate_CPI_reused'] += 1
            else:
                _, cpi, ids = oracles[b].query(c)
                assert list(ids) == d['slice_ids']
                source = 'new_frozen_checkpoint_inference'; counters['candidate_CPI_new'] += 1
            labels.append({'config': c, 'cpi': float(cpi), 'area': a,
                           'known_at_snapshot': known, 'source': source})
        pts = [[x['cpi'], x['area']] for x in labels if x['area'] is not None]
        gain = hvi(f, pts, ref)
        best = min((x['cpi'] for x in labels), default=None)
        methods = {}
        for name in METHODS:
            lower = d['bounds'][name]
            violations = [j for j, x in enumerate(labels)
                          if x['cpi'] < lower-1e-6*max(1., abs(lower), x['cpi'])]
            pruned = d['methods'][name]['prune']
            methods[name] = {'cpi_lower': lower, 'violations': violations,
                             'max_violation': max([0.]+[lower-x['cpi'] for x in labels]),
                             'prune': pruned, 'false_prune': bool(pruned and gain > 1e-9),
                             'queries_avoided': int(pruned)*sum(not x['known_at_snapshot'] for x in labels),
                             'lost_hv': gain if pruned else 0.}
        # Diagnostic only, never used to make or revise the frozen decision.
        ideal = certificate(f, best, d['area_lower']) if best is not None else {'prune': False}
        result = {'id': d['id'], 'bench': b, 'n': len(labels),
                  'area_complete': d['area_complete'], 'area_lower': d['area_lower'],
                  'area_increment_min': d['area_lower']-d['root_point'][1] if labels and d['area_complete'] else None,
                  'best_actual_cpi': best, 'root_cpi': d['root_point'][0],
                  'region_hvi': gain, 'oracle_corner_prunable': ideal['prune'],
                  'methods': methods, 'labels': labels}
        save(out/f"audit_{d['id']}.json", result)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'labels'}), flush=True)
    summary = {'regions': len(results), 'nonempty_regions': sum(x['n'] > 0 for x in results),
               'candidates': sum(x['n'] for x in results), 'counters': dict(counters),
               'physical_forward_configs': sum(o.n_queries for o in oracles.values()),
               'area_phase_seconds': area_seconds, 'area_errors': area.errors,
               'descendant_CPI_reuse_source_sha256': reuse_sources,
               'decisions_sha256': frozen_sha, 'elapsed_seconds': time.monotonic()-started,
               'methods': {}}
    for name in METHODS:
        rows = [r['methods'][name] for r in results]
        summary['methods'][name] = {'violations': sum(len(r['violations']) for r in rows),
                                   'pruned_regions': sum(r['prune'] for r in rows),
                                   'false_prunes': sum(r['false_prune'] for r in rows),
                                   'queries_avoided': sum(r['queries_avoided'] for r in rows),
                                   'lost_hv': sum(r['lost_hv'] for r in rows)}
    assert sum(o.n_queries for o in oracles.values()) <= plan['max_distinct_requests']
    assert sha(out/'decisions.json') == frozen_sha
    assert all(sha(p) == h for p, h in plan['source_sha256'].items())
    summary['preservation_and_freeze_checks_passed'] = True
    save(out/'summary.json', summary)
    print(json.dumps({'phase': 'complete', **summary}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=HERE/'results_s42_01')
    args = ap.parse_args()
    run(args.out.resolve())
