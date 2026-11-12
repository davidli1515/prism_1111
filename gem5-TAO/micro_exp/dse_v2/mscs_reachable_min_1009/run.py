#!/usr/bin/env python3
"""Minimal transfer test: saved gcc sets -> three untouched mcf sets."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'mscs_bounds_1009'))
from audit import region, config_key, compute_w_min, pareto_mask, SliceOracle, DEFAULT_CKPT, NORM_CACHE
from config_dse import ALL_PARAMS, get_param_values

TRAIN, TEST = '602.gcc_s', '605.mcf_s'
DATA = ROOT/'mscs_guard_v5/results_full_01'
OUT = HERE/'results_s42_01'
FE = ('fetchBufferSize', 'l1i_assoc')


def load(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def group(c):
    return tuple((k, v) for k, v in config_key(c) if k not in FE)


def nearby(a, b):
    return sum(a[k] != b[k] for k in a if k not in FE) <= 1


def feature_blocks(c, m):
    """Only root data; no descendant outcomes or benchmark ID."""
    m = np.asarray(m, np.float32).reshape(-1, 4, 4)
    assert m.shape[0] == 50 and np.isfinite(m).all()
    w = compute_w_min(c)
    cs = np.minimum(1/(np.maximum(m[:, :, 0], .01)*w+1e-8), 100.)
    arch = np.array([(get_param_values(k).index(c[k])/(len(get_param_values(k))-1)) for k in ALL_PARAMS])
    candidates, _ = region(c)
    ranges = np.array([len(candidates)/19.] + [v for k in FE for v in
        (min(x[k] for x in candidates)/max(get_param_values(k)),
         max(x[k] for x in candidates)/max(get_param_values(k)),
         (max(x[k] for x in candidates)-c[k])/max(get_param_values(k)))])
    q = np.linspace(0, 1, 50)
    scalar = np.r_[np.quantile(np.log1p(cs), q, axis=0).ravel(), np.log1p(cs.mean())]
    pressure = m[:, :, 1:].astype(float)
    composition = pressure/pressure.sum(2, keepdims=True)
    extra = np.quantile(composition, q, axis=0).ravel()
    blocks = [arch, ranges, scalar, extra]
    return [b/np.sqrt(len(b)) for b in blocks]


def predict(x, y, xt, cap):
    model = Ridge(alpha=1.0)
    model.fit(x, y)
    return np.clip(model.predict(xt), 0., cap)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.monotonic(); sources = {}; archives = {}
    ckpt_sha = sha(DEFAULT_CKPT)
    for b in (TRAIN, TEST):
        archives[b] = {}
        for name in ('initial', 'scalar', 'scalar_guard', 'mscs_guard'):
            p = DATA/b/'s42'/f'{name}.json'
            d = load(p); assert d['checkpoint_sha256'] == ckpt_sha
            archives[b][name] = d; sources[str(p)] = sha(p)
    # Only configurations, root areas and BO-snapshot CPI select regions.
    d = archives[TEST]['initial']; p = np.c_[d['cpis'], d['areas']]
    f = np.flatnonzero(pareto_mask(p)); f = sorted(f, key=lambda i: p[i, 1])
    eligible = [int(i) for i in f if len(region(d['configs'][i])[0]) == 3]
    assert len(eligible) >= 3
    picks = [eligible[i] for i in (0, (len(eligible)-1)//2, len(eligible)-1)]
    tests = [{'bench': TEST, 'root': d['configs'][i], 'root_cpi': d['cpis'][i],
              'source_index': i, 'candidates': region(d['configs'][i])[0]} for i in picks]
    available = {config_key(c) for a in archives[TRAIN].values() for c in a['configs']}
    complete = {}
    for name in ('initial', 'scalar', 'scalar_guard'):
        a = archives[TRAIN][name]
        for c, area, cpi in zip(a['configs'], a['areas'], a['cpis']):
            if any(nearby(c, t['root']) for t in tests):
                continue
            cs, _ = region(c)
            if not 3 <= len(cs) <= 9 or not all(config_key(x) in available for x in cs):
                continue
            rank = (-len(cs), area, config_key(c)); g = group(c)
            if g not in complete or rank < complete[g][0]:
                complete[g] = (rank, {'bench': TRAIN, 'root': c, 'root_cpi': cpi, 'candidates': cs})
    trains = [v[1] for _, v in sorted(complete.items())]
    assert len(trains) >= 3 and len(tests) == 3
    paths = [Path(__file__), HERE/'PROTOCOL.txt', HERE/'test_intent.py', Path(DEFAULT_CKPT),
             ROOT/'mscs_bounds_1009/audit.py', ROOT/'mscs_response_1008/probe.py',
             ROOT/'mscs_response_1008/response.py', ROOT/'common.py',
             ROOT.parent/'dse_exp/config_dse.py', ROOT.parent/'dse_exp/dse_engine.py']
    paths += [Path(NORM_CACHE)/f'18k_s42_{n}.npy' for n in ('mean', 'std')]
    sources.update({str(p): sha(p) for p in paths})
    save(OUT/'plan.json', {'seed': 42, 'train': trains, 'test': tests,
                         'max_new_descendant_cpi': 9, 'source_sha256': sources})
    print(json.dumps({'phase': 'frozen_plan', 'train_sets': len(trains), 'test_sets': 3,
                      'test_configurations': 9}), flush=True)
    oracles = {}; roots = []; train_labels = []
    train_cpi = {config_key(c): float(v) for a in archives[TRAIN].values()
                 for c, v in zip(a['configs'], a['cpis'])}
    for split, rows in (('train', trains), ('test', tests)):
        for j, row in enumerate(rows):
            b = row['bench']
            if b not in oracles:
                oracles[b] = SliceOracle(b)
            m, cpi, ids = oracles[b].query(row['root'])
            assert np.isclose(cpi, row['root_cpi'], rtol=3e-6)
            blocks = feature_blocks(row['root'], m)
            roots.append({'split': split, 'index': j, 'mscs': m.tolist(), 'slice_ids': list(ids),
                          'blocks': [x.tolist() for x in blocks], 'config': row['root']})
            if split == 'train':
                labels = [train_cpi[config_key(c)] for c in row['candidates']]
                train_labels.append({'descendant_cpis': labels,
                                     'gain': max(0., 1-min(labels)/row['root_cpi'])})
    save(OUT/'roots.json', roots)
    save(OUT/'training_labels.json', train_labels)
    scalar = np.array([np.concatenate(r['blocks'][:3]) for r in roots])
    extra = np.array([r['blocks'][3] for r in roots]); n = len(trains)
    y = np.array([x['gain'] for x in train_labels])
    cap = np.array([max(0., 1-1/(compute_w_min(t['root'])*t['root_cpi'])) for t in tests])
    rng = np.random.RandomState(42)
    tr_perm, te_perm = rng.permutation(n), rng.permutation(3)
    designs = {'scalar': scalar, 'mscs': np.c_[scalar, extra],
               'shuffled': np.c_[scalar, np.r_[extra[:n][tr_perm], extra[n:][te_perm]]]}
    predictions = {name: predict(x[:n], y, x[n:], cap).tolist() for name, x in designs.items()}
    save(OUT/'predictions.json', {'predictions': predictions, 'phase': 'before_test_labels',
                                 'train_permutation': tr_perm.tolist(), 'test_permutation': te_perm.tolist()})
    frozen = sha(OUT/'predictions.json')
    print(json.dumps({'phase': 'predictions_frozen', 'train_target_range': [float(y.min()), float(y.max())]}), flush=True)
    # First and only access to held-out descendant CPI values is BELOW this line.
    test_cpi = {config_key(c): (float(v), name) for name, a in archives[TEST].items()
                for c, v in zip(a['configs'], a['cpis'])}
    truth, evaluated = [], []; counts = Counter(); new_configs = set()
    for j, row in enumerate(tests):
        records = []
        for c in row['candidates']:
            k = config_key(c)
            if k in test_cpi:
                cpi, source = test_cpi[k]; counts['reused_candidate_cpi'] += 1
            else:
                assert len(new_configs) < 9
                _, cpi, ids = oracles[TEST].query(c); new_configs.add(k)
                assert list(ids) == roots[n+j]['slice_ids']
                source = 'new_frozen_checkpoint'; counts['new_candidate_cpi'] += 1
            records.append({'config': c, 'cpi': cpi, 'source': source})
        gain = max(0., 1-min(x['cpi'] for x in records)/row['root_cpi'])
        truth.append(gain); evaluated.append({'root': row['root'], 'root_cpi': row['root_cpi'],
                                             'gain': gain, 'records': records})
    save(OUT/'test_labels.json', evaluated)
    target = np.array(truth); stats = {}
    for name, pred in predictions.items():
        pred = np.array(pred)
        stats[name] = {'mae': float(np.mean(abs(pred-target))),
                       'max_underprediction': float(np.maximum(target-pred, 0).max()),
                       'absolute_errors': abs(pred-target).tolist(), 'predictions': pred.tolist()}
    sm, ss, sp = [stats[k] for k in ('mscs', 'scalar', 'shuffled')]
    wins = int(sum(a < min(b, c)-1e-9 for a, b, c in zip(sm['absolute_errors'], ss['absolute_errors'], sp['absolute_errors'])))
    gate = bool(np.ptp(y) > 1e-6 and ss['mae'] > 1e-6 and
                sm['mae'] <= .9*min(ss['mae'], sp['mae']) and wins >= 2 and
                sm['max_underprediction'] <= ss['max_underprediction']+1e-6)
    assert sha(OUT/'predictions.json') == frozen
    assert all(sha(p) == h for p, h in sources.items())
    assert all(not nearby(a['root'], b['root']) for a in trains for b in tests)
    result = {'train_sets': n, 'test_sets': 3, 'true_gains': truth, 'methods': stats,
              'consistent_wins': wins, 'pilot_gate_passed': gate,
              'counts': dict(counts), 'physical_forward_configs': sum(o.n_queries for o in oracles.values()),
              'predictions_sha256': frozen, 'preserved_sources': len(sources),
              'wall_seconds': time.monotonic()-start,
              'scope': 'one held-out benchmark, three finite sets; no area or DSE saving claim'}
    save(OUT/'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
