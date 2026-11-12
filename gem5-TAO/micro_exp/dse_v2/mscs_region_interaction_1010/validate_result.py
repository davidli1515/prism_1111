#!/usr/bin/env python3
"""Independent set/label/weight checks and selected-alpha primal Ridge replay.

Does not import either experiment calibration or experiment region enumeration.
"""
from collections import Counter
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'dse_exp'))
from config_dse import ALL_PARAMS, get_param_values, compute_w_min, is_valid_config

FE = ('fetchBufferSize', 'l1i_assoc')


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def key(c):
    return tuple(c[k] for k in ALL_PARAMS)


def row_weights(rows, ids):
    groups = {}
    for i in ids:
        r = rows[i]
        groups.setdefault(r['group'], {}).setdefault(r['root_id'], []).append(i)
    ww = {}
    for g in groups.values():
        for rr in g.values():
            for i in rr:
                ww[i] = 1/(len(groups)*len(g)*len(rr))
    return np.array([ww[i] for i in ids])


def check_sets(roots, rows):
    for row in rows:
        root = roots[row['root_id']]['root']; domain = row['domain']
        assert set(domain) == set(FE)
        points = []
        for b, a in product(domain[FE[0]], domain[FE[1]]):
            c = dict(root, fetchBufferSize=b, l1i_assoc=a)
            assert is_valid_config(c) and max(c['l1i_assoc'], c['l1d_assoc']) <= c['l1_size']
            assert all(c[p] in get_param_values(p) for p in ALL_PARAMS)
            if c != root:
                points.append(c)
        assert points == row['candidates']
        assert len({key(c) for c in points}) == len(points) == len(row['descendant_cpis'])
        assert all(c[p] == root[p] for c in points for p in root if p not in FE)
        assert all(compute_w_min(c) == compute_w_min(root) for c in points)
        yy = row['descendant_cpis']; c0 = roots[row['root_id']]['root_cpi']
        assert np.isclose(row['gain'], max(0, (c0-min(yy))/c0), atol=1e-12)
        src = roots[row['root_id']]
        if 'descendant_cpis' in src:
            labels = {key(c): v for c, v in zip(src['candidates'], src['descendant_cpis'])}
            np.testing.assert_allclose(yy, [labels[key(c)] for c in points], rtol=0, atol=0)
        varying = [len(domain[p]) > 1 for p in FE]
        pattern = 'joint' if all(varying) else 'buffer' if varying[0] else 'assoc'
        assert row['pattern'] == pattern
    # Every near-root relation, including transitive ones, is covered by group IDs.
    for i, a in enumerate(rows):
        for b in rows[:i]:
            ca, cb = roots[a['root_id']]['root'], roots[b['root_id']]['root']
            if sum(ca[k] != cb[k] for k in ALL_PARAMS if k not in FE) <= 1:
                assert a['group'] == b['group']


def reference_features(roots, rows):
    output = {k: [] for k in ('compact', 'arch', 'detail', 'pressure', 'direction', 'cap')}
    for r in rows:
        root = roots[r['root_id']]; c = root['root']; domain = r['domain']
        m = np.array(root['mscs'], dtype=np.float32).reshape(50, 4, 4)
        base = m[:, :, 0]
        cs = np.minimum(1/(np.maximum(base, .01)*compute_w_min(c)+1e-8), 100.).astype(float)
        ci = np.mean(cs, axis=1)
        assert np.isclose(ci.mean(), root['root_cpi'], rtol=3e-6)
        stats = np.array([np.log1p(ci.mean()), np.std(ci)/ci.mean(),
                          np.log(np.quantile(ci, .9)/np.quantile(ci, .1)),
                          np.mean((cs.max(1)-cs.min(1))/ci), np.log2(compute_w_min(c)),
                          np.log2(c['l1_size']), np.log2(c[FE[0]]), np.log2(c[FE[1]])])
        spans = [np.log2(max(domain[p])/c[p]) for p in FE]
        db, da = spans
        direction = np.array([float(db > 0), float(da > 0), float(db > 0 and da > 0), db, da, db*da])
        scope = np.r_[direction, np.log1p(len(r['candidates'])), [np.log2(max(domain[p])) for p in FE]]
        output['compact'].append(np.r_[stats, scope, [x*d for x in stats[:4] for d in direction]])
        output['arch'].append([get_param_values(p).index(c[p])/(len(get_param_values(p))-1) for p in ALL_PARAMS])
        output['detail'].append(np.quantile(np.log1p(cs), np.linspace(0, 1, 50), axis=0).ravel())
        comp = m[:, :, 1:].astype(float); total = comp.sum(2, keepdims=True)
        comp = np.divide(comp, total, out=np.zeros_like(comp), where=total > 0)
        output['pressure'].append([float(ci @ comp[:, s, k]/ci.sum()) for s, k in ((0, 0), (3, 0), (0, 2), (3, 1))])
        output['direction'].append(direction)
        output['cap'].append(max(0, 1-1/(compute_w_min(c)*ci.mean())))
    return {k: np.array(v) for k, v in output.items()}


def replay_fold(data, rows, y, report):
    tr, va = np.array(report['train']), np.array(report['valid'])
    assert not {rows[i]['group'] for i in tr} & {rows[i]['group'] for i in va}
    w = row_weights(rows, tr)
    answer = {}
    for arm in ('scalar', 'mscs', 'shuffled'):
        info = report[arm]
        for split in info['inner_splits']:
            a, b = split['train'], split['valid']
            assert set(a)|set(b) == set(tr.tolist())
            assert not {rows[i]['group'] for i in a} & {rows[i]['group'] for i in b}
        xx = []
        ps = None
        for which, ids in (('train', tr), ('validation', va)):
            blocks = [data['compact'][ids]]
            if report['representation'] == 'rich':
                blocks += [data['arch'][ids], data['detail'][ids]]
            if arm != 'scalar':
                p = data['pressure'][ids]
                if arm == 'shuffled':
                    lookup = {rows[i]['root_id']: i for i in ids}
                    donors = {int(k): v for k, v in info[f'{which}_donors'].items()}
                    assert set(donors) == set(lookup) and sorted(donors.values()) == sorted(lookup)
                    for a, b in donors.items():
                        assert rows[lookup[a]]['bench'] == rows[lookup[b]]['bench']
                    p = np.array([data['pressure'][lookup[donors[rows[i]['root_id']]]] for i in ids])
                if which == 'train':
                    pm = w @ p; pv = np.sqrt(w @ ((p-pm)**2)); pv[pv < 1e-12] = 1.
                    ps = (pm, pv)
                    np.testing.assert_allclose(np.array(info['pressure_stats']), np.array(ps), atol=1e-11)
                p = (p-ps[0])/ps[1]
                blocks.append(np.array([np.outer(a, b).ravel() for a, b in zip(p, data['direction'][ids])]))
            assert [v.shape[1] for v in blocks] == info['block_sizes']
            xx.append(np.concatenate(blocks, axis=1))
        x, xv = xx
        mean = w @ x; scale = np.sqrt(w @ ((x-mean)**2)); scale[scale < 1e-12] = 1.
        np.testing.assert_allclose(mean, info['train_scaler_mean'], atol=1e-11)
        np.testing.assert_allclose(scale, info['train_scaler_scale'], atol=1e-11)
        block = np.concatenate([np.repeat(np.sqrt(n), n) for n in info['block_sizes']])
        x = (x-mean)/scale/block; xv = (xv-mean)/scale/block
        ym = w @ y[tr]; ww = len(tr)*w
        # Primal normal equations, independently of the production dual solve.
        coef = np.linalg.solve(x.T @ (ww[:, None]*x)+info['selected_alpha']*np.eye(x.shape[1]),
                               x.T @ (ww*(y[tr]-ym)))
        answer[arm] = np.clip(xv @ coef+ym, 0, data['cap'][va])
    constants = {}
    for p in ('buffer', 'assoc', 'joint'):
        loc = np.array([j for j, i in enumerate(tr) if rows[i]['pattern'] == p], dtype=int)
        constants[p] = float(w[loc] @ y[tr[loc]]/w[loc].sum()) if len(loc) else float(w @ y[tr])
        assert np.isclose(constants[p], report['constant'][p], atol=1e-12)
    answer['constant'] = np.array([min(constants[rows[i]['pattern']], data['cap'][i]) for i in va])
    return answer


def compare_metrics(rows, preds, report):
    y = np.array([r['gain'] for r in rows])
    def one(ids, metric):
        w = row_weights(rows, ids)
        for arm, p in preds.items():
            pp = np.array(p)[ids]
            assert np.isclose(w @ abs(y[ids]-pp), metric[arm]['mae'], atol=1e-12)
            assert np.isclose(np.maximum(y[ids]-pp, 0).max(), metric[arm]['max_underprediction'], atol=1e-12)
    one(np.arange(len(rows)), report['metrics'])
    for name, field in (('by_benchmark', 'bench'), ('by_pattern', 'pattern')):
        for value, metric in report[name].items():
            one(np.array([i for i, r in enumerate(rows) if r[field] == value]), metric)
    m = report['metrics']; b = report['by_benchmark']
    cond = [m['scalar']['mae'] < m['constant']['mae'],
            m['mscs']['mae'] <= .9*m['scalar']['mae'], m['mscs']['mae'] <= .9*m['shuffled']['mae'],
            m['mscs']['max_underprediction'] <= m['scalar']['max_underprediction']+1e-8,
            all(b[x]['mscs']['mae'] < b[x]['scalar']['mae'] for x in ('602.gcc_s', '605.mcf_s'))]
    assert bool(all(cond)) == report['gate']['pass']


def verify_development(out):
    ds, report = read(out/'dataset.json'), read(out/'development.json')
    roots, rows = ds['roots'], ds['rows']
    assert len(roots) == 24 and len(rows) == 40
    assert Counter(r['pattern'] for r in rows) == {'buffer': 10, 'assoc': 22, 'joint': 8}
    check_sets(roots, rows)
    data = reference_features(roots, rows)
    saved = read(out/'features.json')
    for k, values in data.items():
        np.testing.assert_allclose(values, saved[k], atol=2e-12)
    # Replay using exact saved float arrays; independent feature reconstruction above
    # can have harmless last-bit differences amplified by nearly constant columns.
    data = {k: np.array(v) for k, v in saved.items()}
    y = np.array([r['gain'] for r in rows]); seen = []
    for f in report['folds']:
        seen += f['valid']
        assert set(f['train']) | set(f['valid']) == set(range(len(rows)))
        replay = replay_fold(data, rows, y, f)
        for arm, value in replay.items():
            np.testing.assert_allclose(value, np.array(report['predictions'][arm])[f['valid']], atol=2e-9, rtol=1e-7)
    assert sorted(seen) == list(range(len(rows)))
    compare_metrics(rows, report['predictions'], report)
    return {'passed': True, 'independent_cartesian_and_label_checks': True,
            'independent_feature_reconstruction': True, 'independent_primal_prediction_replay': True,
            'group_weights_and_metrics': True, 'source_files': {str(out/n): digest(out/n)
            for n in ('dataset.json', 'features.json', 'development.json')}}


def verify_all(out):
    earlier = read(out/'development_validation.json')
    assert earlier['passed']
    assert all(digest(p) == h for p, h in earlier['source_files'].items())
    summary = read(out/'summary.json')
    assert all(digest(p) == h for p, h in summary['source_sha256'].items())
    follow = summary['confirmation']
    if follow['phase'] == 'confirmation_complete':
        ds = read(out/'dataset.json')
        rr = read(out/'fresh_roots.json'); labels = read(out/'fresh_labels.json')
        predicted = read(out/'fresh_predictions.json'); c = read(out/'confirmation.json')
        assert len(rr) == 4 and len(labels) == 12
        assert digest(out/'fresh_predictions.json') == c['predictions_sha256']
        assert digest(out/'fresh_plan.json') == c['plan_sha256']
        all_roots, all_rows = ds['roots']+rr, ds['rows']+labels
        check_sets(all_roots, all_rows)
        data = reference_features(all_roots, all_rows)
        # Reconstruct through production feature builder only to retain exact float
        # rounding; the independent reference has already been compared explicitly.
        from region_data import make_arrays
        exact = make_arrays(all_roots, all_rows)
        for k in data:
            np.testing.assert_allclose(data[k], exact[k], atol=2e-12)
        n = len(ds['rows'])
        model = dict(predicted['models'], train=list(range(n)), valid=list(range(n, len(all_rows))))
        result = replay_fold(exact, all_rows, np.array([r['gain'] for r in all_rows]), model)
        for arm in result:
            np.testing.assert_allclose(result[arm], predicted['predictions'][arm], atol=2e-9, rtol=1e-7)
        compare_metrics(labels, predicted['predictions'], c)
        assert c['physical_forward_configs'] <= 36 and c['counts'].get('new_candidate_cpi', 0) <= 32
    else:
        assert follow['new_candidate_cpi'] == follow['root_queries'] == 0
        assert not (out/'fresh_predictions.json').exists()
    return {'passed': True, 'development_validation_reused_unchanged': True,
            'confirmation_checked': follow['phase'], 'all_preserved_sources_unchanged': True}


if __name__ == '__main__':
    out = HERE/'results_s42_01'
    result = verify_all(out)
    with (out/'validation.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
