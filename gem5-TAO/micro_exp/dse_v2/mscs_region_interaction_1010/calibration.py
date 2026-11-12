"""Nested group validation and train-only, root-consistent pressure shuffling."""
import numpy as np
from sklearn.model_selection import GroupKFold

ALPHAS = (1e-4, .001, .01, .1, 1., 10., 100.)
ARMS = ('scalar', 'mscs', 'shuffled', 'constant')


def weights(rows, ids):
    """Equal groups, then equal roots, then equal scopes within a root."""
    groups = {rows[i]['group'] for i in ids}
    out = []
    for i in ids:
        g, root = rows[i]['group'], rows[i]['root_id']
        nr = len({rows[j]['root_id'] for j in ids if rows[j]['group'] == g})
        nc = sum(rows[j]['root_id'] == root for j in ids)
        out.append(1/(len(groups)*nr*nc))
    return np.asarray(out)


def shuffle_pressure(data, rows, ids):
    """Shuffle roots within benchmark and WITHIN this train/validation partition."""
    rng = np.random.RandomState(42)
    donor = {}
    for b in sorted({rows[i]['bench'] for i in ids}):
        roots = sorted({rows[i]['root_id'] for i in ids if rows[i]['bench'] == b})
        donor.update(dict(zip(roots, rng.permutation(roots).tolist())))
    lookup = {rows[i]['root_id']: i for i in ids}
    return np.array([data['pressure'][lookup[donor[rows[i]['root_id']]]] for i in ids]), donor


def design(data, rows, ids, rep, arm, pressure_stats=None):
    blocks = [data['compact'][ids]]
    if rep == 'rich':
        blocks += [data['arch'][ids], data['detail'][ids]]
    donors = {}
    if arm != 'scalar':
        p = data['pressure'][ids]
        if arm == 'shuffled':
            p, donors = shuffle_pressure(data, rows, ids)
        if pressure_stats is None:
            w = weights(rows, ids)
            pm = w @ p
            ps = np.sqrt(w @ ((p-pm)**2))
            ps[ps < 1e-12] = 1.
            pressure_stats = (pm, ps)
        p = (p-pressure_stats[0])/pressure_stats[1]
        blocks.append(np.einsum('ni,nj->nij', p, data['direction'][ids]).reshape(len(ids), -1))
    sizes = [v.shape[1] for v in blocks]
    return np.concatenate(blocks, axis=1), sizes, donors, pressure_stats


def ridge_fit(x, y, w, sizes, alpha):
    # One normalized block for compact features; separate arch/CPI/pressure blocks.
    # Each column is standardized before block scaling; tiny raw pressure variance
    # therefore does not silently remove the added block.
    w = np.asarray(w, float); w = w/w.sum()
    mean = w @ x
    scale = np.sqrt(w @ ((x-mean)**2))
    scale[scale < 1e-12] = 1.
    blockscale = np.concatenate([np.full(n, np.sqrt(n)) for n in sizes])
    z = (x-mean)/scale/blockscale
    ym = float(w @ y)
    sw = np.sqrt(w*len(y))
    zw, yw = z*sw[:, None], (y-ym)*sw
    coef = zw.T @ np.linalg.solve(zw @ zw.T+alpha*np.eye(len(y)), yw)
    return {'mean': mean, 'scale': scale, 'blockscale': blockscale,
            'coef': coef, 'intercept': ym, 'alpha': alpha}


def predict(fit, x, cap):
    return np.clip(((x-fit['mean'])/fit['scale']/fit['blockscale']) @ fit['coef']+fit['intercept'], 0., cap)


def folds(rows, ids, n):
    groups = np.array([rows[i]['group'] for i in ids])
    n = min(n, len(set(groups)))
    if n < 2:
        raise ValueError('Not enough configuration groups')
    for a, b in GroupKFold(n_splits=n).split(ids, groups=groups):
        tr, va = np.asarray(ids)[a], np.asarray(ids)[b]
        assert not {rows[i]['group'] for i in tr} & {rows[i]['group'] for i in va}
        yield tr, va


def select(data, rows, y, ids, rep, arm):
    losses = []
    splits = list(folds(rows, ids, 3))
    for alpha in ALPHAS:
        pp = {}
        for tr, va in splits:
            x, sizes, _, stats = design(data, rows, tr, rep, arm)
            fit = ridge_fit(x, y[tr], weights(rows, tr), sizes, alpha)
            xv, _, _, _ = design(data, rows, va, rep, arm, stats)
            pv = predict(fit, xv, data['cap'][va])
            pp.update(zip(va.tolist(), pv.tolist()))
        loss = float(weights(rows, ids) @ np.abs(y[ids]-[pp[i] for i in ids]))
        losses.append({'alpha': alpha, 'mae': loss})
    best = min(losses, key=lambda r: (r['mae'], -r['alpha']))
    x, sizes, donor, stats = design(data, rows, ids, rep, arm)
    fit = ridge_fit(x, y[ids], weights(rows, ids), sizes, best['alpha'])
    fit['pressure_stats'] = stats
    return fit, {'selected_alpha': best['alpha'], 'inner_mae': best['mae'], 'scores': losses,
                 'block_sizes': sizes, 'train_donors': donor,
                 'pressure_stats': None if stats is None else [v.tolist() for v in stats],
                 'inner_splits': [{'train': a.tolist(), 'valid': b.tolist()} for a, b in splits]}


def family_constant(rows, y, tr, va, caps):
    ww = weights(rows, tr)
    overall = float(ww @ y[tr])
    values = {}
    for pattern in ('buffer', 'assoc', 'joint'):
        ix = np.array([j for j, i in enumerate(tr) if rows[i]['pattern'] == pattern], dtype=int)
        values[pattern] = float(ww[ix] @ y[tr[ix]]/ww[ix].sum()) if len(ix) else overall
    return np.array([min(values[rows[i]['pattern']], cap) for i, cap in zip(va, caps)]), values


def fit_predict(data, rows, y, tr, va):
    # Representation choice is nested and scalar-only; MSCS cannot choose an
    # easier scalar comparator using the outer validation labels.
    choices = {r: select(data, rows, y, tr, r, 'scalar') for r in ('compact', 'rich')}
    rep = min(choices, key=lambda r: (choices[r][1]['inner_mae'], r != 'compact'))
    fitted = {'scalar': choices[rep]}
    fitted.update({a: select(data, rows, y, tr, rep, a) for a in ('mscs', 'shuffled')})
    out, info = {}, {'representation': rep, 'scalar_representation_scores': {r: v[1]['inner_mae'] for r, v in choices.items()}}
    for arm, (fit, report) in fitted.items():
        xv, _, donors, _ = design(data, rows, va, rep, arm, fit['pressure_stats'])
        out[arm] = predict(fit, xv, data['cap'][va])
        info[arm] = dict(report, validation_donors=donors,
                         train_scaler_mean=fit['mean'].tolist(), train_scaler_scale=fit['scale'].tolist())
    out['constant'], info['constant'] = family_constant(rows, y, tr, va, data['cap'][va])
    return out, info


def metrics(rows, y, preds, ids=None):
    ids = np.arange(len(rows)) if ids is None else np.asarray(ids)
    w = weights(rows, ids)
    return {arm: {'mae': float(w @ np.abs(y[ids]-np.asarray(p)[ids])),
                  'max_underprediction': float(np.maximum(y[ids]-np.asarray(p)[ids], 0).max())}
            for arm, p in preds.items()}


def gate(main, by_bench):
    s, m, p, c = [main[a] for a in ARMS]
    conditions = {
        'scalar_beats_family_constant': s['mae'] < c['mae'],
        'mscs_10pct_better_than_scalar': m['mae'] <= .9*s['mae'],
        'mscs_10pct_better_than_shuffle': m['mae'] <= .9*p['mae'],
        'underprediction_not_worse': m['max_underprediction'] <= s['max_underprediction']+1e-8,
        'gcc_and_mcf_both_improve': all(b in by_bench and by_bench[b]['mscs']['mae'] < by_bench[b]['scalar']['mae']
                                       for b in ('602.gcc_s', '605.mcf_s')),
    }
    return {'pass': bool(all(conditions.values())), 'conditions': conditions}


def summarize(rows, y, preds):
    main = metrics(rows, y, preds)
    by = {field: {v: metrics(rows, y, preds, [i for i, r in enumerate(rows) if r[field] == v])
                  for v in sorted({r[field] for r in rows})} for field in ('bench', 'pattern')}
    return {'metrics': main, 'by_benchmark': by['bench'], 'by_pattern': by['pattern'],
            'gate': gate(main, by['bench'])}


def nested_cv(data, rows, progress=None):
    y = np.array([r['gain'] for r in rows])
    preds = {a: np.full(len(y), np.nan) for a in ARMS}
    reports = []
    for k, (tr, va) in enumerate(folds(rows, np.arange(len(y)), 5)):
        pp, info = fit_predict(data, rows, y, tr, va)
        for a in ARMS:
            preds[a][va] = pp[a]
        reports.append(dict(fold=k, train=tr.tolist(), valid=va.tolist(), **info))
        if progress:
            progress(k+1)
    assert all(np.isfinite(v).all() for v in preds.values())
    return dict(summarize(rows, y, preds), predictions={a: v.tolist() for a, v in preds.items()}, folds=reports)
