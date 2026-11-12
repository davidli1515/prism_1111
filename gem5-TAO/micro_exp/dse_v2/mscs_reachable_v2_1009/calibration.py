"""Train-only scaling/model selection with four extra non-Base features."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = (0.001, 0.01, 0.1, 1., 10., 100., 1000.)


def model(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def select(x, y, groups):
    """No held-out feature rows or labels enter scaler or hyperparameter selection."""
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    ng = len(set(groups.tolist()))
    if ng < 3:
        raise ValueError('Need at least three independent development groups')
    splits = list(GroupKFold(n_splits=3).split(x, y, groups))
    scores = []
    for alpha in ALPHAS:
        oof = np.empty(len(y))
        for tr, va in splits:
            assert not set(groups[tr]) & set(groups[va])
            fit = model(alpha).fit(x[tr], y[tr])
            oof[va] = np.maximum(0., fit.predict(x[va]))
        # Give each independent component equal weight.
        mae = np.mean([np.mean(abs(oof[groups == g]-y[groups == g])) for g in set(groups.tolist())])
        scores.append({'alpha': alpha, 'group_mae': float(mae), 'oof_predictions': oof.tolist()})
    chosen = min(scores, key=lambda r: (r['group_mae'], -r['alpha']))
    fit = model(chosen['alpha']).fit(x, y)
    xs = fit[0].transform(x)
    singular = np.linalg.svd(xs-xs.mean(0), compute_uv=False)
    info = {'selected_alpha': chosen['alpha'], 'cv_group_mae': chosen['group_mae'],
            'cv_scores': scores, 'train_r2': float(fit.score(x, y)),
            'effective_df': float(np.sum(singular**2/(singular**2+chosen['alpha']))),
            'n_features': x.shape[1], 'groups': ng,
            'train_prediction_range': [float(fit.predict(x).min()), float(fit.predict(x).max())]}
    return fit, info


def within_benchmark_permutation(benches, seed=42):
    rng = np.random.RandomState(seed)
    benches = np.asarray(benches)
    order = np.arange(len(benches))
    for b in sorted(set(benches.tolist())):
        ids = np.flatnonzero(benches == b)
        order[ids] = rng.permutation(ids)
    return order


def fit_arms(compact, rich, extra, y, groups, benches):
    fits = {name: select(x, y, groups) for name, x in [('compact', compact), ('rich', rich)]}
    rep = min(fits, key=lambda k: (fits[k][1]['cv_group_mae'], k != 'compact'))
    x = compact if rep == 'compact' else rich
    perm = within_benchmark_permutation(benches)
    scalar = fits[rep]
    mscs = select(np.c_[x, extra], y, groups)
    shuffled = select(np.c_[x, extra[perm]], y, groups)
    return {'representation': rep, 'models': {'scalar': scalar[0], 'mscs': mscs[0], 'shuffled': shuffled[0]},
            'info': {'scalar': scalar[1], 'mscs': mscs[1], 'shuffled': shuffled[1]},
            'scalar_representation_comparison': {k: v[1] for k,v in fits.items()},
            'train_permutation': perm.tolist(), 'constant_mean': float(np.mean(y))}


def infer(arms, compact, rich, extra, benches, cap):
    x = compact if arms['representation'] == 'compact' else rich
    perm = within_benchmark_permutation(benches)
    inputs = {'scalar': x, 'mscs': np.c_[x,extra], 'shuffled': np.c_[x,extra[perm]]}
    result = {k: np.clip(arms['models'][k].predict(v), 0., cap).tolist() for k,v in inputs.items()}
    result['constant_mean'] = np.minimum(arms['constant_mean'], cap).tolist()
    return result, perm.tolist()


def controls():
    """Known learnable signal with tiny input scale; null signal uses scalar only."""
    rng = np.random.RandomState(42)
    n, nt = 36, 12
    groups = np.repeat(np.arange(12), 3)
    z = rng.uniform(-1, 1, (n+nt,4))
    extra = .25+1e-4*z
    x = np.ones((n+nt,12))
    y = .006+.004*z[:,0]
    arms = fit_arms(x[:n], x[:n], extra[:n], y[:n], groups, ['synthetic']*n)
    pred,_ = infer(arms,x[n:],x[n:],extra[n:],['synthetic']*nt,np.full(nt,.1))
    mae = {k: float(np.mean(abs(np.array(v)-y[n:]))) for k,v in pred.items()}
    positive = mae['mscs'] < .1*min(mae['scalar'],mae['shuffled'])
    x[:,0] = rng.uniform(-1,1,n+nt)
    yn = .006+.004*x[:,0]
    armsn = fit_arms(x[:n],x[:n],extra[:n],yn[:n],groups,['synthetic']*n)
    predn,_ = infer(armsn,x[n:],x[n:],extra[n:],['synthetic']*nt,np.full(nt,.1))
    effect = float(np.max(abs(np.array(predn['mscs'])-predn['scalar'])))
    return {'positive_pass': bool(positive), 'positive_mae': mae,
            'negative_pass': bool(effect < 1e-4), 'negative_max_prediction_change': effect,
            'positive_mscs_fit': arms['info']['mscs'],
            'note': 'Synthetic calibration check, not a physical MSCS validation'}
