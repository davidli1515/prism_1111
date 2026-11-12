#!/usr/bin/env python3
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import traceback
import numpy as np
import torch
from core import (HERE, ROOT, ARMS, PARAMS, STEPS, TOPK, DEFAULT_CKPT, NORM_CACHE,
                  config_key, compute_w_min, legal, moves, LocalModel, Inputs,
                  FeatureCache, load_model, exact, cpis, contributions, probabilities)
from common import pareto_mask

sys.path.insert(0, str(ROOT/'mscs_response_1008'))
sys.path.insert(0, str(ROOT/'mscs_guard_v5'))
from area_store import AreaStore

OUT = HERE/'results_s42_01'
START = time.perf_counter()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def save(name, obj):
    path = OUT/name; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False,
                  default=lambda v: v.tolist() if hasattr(v, 'tolist') else v.item())


def status(phase, **kw):
    value = dict(phase=phase, elapsed_seconds=time.perf_counter()-START, **kw)
    path = OUT/'status.tmp'; path.write_text(json.dumps(value, indent=2)); path.replace(OUT/'status.json')
    print(json.dumps(value), flush=True)


class Areas:
    def __init__(self):
        self.store = AreaStore(OUT/'area_store')
        self.old = []
        for p in (ROOT/'mscs_guard_v5/area_store_v1/areas.sqlite',
                  ROOT/'mscs_branch_1009/results_s42_02/area_store/areas.sqlite'):
            if p.exists():
                self.old.append(sqlite3.connect(f'file:{p}?mode=ro', uri=True))
        self.counts = Counter()

    def get(self, c):
        self.counts['logical_requests'] += 1
        k = self.store.key(c)
        value = self.store.lookup(k)
        if value is not None:
            self.counts['new_store_hits'] += 1
            return value
        for db in self.old:
            row = db.execute('SELECT value FROM areas WHERE key=?', (k,)).fetchone()
            if row:
                self.counts['old_safe_store_hits'] += 1
                return self.store._record((k, row[0], None))
        self.counts['physical_mcpat_calls'] += 1
        return self.store(c)


def prepare():
    contexts, sources = [], {}
    checkpoint = sha(DEFAULT_CKPT)
    for bench in ('602.gcc_s', '605.mcf_s'):
        path = ROOT/'mscs_guard_v5/results_full_01'/bench/'s42/initial.json'
        d = load(path); sources[str(path)] = sha(path)
        assert d['checkpoint_sha256'] == checkpoint
        pp = np.c_[d['cpis'], d['areas']]
        fi = np.flatnonzero(pareto_mask(pp))
        fi = fi[np.argsort(pp[fi, 1], kind='stable')]
        assert len(fi) >= 2
        for label, rank in (('low', 0), ('middle', len(fi)//2)):
            i = int(fi[rank]); c = d['configs'][i]
            assert legal(c)
            contexts.append({'id': f'{bench}_{label}', 'bench': bench, 'index': i,
                             'root': c, 'root_cpi': d['cpis'][i], 'root_area': d['areas'][i],
                             'area_cap': 1.10*d['areas'][i], 'initial_candidates': moves(c),
                             'area_sources': d['area_sources']})
    assert len({(r['bench'], config_key(r['root'])) for r in contexts}) == 4
    paths = list(HERE.glob('*.py'))+[HERE/'PROTOCOL.txt', Path(DEFAULT_CKPT)]
    paths += [ROOT/'common.py', ROOT/'search.py', ROOT.parent/'dse_exp/dse_infer.py',
              ROOT.parent/'dse_exp/dse_engine.py', ROOT.parent/'dse_exp/config_dse.py',
              ROOT.parent/'exp1/model.py', ROOT.parent/'exp1/dataset.py',
              ROOT/'mscs_response_1008/safe_area.py', ROOT/'mscs_guard_v5/area_store.py']
    paths += [Path(NORM_CACHE)/f'18k_s42_{k}.npy' for k in ('mean', 'std')]
    sources.update({str(p): sha(p) for p in paths})
    plan = {'seed': 42, 'contexts': contexts, 'arms': ARMS, 'steps': STEPS, 'topk': TOPK,
            'source_sha256': sources, 'max_candidate_verifications': 144,
            'max_initial_audit_configurations': 360, 'scope': 'four-start constrained greedy pilot'}
    save('plan.json', plan)
    return plan


def array_save(name, **arrays):
    path = OUT/name; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **{k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                                 for k, v in arrays.items() if v is not None})


def check_root_derivatives(model, local, delta):
    """Independent directional derivative of all four CPIs, at the same input."""
    def fn(x):
        return cpis(probabilities(model, x)[1], local.width)
    _, jv = torch.autograd.functional.jvp(fn, local.x, delta)
    predicted = torch.einsum('nsd,nd->ns', local.gc, delta)
    torch.testing.assert_close(predicted, jv, rtol=2e-4, atol=2e-6)
    return {'passed': True, 'max_absolute_error': float((predicted-jv).abs().max()),
            'scope': 'independent network JVP in full feature space; not hardware validation'}


def walk(context, arm, model, inputs, area, record_truth):
    prefix = Path('walks')/context['id']/arm
    cache = FeatureCache(inputs)
    cur = dict(context['root']); width = compute_w_min(cur)
    cap, radius = context['area_cap'], 4
    seen = {config_key(cur)}
    current_cpi = context['root_cpi']; current_area = context['root_area']
    initial_cpi = current_cpi
    rounds, counts = [], Counter()
    timing = {k: 0. for k in ('root_forward', 'jacobian', 'ranking', 'candidate_forward', 'area')}
    local, local_key = None, None
    start = time.perf_counter()
    initial = None
    for step in range(STEPS):
        if local_key != config_key(cur):
            local = LocalModel(model, cache.get(cur), width, arm); local_key = config_key(cur)
            assert np.isclose(local.cpi, current_cpi, rtol=3e-6)
            counts['root_forward_calls'] += 1; counts['backward_scalar_outputs'] += local.backward_outputs
            timing['root_forward'] += local.forward_seconds; timing['jacobian'] += local.backward_seconds
            if step == 0:
                record_truth(cur, local.cpi, local.m)
        cand = moves(cur, radius=radius, seen=seen)
        if not cand:
            break
        xs = torch.stack([cache.get(c['config']) for c in cand])
        t = time.perf_counter(); prediction = local.predict(xs)
        scores = prediction['cpi'].cpu().numpy()
        order = np.argsort(scores, kind='stable')[:TOPK].tolist()
        timing['ranking'] += time.perf_counter()-t
        decision = {'step': step, 'root': cur, 'root_cpi': current_cpi, 'root_area': current_area,
                    'radius': radius, 'candidates': cand, 'predicted_cpis': scores.tolist(),
                    'predicted_cpi_slopes': prediction['cpi_slope'].tolist(), 'chosen': order,
                    'phase': 'before_selected_candidate_network_and_area_evaluations'}
        decision_name = prefix/f'decision_{step:02}.json'
        save(decision_name, decision); frozen = sha(OUT/decision_name)
        if step == 0:
            initial = dict(decision)
            array_save(prefix/'initial_predictions.npz', root_m=local.m, root_cs=local.cs,
                       root_ell=local.ell, stage_slope=prediction['stage_slope'],
                       logodds_slope=prediction['logodds_slope'],
                       partial_arch_slope=local.partial_arch_slope(xs),
                       predicted_m=prediction['mscs'])
            if arm == 'mscs_logit':
                save(prefix/'derivative_check.json', check_root_derivatives(model, local, xs[0]-local.x))
        trials = []
        for j in order:
            action = cand[j]; c = action['config']; k = config_key(c)
            assert k not in seen
            t = time.perf_counter(); value, mm = exact(model, xs[j], width)
            timing['candidate_forward'] += time.perf_counter()-t
            counts['candidate_forward_calls'] += 1
            t = time.perf_counter(); aa = area.get(c); timing['area'] += time.perf_counter()-t
            counts['area_requests'] += 1; seen.add(k)
            record_truth(c, value, mm)
            trial = dict(action, predicted_cpi=float(scores[j]), cpi=value, area=aa,
                         feasible=aa is not None and aa <= cap+1e-9,
                         query=int(counts['candidate_forward_calls']),
                         predicted_gain=(current_cpi-float(scores[j]))/current_cpi,
                         actual_gain=(current_cpi-value)/current_cpi)
            if arm == 'mscs_logit':
                mp = prediction['mscs'][j]
                pc0 = contributions(local.m, width)
                trial['explanation'] = {
                    'root_pressure_cpi': pc0.tolist(),
                    'predicted_pressure_cpi_change': (contributions(mp, width)-pc0).tolist(),
                    'actual_pressure_cpi_change': (contributions(mm, width)-pc0).tolist(),
                    'predicted_share_change': (mp.mean(0)-local.m.mean(0)).tolist(),
                    'actual_share_change': (mm.mean(0)-local.m.mean(0)).tolist(),
                    'root_clamped_slices_stages': int((local.m[..., 0] <= .01).sum()),
                    'predicted_clamped_slices_stages': int((mp[..., 0] <= .01).sum()),
                    'actual_clamped_slices_stages': int((mm[..., 0] <= .01).sum()),
                    'validation': 'exact frozen surrogate; not gem5'}
            trials.append(trial)
        accepted = [i for i, v in enumerate(trials) if v['feasible'] and v['cpi'] < current_cpi*(1-1e-6)]
        best = min(accepted, key=lambda i: trials[i]['cpi']) if accepted else None
        for j, trial in enumerate(trials):
            trial['accepted'] = j == best
        rec = dict(step=step, root=cur, root_cpi=current_cpi, root_area=current_area,
                   radius=radius, trials=trials, accepted_trial=best, decision_file=str(decision_name),
                   decision_sha256=frozen)
        if best is not None:
            cur = dict(trials[best]['config']); current_cpi = trials[best]['cpi']; current_area = trials[best]['area']
            radius = 4
        else:
            radius = max(1, radius//2)
        rec.update(next_root=cur, next_cpi=current_cpi, next_area=current_area)
        assert sha(OUT/decision_name) == frozen
        save(prefix/f'round_{step:02}.json', rec); rounds.append(rec)
        status('greedy', context=context['id'], arm=arm, round=step+1,
               verified=counts['candidate_forward_calls'], gain_pct=100*(1-current_cpi/initial_cpi))
        del xs, prediction
    assert counts['candidate_forward_calls'] <= STEPS*TOPK
    timing['feature_build'] = cache.seconds
    result = {'context': context['id'], 'arm': arm, 'initial_cpi': initial_cpi,
              'final_cpi': current_cpi, 'final_area': current_area, 'final_config': cur, 'area_cap': cap,
              'gain': 1-current_cpi/initial_cpi, 'rounds': len(rounds), 'counts': dict(counts),
              'feature_config_builds': cache.builds, 'timing_seconds': timing,
              'compute_seconds_excluding_area': sum(v for k, v in timing.items() if k != 'area'),
              'wall_seconds': time.perf_counter()-start, 'seen_configurations': len(seen),
              'accepted_steps': sum(r['accepted_trial'] is not None for r in rounds),
              'initial_prediction_file': str(prefix/'initial_predictions.npz')}
    save(prefix/'summary.json', result)
    return result, initial


def audit_initial(context, initial, model, inputs, registry):
    """Oracle labels after all arms finish; never passed back into greedy."""
    cs = context['initial_candidates']; width = compute_w_min(context['root'])
    actual, pressure, counts = [], [], Counter()
    for action in cs:
        c = action['config']; k = config_key(c)
        if k in registry:
            cp, mm = registry[k]; counts['reused_search_evaluations'] += 1
        else:
            cp, mm = exact(model, inputs.build(c), width); counts['extra_forward_calls'] += 1
        actual.append(cp); pressure.append(contributions(mm, width).tolist())
    metrics = {}
    truth = np.array(actual); c0 = context['root_cpi']
    gstar = max(0., 1-float(truth.min())/c0)
    for arm in ARMS:
        row = initial[arm]
        assert row['candidates'] == cs
        pred = np.array(row['predicted_cpis']); top = np.argsort(pred, kind='stable')[:2]
        captured = max(0., 1-float(truth[top].min())/c0)
        metrics[arm] = {'relative_cpi_MAE': float(np.mean(abs(pred-truth))/c0),
                        'top2_gain': captured, 'oracle_gain': gstar,
                        'top2_regret': gstar-captured,
                        'top2_gain_capture': captured/gstar if gstar > 1e-8 else None,
                        'sign_agreement': float(np.mean(np.sign(c0-pred) == np.sign(c0-truth)))}
    arrays = np.load(OUT/Path('walks')/context['id']/'mscs_logit/initial_predictions.npz')
    dc = arrays['stage_slope'].mean((1, 2)); partial = arrays['partial_arch_slope']
    predicted_m = torch.from_numpy(arrays['predicted_m'])
    predicted_pressure = contributions(predicted_m, width).numpy()
    response_mae = float(np.mean(abs(predicted_pressure-np.array(pressure))))
    result = {'context': context['id'], 'candidates': cs, 'actual_cpis': actual,
              'actual_pressure_cpi': pressure, 'predicted_pressure_cpi': predicted_pressure.tolist(),
              'metrics': metrics, 'counts': dict(counts),
              'component_CPI_MAE': response_mae,
              'full_vs_arch_slope_mean_abs_difference': float(np.mean(abs(dc-partial))),
              'scope': 'offline unconstrained initial-pool diagnostic; all-candidate areas not evaluated'}
    save(Path('audits')/f"{context['id']}.json", result)
    return result


def main():
    OUT.mkdir(exist_ok=False); status('preparing')
    plan = prepare(); area = Areas(); model = load_model()
    for ctx in plan['contexts']:
        assert area.store.sources == ctx['area_sources']
    all_walks, audits = [], []
    for ri, ctx in enumerate(plan['contexts']):
        status('loading_inputs', context=ctx['id'])
        inputs = Inputs(ctx['bench'], ctx['root']['l1_size'])
        save(Path('inputs')/f"{ctx['id']}.json", {'slice_ids': inputs.ids, 'load_seconds': inputs.load_seconds})
        root_area = area.get(ctx['root'])
        assert root_area is not None and abs(root_area-ctx['root_area']) < 1e-6
        registry, initial = {}, {}
        def record(c, cp, m):
            k = config_key(c)
            if k in registry:
                assert np.isclose(cp, registry[k][0], rtol=3e-6)
            registry[k] = (float(cp), m.detach().clone())
        order = list(ARMS[ri % 3:])+list(ARMS[:ri % 3])
        for arm in order:
            result, init = walk(ctx, arm, model, inputs, area, record)
            all_walks.append(result); initial[arm] = init
        status('offline_initial_audit', context=ctx['id'])
        audits.append(audit_initial(ctx, initial, model, inputs, registry))
        del inputs, registry
    paired = []
    for ctx in plan['contexts']:
        rr = {r['arm']: r for r in all_walks if r['context'] == ctx['id']}
        m = rr['mscs_logit']['final_cpi']
        paired.append({'context': ctx['id'],
                       'final_cpi': {a: r['final_cpi'] for a, r in rr.items()},
                       'gain_pct': {a: 100*r['gain'] for a, r in rr.items()},
                       'mscs_vs_scalar_excess_gain_pp': 100*(rr['scalar_excess']['final_cpi']-m)/ctx['root_cpi'],
                       'mscs_vs_best_scalar_gain_pp': 100*(min(rr[a]['final_cpi'] for a in ARMS[:2])-m)/ctx['root_cpi']})
    assert all(sha(p) == h for p, h in plan['source_sha256'].items())
    summary = {'paired': paired, 'walks': all_walks,
               'audit_metrics': [{k: v for k, v in a.items() if k in ('context', 'metrics', 'counts', 'component_CPI_MAE', 'full_vs_arch_slope_mean_abs_difference')} for a in audits],
               'area_counts': dict(area.counts), 'area_failures': area.store.n_fail,
               'wall_seconds': time.perf_counter()-START, 'source_preservation_passed': True,
               'limitations': 'four roots/one seed, frozen surrogate only; area cache order prevents end-to-end speedup claims'}
    save('summary.json', summary)
    from validate import validate
    save('validation.json', validate(OUT))
    from report import report
    report(OUT)
    status('complete', wall_seconds=time.perf_counter()-START, area_failures=area.store.n_fail)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT.exists():
            (OUT/'ERROR.txt').write_text(traceback.format_exc())
            status('failed', traceback_file=str(OUT/'ERROR.txt'))
        raise
