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
from mechanism import (HERE, ROOT, PREVIOUS, MODES, TAU, candidates, context_rows, select, witnesses,
                       LocalModel, Inputs, load_model, exact, contributions, config_key,
                       compute_w_min, DEFAULT_CKPT, NORM_CACHE)
from common import pareto_mask

sys.path.insert(0, str(ROOT/'mscs_response_1008'))
sys.path.insert(0, str(ROOT/'mscs_guard_v5'))
from area_store import AreaStore

OUT = HERE/'results_s42_01'
START = time.perf_counter()


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, obj):
    p = OUT/name; p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False,
                  default=lambda v: v.tolist() if hasattr(v, 'tolist') else v.item())


def status(phase, **kwargs):
    d = dict(phase=phase, elapsed_seconds=time.perf_counter()-START, **kwargs)
    p = OUT/'status.tmp'; p.write_text(json.dumps(d, indent=2)); p.replace(OUT/'status.json')
    print(json.dumps(d), flush=True)


class Areas:
    def __init__(self):
        self.store = AreaStore(OUT/'area_store'); self.counts = Counter()
        self.old = []
        for p in (ROOT/'mscs_guard_v5/area_store_v1/areas.sqlite',
                  ROOT/'mscs_branch_1009/results_s42_02/area_store/areas.sqlite',
                  PREVIOUS/'results_s42_01/area_store/areas.sqlite'):
            if p.exists():
                self.old.append(sqlite3.connect(f'file:{p}?mode=ro', uri=True))

    def get(self, c):
        self.counts['logical_requests'] += 1
        key = self.store.key(c); value = self.store.lookup(key)
        if value is not None:
            self.counts['new_store_hits'] += 1
            return value
        for db in self.old:
            row = db.execute('SELECT value FROM areas WHERE key=?', (key,)).fetchone()
            if row:
                self.counts['old_readonly_hits'] += 1
                return self.store._record((key, row[0], None))
        self.counts['physical_mcpat_calls'] += 1
        return self.store(c)


def prepare():
    prior = PREVIOUS/'results_s42_01/plan.json'; old = read(prior)
    used = {(c['bench'], config_key(c['root'])) for c in old['contexts']}
    paths = [prior]; contexts = []; ckpt = sha(DEFAULT_CKPT)
    for bench in ('602.gcc_s', '605.mcf_s', '623.xalancbmk_s', '641.leela_s'):
        p = ROOT/'mscs_guard_v5/results_full_01'/bench/'s42/initial.json'
        paths.append(p); d = read(p); assert d['checkpoint_sha256'] == ckpt
        points = np.c_[d['cpis'], d['areas']]
        front = np.flatnonzero(pareto_mask(points)); front = front[np.argsort(points[front, 1], kind='stable')]
        target = len(front)//4 if bench in ('602.gcc_s', '605.mcf_s') else len(front)//2
        order = sorted(range(len(front)), key=lambda j: (abs(j-target), j))
        rank = next(j for j in order if (bench, config_key(d['configs'][int(front[j])])) not in used)
        i = int(front[rank]); root = d['configs'][i]
        ctx = {'id': f'{bench}_rank{rank}', 'bench': bench, 'rank': rank, 'archive_index': i,
               'root': root, 'root_cpi': d['cpis'][i], 'root_area': d['areas'][i],
               'area_cap': 1.10*d['areas'][i], 'area_sources': d['area_sources']}
        contexts.append(ctx)
    paths += list(HERE.glob('*.py'))+[HERE/'PROTOCOL.txt', Path(DEFAULT_CKPT)]
    paths += list(PREVIOUS.glob('*.py'))+[PREVIOUS/'PROTOCOL.txt']
    paths += [ROOT/'common.py', ROOT.parent/'dse_exp/config_dse.py', ROOT.parent/'dse_exp/dse_infer.py',
              ROOT.parent/'dse_exp/dse_engine.py', ROOT.parent/'exp1/model.py', ROOT.parent/'exp1/dataset.py',
              ROOT/'mscs_response_1008/safe_area.py', ROOT/'mscs_guard_v5/area_store.py']
    paths += [Path(NORM_CACHE)/f'18k_s42_{kind}.npy' for kind in ('mean', 'std')]
    plan = {'contexts': contexts, 'methods': MODES, 'seed': 42, 'tau': TAU,
            'logical_calls_per_method_root': 2, 'max_logical_search_calls': 40,
            'max_extra_audit_calls': 24, 'source_sha256': {str(p): sha(p) for p in paths},
            'scope': 'one greedy decision at four different roots, plus bounded conditional audits'}
    save('plan.json', plan)
    return plan


def predictions(ctx, model, inputs):
    pool = candidates(ctx['root']); width = compute_w_min(ctx['root']); t0 = time.perf_counter()
    x0 = inputs.build(ctx['root'])
    feature_seconds = time.perf_counter()-t0
    ms = LocalModel(model, x0, width, 'mscs_logit')
    sc = LocalModel(model, x0, width, 'scalar_excess')
    torch.testing.assert_close(ms.gc, sc.gc, rtol=3e-4, atol=2e-6)
    assert np.isclose(ms.cpi, ctx['root_cpi'], rtol=3e-6)
    arrays = {k: [] for k in ('mscs_cpi', 'scalar_cpi', 'predicted_m', 'stage_slope', 'logodds_slope')}
    ranking_seconds = 0.
    for start in range(0, len(pool), 16):
        t0 = time.perf_counter()
        xs = torch.stack([inputs.build(c['config']) for c in pool[start:start+16]])
        feature_seconds += time.perf_counter()-t0
        t0 = time.perf_counter(); pm = ms.predict(xs); ps = sc.predict(xs)
        ranking_seconds += time.perf_counter()-t0
        torch.testing.assert_close(pm['stage_slope'], ps['stage_slope'], rtol=3e-4, atol=3e-6)
        for key, value in (('mscs_cpi', pm['cpi']), ('scalar_cpi', ps['cpi']),
                           ('predicted_m', pm['mscs']), ('stage_slope', pm['stage_slope']),
                           ('logodds_slope', pm['logodds_slope'])):
            arrays[key].append(value.numpy())
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    arrays.update(root_m=ms.m.numpy(), root_cs=ms.cs.numpy(), root_ell=ms.ell.numpy())
    base = Path('roots')/ctx['id']; (OUT/base).mkdir(parents=True)
    with (OUT/base/'predictions.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    save(base/'pool.json', pool)
    selections = {mode: select(pool, arrays['scalar_cpi'], arrays['mscs_cpi'], ms.cpi, mode) for mode in MODES}
    audit = witnesses(pool, arrays['scalar_cpi'], arrays['mscs_cpi'], ms.cpi, selections['mscs'])
    decision = {'context': ctx['id'], 'root_cpi': ms.cpi, 'selections': selections, 'witnesses': audit,
                'phase': 'frozen before any candidate exact CPI or area labels', 'slice_ids': inputs.ids,
                'timing_seconds': {'features': feature_seconds, 'ranking': ranking_seconds,
                                   'root_forward_scalar': sc.forward_seconds, 'root_forward_mscs': ms.forward_seconds,
                                   'backward_scalar': sc.backward_seconds, 'backward_mscs': ms.backward_seconds},
                'feature_configurations': len(pool)+1, 'root_tangents_match': True,
                'max_absolute_gradient_difference': float((ms.gc-sc.gc).abs().max())}
    save(base/'decision.json', decision)
    return pool, decision, arrays, sha(OUT/base/'decision.json')


def evaluate_root(ctx, model, inputs, areas, pool, decision, arrays, decision_hash):
    base = Path('roots')/ctx['id']; width = compute_w_min(ctx['root']); c0 = decision['root_cpi']
    labels, counts = {}, Counter()
    timing = dict(feature=0., candidate_forward=0., area=0.)
    def query(i, phase):
        if i in labels:
            counts[f'{phase}_physical_cache_hits'] += 1
            return labels[i]
        c = pool[i]['config']; t0 = time.perf_counter(); x = inputs.build(c)
        timing['feature'] += time.perf_counter()-t0
        t0 = time.perf_counter(); cp, mm = exact(model, x, width)
        timing['candidate_forward'] += time.perf_counter()-t0
        t0 = time.perf_counter(); area = areas.get(c); timing['area'] += time.perf_counter()-t0
        counts[f'{phase}_physical_forward_calls'] += 1
        label = {'candidate': i, 'config': c, 'cpi': cp, 'area': area,
                 'feasible': area is not None and area <= ctx['area_cap']+1e-9,
                 'gain': 1-cp/c0, 'pressure_cpi': contributions(mm, width).tolist(),
                 'mean_stack': mm.mean(0).tolist(), 'phase_first_evaluated': phase,
                 'clamped_slice_stages': int((mm[..., 0] <= .01).sum())}
        labels[i] = label; save(base/'labels'/f'{i:04}.json', label)
        return label
    outcomes = {}
    for mode in MODES:
        choices = decision['selections'][mode]['chosen']
        trial = [query(i, 'search') for i in choices]
        ok = [t for t in trial if t['feasible'] and t['cpi'] < c0*(1-1e-6)]
        best = min(ok, key=lambda t: t['cpi']) if ok else None
        outcomes[mode] = {'chosen': choices, 'logical_candidate_calls': 2,
                          'accepted_candidate': best['candidate'] if best else None,
                          'final_cpi': best['cpi'] if best else c0,
                          'final_area': best['area'] if best else ctx['root_area'],
                          'gain_pct': 100*best['gain'] if best else 0.}
    save(base/'search_outcomes.json', outcomes)
    status('greedy_done', context=ctx['id'], gain_pct={m:r['gain_pct'] for m,r in outcomes.items()})
    witness_results = {}
    root_pressure = (arrays['root_cs'][..., None]*arrays['root_m'][..., 1:]).mean(0)
    for kind, k in decision['witnesses'].items():
        u, v = pool[k]['parts']; joint, aa, bb = query(k, 'audit'), query(u, 'audit'), query(v, 'audit')
        interaction = (joint['cpi']-aa['cpi']-bb['cpi']+c0)/c0
        alone = (bb['cpi']-c0)/c0; after = (joint['cpi']-aa['cpi'])/c0
        if kind == 'decouple':
            verified = interaction < -1e-8 and joint['gain'] > 0
        else:
            verified = 0 <= alone < TAU and after >= TAU and aa['gain'] > 0
        saving_alone = bb['area'] is not None and bb['area'] < ctx['root_area']
        saving_after = joint['area'] is not None and aa['area'] is not None and joint['area'] < aa['area']
        pp = contributions(torch.from_numpy(arrays['predicted_m'][[k, u, v]]), width).numpy()
        predicted_interaction_components = pp[0]-pp[1]-pp[2]+root_pressure
        actual_interaction_components = (np.array(joint['pressure_cpi'])-aa['pressure_cpi']
                                         -np.array(bb['pressure_cpi'])+root_pressure)
        predictions_by_method = {m: context_rows(pool, arrays[a], c0)[k] for m,a in
                                 (('scalar', 'scalar_cpi'), ('mscs', 'mscs_cpi'))}
        witness_results[kind] = {'candidate': k, 'parts': [u, v], 'predictions': predictions_by_method,
                                'actual_interaction': interaction, 'actual_alone_cost': alone,
                                'actual_after_investment_cost': after,
                                'actual_investment_gain': aa['gain'], 'actual_joint_gain': joint['gain'],
                                'mechanism_verified': bool(verified),
                                'area_saved_alone': bool(saving_alone), 'area_saved_after': bool(saving_after),
                                'investment_feasible': aa['feasible'], 'joint_feasible': joint['feasible'],
                                'predicted_interaction_components': predicted_interaction_components.tolist(),
                                'actual_interaction_components': actual_interaction_components.tolist(),
                                'scope': 'bounded surrogate counterfactual audit; no label feedback into search'}
    assert decision_hash == sha(OUT/base/'decision.json')
    save(base/'witness_results.json', witness_results)
    result = {'context': ctx['id'], 'candidate_count': len(pool), 'outcomes': outcomes,
              'witness_results': witness_results, 'counts': dict(counts),
              'prediction_timing_seconds': decision['timing_seconds'], 'verification_timing_seconds': timing,
              'decision_sha256': decision_hash,
              'mscs_vs_scalar_gain_pp': outcomes['mscs']['gain_pct']-outcomes['scalar']['gain_pct']}
    save(base/'summary.json', result)
    return result


def main():
    OUT.mkdir(exist_ok=False); status('prepare'); plan = prepare(); areas = Areas(); model = load_model()
    results = []
    for ctx in plan['contexts']:
        assert ctx['area_sources'] == areas.store.sources
        root_area = areas.get(ctx['root']); assert abs(root_area-ctx['root_area']) < 1e-6
        status('predicting', context=ctx['id'])
        inputs = Inputs(ctx['bench'], ctx['root']['l1_size'])
        pool, decision, arrays, digest = predictions(ctx, model, inputs)
        status('verifying', context=ctx['id'], pool_size=len(pool), witnesses=decision['witnesses'])
        results.append(evaluate_root(ctx, model, inputs, areas, pool, decision, arrays, digest))
    assert all(sha(p) == digest for p,digest in plan['source_sha256'].items())
    summary = {'results': results, 'area_counts': dict(areas.counts), 'area_failures': areas.store.n_fail,
               'source_preservation_passed': True, 'elapsed_seconds': time.perf_counter()-START,
               'scope': 'one greedy decision per method/root, not a full search or gem5 validation'}
    save('summary.json', summary)
    from context_replay import replay
    save('validation.json', replay(OUT))
    from context_report import report
    report(OUT)
    status('complete', area_failures=areas.store.n_fail)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT.exists():
            (OUT/'ERROR.txt').write_text(traceback.format_exc()); status('failed')
        raise
