"""Replay saved decisions without model inference or importing the search driver."""
import hashlib
import json
from pathlib import Path
import numpy as np
from core import ARMS, PARAMS, legal, moves, config_key, compute_w_min


def read(p):
    return json.loads(Path(p).read_text())


def close(a, b, rtol=5e-5, atol=2e-6):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def softmax(z):
    z = z-z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p/p.sum(axis=-1, keepdims=True)


def stage_cpi(p, width):
    return np.minimum(100., 1/(np.maximum(.01, p[..., 0])*width+1e-8))


def validate(out):
    out = Path(out)
    plan, summary = read(out/'plan.json'), read(out/'summary.json')
    for p, h in plan['source_sha256'].items():
        assert sha(p) == h, p
    report = {'passed': True, 'sources_unchanged': True, 'walks': [], 'root_checks': []}
    total_queries = 0
    for ctx in plan['contexts']:
        initial_arrays = {}
        root_results = {}
        width = compute_w_min(ctx['root'])
        for arm in ARMS:
            base = out/'walks'/ctx['id']/arm
            result = read(base/'summary.json'); root_results[arm] = result
            current, cp, area = ctx['root'], ctx['root_cpi'], ctx['root_area']
            seen, radius, queries, accepted_count, roots = {config_key(current)}, 4, 0, 0, set()
            rounds = sorted(base.glob('round_*.json'))
            assert len(rounds) == result['rounds'] <= plan['steps']
            for step, file in enumerate(rounds):
                r = read(file); dpath = out/r['decision_file']; d = read(dpath)
                assert sha(dpath) == r['decision_sha256']
                assert r['step'] == d['step'] == step
                assert current == r['root'] == d['root']
                assert r['radius'] == d['radius'] == radius
                close(cp, r['root_cpi']); close(cp, d['root_cpi'])
                close(area, r['root_area']); close(area, d['root_area'])
                assert d['candidates'] == moves(current, radius=radius, seen=seen)
                chosen = np.argsort(d['predicted_cpis'], kind='stable')[:plan['topk']].tolist()
                assert chosen == d['chosen'] and len(chosen) == len(r['trials'])
                roots.add(config_key(current))
                feasible = []
                for i, (j, t) in enumerate(zip(chosen, r['trials'])):
                    action = d['candidates'][j]; c = t['config']; key = config_key(c)
                    assert all(t[k] == v for k, v in action.items())
                    assert legal(c) and key not in seen
                    assert compute_w_min(c) == width
                    assert c['l1_size'] == ctx['root']['l1_size']
                    assert c['bp_type'] == ctx['root']['bp_type']
                    assert [k for k in current if c[k] != current[k]] == [t['parameter']]
                    assert t['parameter'] in PARAMS
                    seen.add(key); queries += 1
                    assert t['query'] == queries
                    close(t['predicted_cpi'], d['predicted_cpis'][j])
                    close(t['actual_gain'], 1-t['cpi']/cp)
                    close(t['predicted_gain'], 1-t['predicted_cpi']/cp)
                    is_feasible = t['area'] is not None and t['area'] <= ctx['area_cap']+1e-9
                    assert t['feasible'] == is_feasible
                    if is_feasible and t['cpi'] < cp*(1-1e-6):
                        feasible.append(i)
                    if arm == 'mscs_logit':
                        e = t['explanation']
                        for keyname in ('predicted_pressure_cpi_change', 'actual_pressure_cpi_change'):
                            assert np.array(e[keyname]).shape == (4, 3)
                        for keyname in ('predicted_share_change', 'actual_share_change'):
                            change = np.array(e[keyname]); assert change.shape == (4, 4)
                            close(change.sum(-1), 0., atol=3e-7)
                        if e['root_clamped_slices_stages'] == 0:
                            if e['actual_clamped_slices_stages'] == 0:
                                close(np.array(e['actual_pressure_cpi_change']).sum(-1).mean(),
                                      t['cpi']-cp, atol=5e-6)
                            if e['predicted_clamped_slices_stages'] == 0:
                                close(np.array(e['predicted_pressure_cpi_change']).sum(-1).mean(),
                                      t['predicted_cpi']-cp, atol=5e-6)
                best = min(feasible, key=lambda i: r['trials'][i]['cpi']) if feasible else None
                assert best == r['accepted_trial']
                assert [t['accepted'] for t in r['trials']] == [i == best for i in range(len(chosen))]
                if best is not None:
                    chosen_trial = r['trials'][best]
                    current, cp, area = chosen_trial['config'], chosen_trial['cpi'], chosen_trial['area']
                    accepted_count += 1; radius = 4
                else:
                    radius = max(1, radius//2)
                assert current == r['next_root']
                close(cp, r['next_cpi']); close(area, r['next_area'])
            assert queries == result['counts']['candidate_forward_calls'] <= plan['steps']*plan['topk']
            assert queries == result['counts']['area_requests']
            assert len(roots) == result['counts']['root_forward_calls']
            assert result['counts']['backward_scalar_outputs'] == len(roots)*(12 if arm == 'mscs_logit' else 4)
            assert accepted_count == result['accepted_steps']
            assert len(seen) == result['seen_configurations']
            assert current == result['final_config'] and area <= ctx['area_cap']+1e-9
            close(cp, result['final_cpi']); close(area, result['final_area'])
            close(1-cp/ctx['root_cpi'], result['gain'])
            close(sum(v for k, v in result['timing_seconds'].items() if k != 'area'),
                  result['compute_seconds_excluding_area'])
            total_queries += queries
            arr = np.load(base/'initial_predictions.npz'); initial_arrays[arm] = arr
            d = read(base/'decision_00.json')
            assert d['candidates'] == ctx['initial_candidates']
            cs, slope = arr['root_cs'], arr['stage_slope']
            close(cs, stage_cpi(arr['root_m'], width))
            lo, hi = 1/width, min(100., 1/(.01*width+1e-8))
            if arm == 'scalar_linear':
                pred = np.clip(cs[None]+slope, lo, hi)
            elif arm == 'scalar_excess':
                excess = np.maximum(cs-lo, 1e-10)
                pred = np.clip(lo+excess[None]*np.exp(np.clip(slope/excess[None], -80, 80)), lo, hi)
            else:
                ell = arr['root_ell'][None]+arr['logodds_slope']
                z = np.concatenate([np.zeros_like(ell[..., :1]), ell], -1)
                m = softmax(z)
                close(m, arr['predicted_m'])
                pred = stage_cpi(m, width)
                derivative_check = read(base/'derivative_check.json')
                assert derivative_check['passed']
            close(pred.mean((1, 2)), d['predicted_cpis'])
            close(slope.mean((1, 2)), d['predicted_cpi_slopes'])
            report['walks'].append({'context': ctx['id'], 'arm': arm, 'queries': queries,
                                    'accepted': accepted_count, 'passed': True})
        a0 = initial_arrays[ARMS[0]]
        for arm in ARMS[1:]:
            close(initial_arrays[arm]['root_m'], a0['root_m'])
            close(initial_arrays[arm]['stage_slope'], a0['stage_slope'], rtol=2e-4, atol=3e-6)
        audit = read(out/'audits'/f"{ctx['id']}.json")
        actual = np.array(audit['actual_cpis']); c0 = ctx['root_cpi']
        assert audit['candidates'] == ctx['initial_candidates']
        assert len(actual) == sum(audit['counts'].values())
        oracle = max(0., 1-actual.min()/c0)
        for arm in ARMS:
            d = read(out/'walks'/ctx['id']/arm/'decision_00.json')
            pred = np.array(d['predicted_cpis'])
            top = np.argsort(pred, kind='stable')[:plan['topk']]
            captured = max(0., 1-actual[top].min()/c0)
            metric = audit['metrics'][arm]
            close(np.mean(abs(pred-actual))/c0, metric['relative_cpi_MAE'])
            close(captured, metric['top2_gain']); close(oracle-captured, metric['top2_regret'])
            close(oracle, metric['oracle_gain'])
            if oracle > 1e-8:
                close(captured/oracle, metric['top2_gain_capture'])
        close(np.mean(abs(np.array(audit['actual_pressure_cpi'])-audit['predicted_pressure_cpi'])),
              audit['component_CPI_MAE'])
        pair = next(r for r in summary['paired'] if r['context'] == ctx['id'])
        for arm in ARMS:
            close(pair['final_cpi'][arm], root_results[arm]['final_cpi'])
            close(pair['gain_pct'][arm], 100*root_results[arm]['gain'])
        close(pair['mscs_vs_best_scalar_gain_pp'],
              100*(min(pair['final_cpi'][a] for a in ARMS[:2])-pair['final_cpi']['mscs_logit'])/c0)
        report['root_checks'].append({'context': ctx['id'], 'initial_tangents_agree': True,
                                      'independent_prediction_replay': True, 'audit_replay': True})
    assert total_queries <= plan['max_candidate_verifications']
    assert total_queries == sum(w['counts']['candidate_forward_calls'] for w in summary['walks'])
    report['total_candidate_queries'] = total_queries
    return report


if __name__ == '__main__':
    import sys
    print(json.dumps(validate(Path(sys.argv[1])), indent=2))
