"""Replay saved decisions without model inference or importing the search driver."""
import hashlib
import json
from pathlib import Path
import numpy as np
from hybrid_support import ARMS, PARAMS, PREVIOUS, legal, moves, config_key, compute_w_min


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


def prediction_replay(out, d, arm, width):
    p = out/d['prediction_file']
    assert sha(p) == d['prediction_sha256']
    arr = np.load(p)
    cs, slope = arr['root_cs'], arr['stage_slope']
    close(cs, stage_cpi(arr['root_m'], width))
    lo, hi = 1/width, min(100., 1/(.01*width+1e-8))
    excess = np.maximum(cs-lo, 1e-10)
    scalar = np.clip(lo+excess[None]*np.exp(np.clip(slope/excess[None], -80,80)),lo,hi).mean((1,2))
    close(scalar, d['scalar_predicted_cpis'])
    if arm == 'scalar_excess':
        close(scalar, d['predicted_cpis'])
        assert d['mscs_predicted_cpis'] is None
    else:
        ell = arr['root_ell'][None]+arr['logodds_slope']
        m = softmax(np.concatenate([np.zeros_like(ell[..., :1]),ell],-1))
        close(m, arr['predicted_m'])
        mscs = stage_cpi(m,width).mean((1,2))
        close(mscs, d['mscs_predicted_cpis']); close(mscs, d['predicted_cpis'])
    close(slope.mean((1,2)),d['predicted_cpi_slopes'])
    close(cs.mean(), d['root_cpi'], rtol=3e-6)
    return arr


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
                arr = prediction_replay(out, d, arm, width)
                scalar_order = np.argsort(d['scalar_predicted_cpis'],kind='stable').tolist()
                if arm == 'scalar_excess':
                    chosen = scalar_order[:2]; sources = ['scalar']*len(chosen)
                else:
                    m_order = np.argsort(d['mscs_predicted_cpis'],kind='stable').tolist()
                    if arm == 'mscs_logit':
                        chosen = m_order[:2]; sources = ['mscs']*len(chosen)
                    else:
                        chosen = [scalar_order[0]]+[i for i in m_order if i != scalar_order[0]][:1]
                        sources = ['scalar_anchor']+['mscs_alternative']*(len(chosen)-1)
                assert sources == d['proposal_sources']
                if step == 0:
                    initial_arrays[arm] = arr
                    assert d['candidates'] == ctx['initial_candidates']
                    if arm != 'scalar_excess':
                        assert read(base/'derivative_check.json')['passed']
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
                    close(t['scalar_predicted_cpi'], d['scalar_predicted_cpis'][j])
                    assert t['proposed_by'] == sources[i]
                    if arm != 'scalar_excess':
                        close(t['mscs_predicted_cpi'], d['mscs_predicted_cpis'][j])
                    close(t['actual_gain'], 1-t['cpi']/cp)
                    close(t['predicted_gain'], 1-t['predicted_cpi']/cp)
                    is_feasible = t['area'] is not None and t['area'] <= ctx['area_cap']+1e-9
                    assert t['feasible'] == is_feasible
                    if is_feasible and t['cpi'] < cp*(1-1e-6):
                        feasible.append(i)
                    if arm != 'scalar_excess':
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
            assert result['counts']['backward_scalar_outputs'] == len(roots)*(4 if arm == 'scalar_excess' else 12)
            assert accepted_count == result['accepted_steps']
            assert len(seen) == result['seen_configurations']
            assert current == result['final_config'] and area <= ctx['area_cap']+1e-9
            close(cp, result['final_cpi']); close(area, result['final_area'])
            close(1-cp/ctx['root_cpi'], result['gain'])
            close(sum(v for k, v in result['timing_seconds'].items() if k != 'area'),
                  result['compute_seconds_excluding_area'])
            total_queries += queries
            if ctx['group'] == 'regression' and arm != 'hybrid':
                old = read(PREVIOUS/'results_s42_01/walks'/ctx['id']/arm/'summary.json')
                close(result['final_cpi'],old['final_cpi'],rtol=3e-6)
                assert result['final_config'] == old['final_config']
            report['walks'].append({'context': ctx['id'], 'arm': arm, 'queries': queries,
                                    'accepted': accepted_count, 'passed': True})
        a0 = initial_arrays[ARMS[0]]
        for arm in ARMS[1:]:
            close(initial_arrays[arm]['root_m'], a0['root_m'])
            close(initial_arrays[arm]['stage_slope'], a0['stage_slope'], rtol=2e-4, atol=3e-6)
        pair = next(r for r in summary['paired'] if r['context'] == ctx['id'])
        for arm in ARMS:
            close(pair['final_cpi'][arm], root_results[arm]['final_cpi'])
            close(pair['gain_pct'][arm], 100*root_results[arm]['gain'])
        scalar = pair['final_cpi']['scalar_excess']; mscs = pair['final_cpi']['mscs_logit']; hybrid = pair['final_cpi']['hybrid']
        close(pair['mscs_vs_scalar_gain_pp'], 100*(scalar-mscs)/ctx['root_cpi'])
        close(pair['hybrid_vs_scalar_gain_pp'], 100*(scalar-hybrid)/ctx['root_cpi'])
        close(pair['hybrid_vs_mscs_gain_pp'], 100*(mscs-hybrid)/ctx['root_cpi'])
        report['root_checks'].append({'context':ctx['id'],'initial_tangents_agree':True,
                                      'all_round_predictions_replayed':True, 'hybrid_selection_replayed':True})
    assert total_queries <= plan['max_candidate_verifications']
    assert total_queries == sum(w['counts']['candidate_forward_calls'] for w in summary['walks'])
    report['total_candidate_queries'] = total_queries
    return report


if __name__ == '__main__':
    import sys
    print(json.dumps(validate(Path(sys.argv[1])), indent=2))
