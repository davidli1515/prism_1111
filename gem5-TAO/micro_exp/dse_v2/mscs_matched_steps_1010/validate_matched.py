"""Independent selection/feedback/acceptance replay; no policy implementation."""
from itertools import combinations
import copy
from matched_support import (read, sha, ARMS, KINDS, STRIDE, get_param_values,
                             compute_w_min, config_key, label_id)
from pathlib import Path
import numpy as np
from core import moves, legal
from adaptive_validate import prediction_replay


def same(a, b):
    np.testing.assert_allclose(a, b, rtol=5e-5, atol=2e-6)


def independent_pool(d, kind):
    if kind == 'full':
        return list(range(len(d['candidates'])))
    ids = []; stride = int(kind[-1])
    for i, a in enumerate(d['candidates']):
        values = get_param_values(a['parameter'])
        full_index = values.index(d['root'][a['parameter']])+a['sign']*stride*STRIDE.get(a['parameter'], 1)
        if a['stride'] == stride and 0 <= full_index < len(values) and a['config'][a['parameter']] == values[full_index]:
            ids.append(i)
    return ids


def independent_pairs(d, ids):
    s, m = d['scalar_predicted_cpis'], d['mscs_predicted_cpis']
    so = sorted(ids, key=lambda i: (s[i], i)); mo = sorted(ids, key=lambda i: (m[i], i))
    corrected = {}
    for i in ids:
        a = d['candidates'][i]; key = f"{a['parameter']}:{a['sign']:+d}"
        losses = d['allocator_before']['buckets'].get(key, {}).get('loss2', dict(scalar=0, mscs=0))
        corrected[i] = m[i] if losses['mscs'] < losses['scalar'] else s[i]
    lo = sorted(ids, key=lambda i: (corrected[i], i))
    return dict(scalar=so[:2], mscs=mo[:2],
                hybrid=[so[0], next(i for i in mo if i != so[0])],
                local=[lo[0], next(i for i in mo if i != lo[0])])


def advance_feedback(state, r):
    """Rebuild pre-round local credit using only earlier logged pairs."""
    groups = {}
    for trial in r['trials']:
        key = f"{trial['parameter']}:{trial['sign']:+d}"
        groups.setdefault(key, []).append(trial)
    cp = r['root_cpi']; tol = cp*1e-6
    for key, trials in groups.items():
        st = state['buckets'].setdefault(key, dict(loss2=dict(scalar=0, mscs=0),
            comparisons=0, discriminating_comparisons=0, observations=0))
        st['observations'] += len(trials)
        values = [cp]+[t['cpi'] for t in trials]
        pred = {a: [cp]+[t[a+'_predicted_cpi'] for t in trials] for a in ('scalar','mscs')}
        for i, j in combinations(range(len(values)), 2):
            if abs(values[i]-values[j]) <= tol:
                continue
            truth = np.sign(values[i]-values[j]); losses = []
            for a in ('scalar','mscs'):
                delta = pred[a][i]-pred[a][j]
                loss = 1 if abs(delta) <= tol else 0 if np.sign(delta) == truth else 2
                st['loss2'][a] += loss; losses.append(loss)
            st['comparisons'] += 1
            st['discriminating_comparisons'] += int(losses[0] != losses[1])
    state['completed_rounds'] += 1


def validate(out):
    plan = read(out/'plan.json'); labels_data = read(out/'labels.json')
    labels = {r['id']: r for r in labels_data['records']}
    assert all(sha(p) == h for p, h in plan['source_sha256'].items())
    summary = read(out/'summary.json')
    seen_selection = []; inventory = {}; round_count = 0; pool_count = 0
    for ctx in plan['contexts']:
        directory = Path(ctx['source_directory']); base = directory/'walks'/ctx['id']/'local'
        state = dict(completed_rounds=0, buckets={}); seen = {config_key(ctx['root'])}
        root = ctx['root']; cp = ctx['root_cpi']; radius = 4; found = set()
        for path in sorted(base.glob('decision_*.json')):
            d = read(path); round_count += 1
            assert d['root'] == root and d['radius'] == radius
            same(d['root_cpi'], cp)
            assert d['allocator_before'] == state
            assert d['candidates'] == moves(root, radius, seen)
            arr = prediction_replay(directory, d, 'local', compute_w_min(root)); arr.close()
            for kind in KINDS:
                pool_count += 1; ids = independent_pool(d, kind)
                row = next(r for r in plan['survey'] if r['decision_file'] == str(path) and r['pool'] == kind)
                assert ids == row['pool_indices']
                if len(ids) < 2:
                    assert not row['selected']; continue
                pairs = independent_pairs(d, ids)
                assert row['chosen'] == pairs
                if kind == 'full':
                    assert pairs['local'] == d['chosen']
                changed = [a for a in ARMS[1:] if set(pairs[a]) != set(pairs['scalar'])]
                lc = set(pairs['local']) != set(pairs['hybrid'])
                assert changed == row['disagreements'] and lc == row['local_vs_hybrid']
                triggers = []
                if changed and kind not in found:
                    triggers.append('first_scalar_disagreement_'+kind); found.add(kind)
                if kind == 'full' and lc and 'local_change' not in found:
                    triggers.append('first_local_vs_hybrid'); found.add('local_change')
                assert row['selected'] == bool(triggers)
                if not triggers:
                    continue
                cell = plan['cells'][len(seen_selection)]
                seen_selection.append(cell['cell_id'])
                assert cell['triggers'] == triggers and cell['chosen'] == pairs
                assert cell['root'] == root and cell['root_cpi'] == cp and cell['pool'] == kind
                assert cell['area_cap'] == ctx['area_cap']
                required_ids = sorted(set(sum(pairs.values(), [])))
                assert sorted(map(int, cell['label_ids'])) == required_ids
                for arm in ARMS:
                    assert len(pairs[arm]) == len(set(pairs[arm])) == 2
                for i in required_ids:
                    a = d['candidates'][i]; lid = label_id(ctx['bench'], a['config'])
                    assert cell['label_ids'][str(i)] == lid
                    assert legal(a['config']) and config_key(a['config']) not in seen
                    assert [p for p in root if root[p] != a['config'][p]] == [a['parameter']]
                    record = inventory.setdefault(lid, dict(id=lid, bench=ctx['bench'],
                        config=a['config'], cells=[], root_cpis=[]))
                    record['cells'].append(cell['cell_id']); record['root_cpis'].append(cp)
            r = read(base/f"round_{d['step']:02}.json")
            assert sha(path) == r['decision_sha256']
            assert r['trials'] and len(r['trials']) == 2
            for i, t in zip(d['chosen'], r['trials']):
                assert all(t[k] == v for k, v in d['candidates'][i].items())
                seen.add(config_key(t['config']))
            advance_feedback(state, r); assert state == r['allocator_after']
            accepted = [i for i, t in enumerate(r['trials']) if t['area'] is not None and
                        t['area'] <= ctx['area_cap']+1e-9 and t['cpi'] < cp*(1-1e-6)]
            best = min(accepted, key=lambda i: r['trials'][i]['cpi']) if accepted else None
            assert best == r['accepted_trial']
            if best is None:
                radius = max(1, radius//2)
            else:
                t = r['trials'][best]; root = t['config']; cp = t['cpi']; radius = 4
            assert r['next_root'] == root; same(cp, r['next_cpi'])
    assert len(seen_selection) == len(plan['cells']) <= 25
    assert list(inventory.values()) == plan['required_labels']
    assert set(labels) == set(inventory)
    provenance = read(out/'historical_provenance.json')
    new_count = 0
    for lid, record in labels.items():
        assert all(record[k] == v for k, v in inventory[lid].items())
        if record['origin'] == 'new_forward':
            new_count += 1
            assert sha(record['stack_file']) == record['stack_sha256']
            with np.load(record['stack_file']) as arr:
                m = arr['m']; width = compute_w_min(record['config'])
                cs = np.minimum(100., 1/(np.maximum(.01, m[..., 0])*width+1e-8))
                same(record['cpi'], cs.mean())
                same(record['pressure_cpi'], (cs[..., None]*m[..., 1:]).mean(0))
        else:
            assert provenance[lid]
            for src in provenance[lid]:
                for key, hashkey in [('file','sha256'), ('plan','plan_sha256'), ('inputs_file','inputs_sha256')]:
                    assert sha(src[key]) == src[hashkey]
                t = read(src['file'])['trials'][src['trial_index']]
                assert t['config'] == record['config']; same(t['cpi'], record['cpi'])
                if t['area'] is not None:
                    same(t['area'], record['area'])
        needed = any(record['cpi'] < cp*(1-1e-6) for cp in record['root_cpis'])
        assert needed == record['area_required_for_outcome']
    for cell, row in zip(plan['cells'], summary['cells']):
        assert cell['cell_id'] == row['cell_id'] and row['chosen'] == cell['chosen']
        for field in ('outcomes', 'unconstrained'):
            for arm in ARMS:
                records = [labels[cell['label_ids'][str(i)]] for i in cell['chosen'][arm]]
                cp = cell['root_cpi']; best = cp; lower = cp
                for r in records:
                    if r['cpi'] >= cp*(1-1e-6):
                        continue
                    if field == 'unconstrained' or (r['area'] is not None and r['area'] <= cell['area_cap']+1e-9):
                        best = min(best, r['cpi']); lower = min(lower, r['cpi'])
                    elif r['area'] is None:
                        lower = min(lower, r['cpi'])
                value = row[field][arm]
                assert value['resolved'] == (lower == best)
                assert value['best_cpi'] == (best if lower == best else None)
                assert value['cpi_lower'] == lower and value['cpi_upper'] == best
                if lower == best:
                    same(value['gain_pct'], 100*(1-best/cp))
    assert new_count == labels_data['new_forward_calls'] == summary['new_forward_calls']
    assert new_count+len(provenance) == len(labels)
    assert summary['logical_queries_all_arms'] == len(plan['cells'])*8
    assert round_count == 30 and pool_count == 120
    return dict(passed=True, original_rounds_replayed=round_count,
                pools_replayed=pool_count, selected_cells=len(seen_selection),
                unique_labels=len(labels), new_forward_calls=new_count,
                policy_unchanged=True, prior_feedback_only=True, sources_unchanged=True,
                equal_two_candidate_budget=True, unknown_areas_propagated=True,
                scope='independent implementation replay; frozen surrogate, not hardware')


if __name__ == '__main__':
    import sys
    import json
    print(json.dumps(validate(Path(sys.argv[1])), indent=2))
