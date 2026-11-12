"""Independent control-flow/prediction replay and exact feature-label mapping."""
from fresh_disagreement_support import (OUT, read, sha, config_key, label_id,
    ALL_PARAMS, BENCHES, SPEC17, legal, moves, compute_w_min, Inputs, verify_sources)
from validate_matched import independent_pool, independent_pairs
from adaptive_validate import prediction_replay, close, stage_cpi
from pathlib import Path
import numpy as np


def inventory_again(files):
    found, unknown = set(), set()
    for file in files:
        hint = next((b for part in Path(file).parts for b in SPEC17 if part.startswith(b)), None)
        stack = [(read(file), hint)]
        while stack:
            obj, bench = stack.pop()
            if isinstance(obj, list):
                stack.extend((v, bench) for v in obj if isinstance(v, (list, dict))); continue
            if not isinstance(obj, dict):
                continue
            for name in ('bench','benchmark','context','id','root_id'):
                value = str(obj.get(name)); bb = next((b for b in SPEC17 if value.startswith(b)), None)
                if bb:
                    bench = bb; break
            if set(ALL_PARAMS).issubset(obj) and all(isinstance(obj[k], (float,int)) for k in ALL_PARAMS):
                key = config_key({k:obj[k] for k in ALL_PARAMS})
                if bench:
                    found.add((bench,key))
                else:
                    unknown.add(key)
            for k, v in obj.items():
                if isinstance(v, (dict,list)):
                    bb = next((b for b in SPEC17 if str(k).startswith(b)), bench)
                    stack.append((v,bb))
    found.update((b,c) for c in unknown for b in BENCHES)
    return found, unknown


def validate(out):
    screen = read(out/'screen_plan.json'); plan = read(out/'confirmation_plan.json')
    labels_data = read(out/'labels.json'); summary = read(out/'summary.json')
    labels = {r['id']:r for r in labels_data['records']}
    assert sha(out/'screen_plan.json') == plan['screen_plan_sha256']
    verify_sources(plan)
    ex = read(out/'exclusions.json'); assert sha(out/'exclusions.json') == screen['exclusions_sha256']
    exposed, unknown = inventory_again(ex['source_sha256'])
    assert exposed == {(r['bench'],config_key(r['config'])) for r in ex['exposed']}
    assert unknown == {config_key(c) for c in ex['unknown_context_configs']}
    expected_selected = []; expected_cells = []; new_union = {}; inputs = {}
    screens_count = 0
    for schedule, source in zip(screen['schedules'], screen['archives']):
        archive = read(source['path']); points = np.c_[archive['cpis'],archive['areas']]
        # Pairwise dominance + deterministic duplicate removal, independent of the original sort scan.
        front = []
        for i, p in enumerate(points):
            dominates = np.all(points <= p, axis=1) & np.any(points < p, axis=1)
            previous_duplicate = np.any(np.all(points[:i] == p, axis=1))
            if not dominates.any() and not previous_duplicate:
                front.append(i)
        assert front == [r['archive_index'] for r in source['front_eligibility']]
        eligible = [i for i in front if legal(archive['configs'][i]) and
                    (source['bench'],config_key(archive['configs'][i])) not in exposed][:4]
        assert eligible == [c['archive_index'] for c in schedule['roots']]
        actual = [r for r in plan['screened'] if r['context']['bench'] == schedule['bench']]
        assert len(actual) <= len(schedule['roots']) <= 4
        for j, rec in enumerate(actual):
            ctx = rec['context']; assert ctx == schedule['roots'][j]
            screens_count += 1; d = read(rec['decision_file']); c = ctx['root']; width = compute_w_min(c)
            assert (ctx['bench'],config_key(c)) not in exposed
            assert d['root'] == c and d['candidates'] == moves(c, 4, {config_key(c)})
            assert d['allocator_before'] == dict(completed_rounds=0, buckets={})
            close(d['root_cpi'], ctx['root_cpi'], rtol=3e-6)
            arr = prediction_replay(out, d, 'local', width); arr.close()
            first = None
            for pool in rec['pools']:
                ids = independent_pool(d, pool['pool']); assert ids == pool['pool_indices']
                if len(ids) < 2:
                    assert pool['chosen'] is None and pool['disagreements'] == []; continue
                pairs = independent_pairs(d, ids); assert pairs == pool['chosen']
                assert pairs['local'] == pairs['hybrid']
                changed = [a for a in ('mscs','hybrid','local') if set(pairs[a]) != set(pairs['scalar'])]
                assert changed == pool['disagreements']
                if first is None and changed:
                    first = pool['pool']
            assert first == rec['first_disagreement'] and rec['selected'] == (first is not None)
            if first is None:
                continue
            assert j == len(actual)-1
            selected = dict(ctx, root_cpi=d['root_cpi']); expected_selected.append(selected)
            assert read(out/'screened'/ctx['id']/'derivative_check.json')['passed']
            kinds = ['full']+([] if first == 'full' else [first])
            for kind in kinds:
                cell = plan['cells'][len(expected_cells)]; expected_cells.append(cell['cell_id'])
                assert cell['cell_id'] == f"{ctx['id']}_{kind}"
                assert cell['area_cap'] == ctx['area_cap'] and cell['root'] == c
                assert cell['root_cpi'] == d['root_cpi'] and cell['primary'] == (kind == 'full')
                ids = independent_pool(d, kind); pairs = independent_pairs(d, ids)
                assert cell['chosen'] == pairs and cell['pool_indices'] == ids
                required = sorted(set(sum(pairs.values(), [])))
                assert len(required) <= 4 and sorted(map(int,cell['label_ids'])) == required
                for order in pairs.values():
                    assert len(order) == len(set(order)) == 2
                for i in required:
                    a = d['candidates'][i]; cn = a['config']; lid = label_id(ctx['bench'],cn)
                    assert cell['label_ids'][str(i)] == lid
                    assert [p for p in c if cn[p] != c[p]] == [a['parameter']]
                    assert legal(cn) and compute_w_min(cn) == width
                    r = new_union.setdefault(lid, dict(config=cn,bench=ctx['bench'],cells=[],root_cpis=[]))
                    r['cells'].append(cell['cell_id']); r['root_cpis'].append(d['root_cpi'])
        if actual and actual[-1]['selected']:
            assert len(actual) <= len(schedule['roots'])
        else:
            assert len(actual) == len(schedule['roots'])
    assert expected_selected == plan['contexts']
    assert len(expected_cells) == len(plan['cells']) <= 4
    assert plan['selected_roots'] == len(expected_selected) and plan['unfilled_root_quota'] == 2-len(expected_selected)
    assert set(new_union) == set(labels) == {r['id'] for r in plan['required_labels']}
    for req in plan['required_labels']:
        lid = req['id']; r = labels[lid]
        for k, v in new_union[lid].items():
            assert req[k] == r[k] == v
        assert sha(req['feature_file']) == req['feature_sha256']
        # Rebuild exact encoded inputs to verify candidate IDs were not mismatched.
        key = (r['bench'],r['config']['l1_size'])
        if key not in inputs:
            inputs[key] = Inputs(*key)
        original = inputs[key].build(r['config']).numpy()
        np.testing.assert_array_equal(original, np.load(req['feature_file']))
        assert sha(r['stack_file']) == r['stack_sha256']
        with np.load(r['stack_file']) as arr:
            m = arr['m']; cs = stage_cpi(m, compute_w_min(r['config']))
            close(cs.mean(),r['cpi']); close((cs[...,None]*m[...,1:]).mean(0),r['pressure_cpi'])
        assert r['in_prior_exposure_inventory'] == ((r['bench'],config_key(r['config'])) in exposed)
        needs_area = any(r['cpi'] < cp*(1-1e-6) for cp in r['root_cpis'])
        assert r['area_required_for_outcome'] == needs_area
        if r['area'] is None and needs_area:
            assert r['area_status'] == 'failed_required' and r['area_error']
    for cell, row in zip(plan['cells'], summary['cells']):
        assert cell['cell_id'] == row['cell_id'] and cell['chosen'] == row['chosen']
        for field in ('outcomes','unconstrained'):
            for arm, ids in cell['chosen'].items():
                root = cell['root_cpi']; high = root; low = root
                for i in ids:
                    r = labels[cell['label_ids'][str(i)]]
                    if r['cpi'] >= root*(1-1e-6):
                        continue
                    if field == 'unconstrained' or (r['area'] is not None and r['area'] <= cell['area_cap']+1e-9):
                        high = min(high,r['cpi']); low = min(low,r['cpi'])
                    elif r['area'] is None:
                        low = min(low,r['cpi'])
                v = row[field][arm]
                assert v['cpi_lower'] == low and v['cpi_upper'] == high and v['resolved'] == (low==high)
                assert v['best_cpi'] == (high if low==high else None)
                if low == high:
                    close(v['gain_pct'],100*(1-high/root))
        assert row['outcomes']['hybrid'] == row['outcomes']['local']
    for name, group in summary['groups'].items():
        rows = [r for r in summary['cells'] if r['primary'] == (name == 'primary_full')]
        assert len(rows) == group['cells']
        for comparison, value in group['comparisons'].items():
            arm = comparison.split('_vs_')[0]; counts = dict(win=0,tie=0,loss=0,unresolved=0)
            for row in rows:
                a, s = row['outcomes'][arm], row['outcomes']['scalar']
                if a['best_cpi'] is None or s['best_cpi'] is None:
                    counts['unresolved'] += 1; continue
                diff = (s['best_cpi']-a['best_cpi'])/row['root_cpi']
                counts['win' if diff>1e-6 else 'loss' if diff< -1e-6 else 'tie'] += 1
            assert counts == value
    assert plan['candidate_forwards_before_plan'] == 0
    assert screens_count == plan['root_forward_calls'] and plan['backward_scalar_outputs'] == 12*screens_count
    assert len(labels) == summary['candidate_forwards'] == labels_data['new_candidate_forwards'] <= 16
    assert summary['logical_queries_all_arms'] == len(plan['cells'])*8
    return dict(passed=True, sources_unchanged=True, exclusion_inventory_independently_rebuilt=True,
        root_order_and_stop_replayed=True, screened_roots=screens_count,
        selected_roots=len(expected_selected), selected_cells=len(expected_cells),
        predictors_replayed=True, scalar_derivatives_independently_checked=True,
        cold_local_equals_hybrid=True, candidate_input_mappings_rebuilt=len(labels),
        candidate_forwards=len(labels), equal_two_query_budget=True,
        unresolved_area_propagation_checked=True, all_outcomes_and_counts_recomputed=True,
        scope='frozen-surrogate conditional confirmation; not hardware or full search')


if __name__ == '__main__':
    import json
    print(json.dumps(validate(OUT),indent=2))
