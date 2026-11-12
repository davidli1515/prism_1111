"""Independent replay of a one-sided, history-based withdrawal diagnostic."""
import numpy as np
from secant_support import HERE, read, sha, config_key, compute_w_min
from matched_support import label_id
from width_priority_validate import cpi


def validate(out):
    plan = read(out / 'plan.json')
    summary = read(out / 'summary.json')
    freeze = read(out / 'freeze.json')
    source = HERE / 'active_resources_s42_01'
    assert read(source / 'validation.json')['passed']
    assert all(sha(p) == h for p, h in plan['source_sha256'].items())
    assert plan['steps'] == list(range(1, 6))
    assert plan['arms'] == ['scalar_raw', 'mscs_raw', 'scalar_memory',
                            'mscs_memory', 'scalar_guard', 'mscs_guard']
    expected = [(b, s) for b in plan['benchmarks'] for s in plan['steps']]
    assert [(r['bench'], r['step']) for r in summary['rows']] == expected
    assert len(freeze['decision_sha256']) == 15
    labels = {}; logical = 0; advantages = []; against_raw = []
    improved = 0; winning = set()
    for row in summary['rows']:
        bench, step = row['bench'], row['step']
        path = out / 'decisions' / f'{bench}__{step}.json'
        d = read(path)
        assert sha(path) == freeze['decision_sha256'][str(path)]
        src = source / 'walks' / bench / 'mscs_memory' / f'decision_{step:02}.json'
        assert d['source'] == str(src) and sha(src) == d['source_sha256']
        old = read(src); candidates = old['candidates']
        history = [read(source / 'plan.json')['root']]
        for s in range(step):
            history.append(read(src.parent / f'round_{s:02}.json')['trial']['config'])
        assert history == old['center_configs']
        assert [label_id(bench, c) for c in history] == old['history_ids']
        assert len({config_key(c) for c in history}) == len(history)
        mask = [any(a['config'][k] < v for k, v in old['root'].items())
                for a in candidates]
        assert mask == d['withdrawal']
        for kind in ('scalar', 'mscs'):
            raw = old['raw_' + kind + '_scores']; memory = old[kind + '_scores']
            assert d['scores'][kind + '_raw'] == raw
            assert d['scores'][kind + '_memory'] == memory
            guard = [max(r, m) if down else r for r, m, down in zip(raw, memory, mask)]
            assert d['scores'][kind + '_guard'] == guard
            assert all(g >= r for g, r in zip(guard, raw))
            assert all(g == r for g, r, down in zip(guard, raw, mask) if not down)
        assert all(all(a['config'][k] == old['root'][k] for k in ('l2_size', 'l2_assoc'))
                   for a in candidates)
        resolved = True
        for arm in plan['arms']:
            scores = d['scores'][arm]
            assert np.isfinite(scores).all()
            order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
            prefix = d['area_prefixes'][arm]
            assert [p['index'] for p in prefix] == order[:len(prefix)]
            choice = None
            for j, p in enumerate(prefix):
                if p['area'] is None:
                    assert p['error'] and j == len(prefix) - 1
                    resolved = False; break
                if p['area'] <= plan['area_cap'] + 1e-9:
                    assert j == len(prefix) - 1
                    choice = p['index']; break
            assert choice == d['choices'][arm]
            resolved &= choice is not None
        assert resolved == d['resolved'] == row['resolved']
        if not resolved:
            assert not row['arms']; continue
        assert set(row['arms']) == set(plan['arms'])
        for arm, r in row['arms'].items():
            i = d['choices'][arm]; config = candidates[i]['config']; lid = label_id(bench, config)
            assert (r['index'], r['config'], r['label_id']) == (i, config, lid)
            assert lid not in old['history_ids']
            with np.load(out / 'truth' / bench / f'{lid}.npz') as z:
                predicted = float(cpi(z['m'], compute_w_min(config)).mean())
            np.testing.assert_allclose(predicted, r['cpi'], rtol=3e-6)
            np.testing.assert_allclose(max(0., (old['root_cpi'] - r['cpi']) / old['root_cpi']),
                                       r['gain'], atol=1e-12)
            labels[(bench, lid)] = config; logical += 1
        a = row['arms']; gain = a['mscs_guard']['gain']
        envelope = max(a[k]['gain'] for k in ('scalar_raw', 'scalar_memory', 'scalar_guard'))
        advantages.append(gain - envelope); against_raw.append(gain - a['mscs_raw']['gain'])
        improved += gain > a['mscs_raw']['gain'] + 1e-6
        if gain > envelope + 1e-6:
            winning.add(bench)
    physical = reused = 0
    for (bench, lid), config in labels.items():
        rec = read(out / 'truth' / bench / f'{lid}.json')
        assert rec['config'] == config
        if rec['origin'] == 'new_forward':
            assert rec['source'] is None; physical += 1
        else:
            assert rec['origin'] == 'old_label_after_freeze'; reused += 1
            assert rec['source'] == str(source / 'truth' / bench / f'{lid}.npz')
            with np.load(rec['source']) as old, np.load(out / 'truth' / bench / f'{lid}.npz') as new:
                np.testing.assert_array_equal(old['m'], new['m'])
    assert physical == summary['new_candidate_forwards']
    assert reused == summary['reused_old_labels']
    assert logical == summary['logical_candidate_queries']
    assert summary['new_root_jacobians'] == 0
    gate = (all(r['resolved'] for r in summary['rows']) and improved >= 2 and len(winning) >= 2
            and np.mean(advantages) > 1e-6 and np.mean(against_raw) > 1e-6)
    assert bool(gate) == summary['pilot_gate_passed']
    assert improved == summary['improved_vs_raw_mscs']
    assert sorted(winning) == summary['winning_workloads']
    np.testing.assert_allclose(np.mean(advantages), summary['mean_advantage_vs_scalar_envelope'], atol=1e-12)
    np.testing.assert_allclose(np.mean(against_raw), summary['mean_advantage_vs_raw_mscs'], atol=1e-12)
    return dict(passed=True,all_states_and_private_histories_replayed=True,
                raw_memory_and_guard_scores_checked=True,withdrawals_never_more_optimistic=True,
                rankings_areas_labels_and_budgets_replayed=True,all_choices_frozen_before_labels=True,
                sources_preserved=True)


if __name__ == '__main__':
    import json
    print(json.dumps(validate(HERE / 'withdrawal_guard_s42_01'), indent=2))
