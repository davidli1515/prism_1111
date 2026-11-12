#!/usr/bin/env python3
"""One frozen offline diagnostic with one strictly gated small confirmation."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from region_data import (HERE, ROOT, OLD, CKPT, FE, PATTERNS, sha, read, key, legal, axes,
                         enumerate_set, near, scopes, load_saved, make_arrays)
from calibration import ARMS, nested_cv, fit_predict, summarize
from test_intent import controls

OUT = HERE / 'results_s42_01'
START = time.monotonic()
SOURCES = {}


def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False,
                  default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item())


def status(phase, **kw):
    record = dict(phase=phase, pid=os.getpid(), elapsed_seconds=time.monotonic()-START, **kw)
    p = OUT / 'status.tmp'; p.write_text(json.dumps(record, indent=2)); p.replace(OUT / 'status.json')
    print(json.dumps(record), flush=True)


def sources():
    previous = read(OLD / 'summary.json')['source_sha256']
    for p, h in previous.items():
        assert sha(p) == h, f'Prior preserved source changed: {p}'
    paths = list(HERE.glob('*.py')) + [HERE / 'PROTOCOL.txt', CKPT]
    paths += [OLD / n for n in ('plan.json', 'fresh_roots.json', 'fresh_labels.json', 'summary.json')]
    return dict(previous, **{str(p): sha(p) for p in paths})


def signature(domain):
    return tuple(tuple(domain[p]) for p in FE)


def choose_fresh(roots, rows):
    tests, blocked = [], [r['root'] for r in roots]
    selection_sources = {}
    rng = np.random.RandomState(42)
    for bench in ('602.gcc_s', '605.mcf_s'):
        path = ROOT / 'mscs_guard_v5/results_full_01' / bench / 's42/initial.json'
        archive = read(path)
        assert archive['checkpoint_sha256'] == sha(CKPT)
        selection_sources[str(path)] = sha(path)
        # Eligibility uses configurations only, never candidate CPI, MSCS or area.
        known = {key(c) for c in archive['configs']}
        supported = {signature(r['domain']) for r in rows if r['bench'] == bench and r['pattern'] == 'joint'}
        picked = 0
        for index in rng.permutation(len(archive['configs'])):
            c = archive['configs'][index]
            if not legal(c) or any(near(c, b) for b in blocked):
                continue
            domain = axes(c)
            if any(not 2 <= len(v) <= 3 for v in domain.values()) or signature(domain) not in supported:
                continue
            candidates = enumerate_set(c, domain)
            if any(key(v) in known for v in candidates):
                continue
            tests.append({'bench': bench, 'root': c, 'source_index': int(index),
                          'domain': domain, 'candidates': candidates})
            blocked.append(c); picked += 1
            if picked == 2:
                break
        if picked != 2:
            return None, {'reason': 'coverage_insufficient', 'bench': bench, 'selected': picked,
                          'selection_sources': selection_sources}
    assert len(tests) == 4 and sum(len(t['candidates']) for t in tests) <= 32
    return tests, selection_sources


def confirmation(roots, rows):
    tests, extra = choose_fresh(roots, rows)
    if tests is None:
        save('fresh_selection_stop.json', extra)
        return {'phase': 'coverage_insufficient', 'new_candidate_cpi': 0, 'root_queries': 0}
    SOURCES.update(extra)
    save('fresh_plan.json', {'tests': tests, 'source_sha256': dict(SOURCES),
                            'candidate_limit': 32, 'root_limit': 4})
    plan_sha = sha(OUT / 'fresh_plan.json')
    # The surrogate is imported only after the real-data continuation gate passed.
    sys.path.insert(0, str(ROOT / 'mscs_response_1008'))
    from probe import SliceOracle
    oracles, rr, fresh_rows = {}, [], []
    next_group = max(r['group'] for r in rows)+1
    for i, t in enumerate(tests):
        b = t['bench']
        if b not in oracles:
            oracles[b] = SliceOracle(b)
        m, cpi, ids = oracles[b].query(t['root'])
        rr.append(dict(t, mscs=m.tolist(), root_cpi=cpi, slice_ids=list(ids)))
        for kind, domain, candidates in scopes(t['root']):
            fresh_rows.append({'root_id': len(roots)+i, 'group': next_group+i, 'bench': b,
                               'pattern': kind, 'domain': domain, 'candidates': candidates})
        status('extracting_fresh_roots', completed=i+1, total=4)
    save('fresh_roots.json', rr)
    all_rows, all_roots = rows+fresh_rows, roots+rr
    data = make_arrays(all_roots, all_rows)
    train = np.arange(len(rows)); valid = np.arange(len(rows), len(all_rows))
    y = np.r_[[r['gain'] for r in rows], np.full(len(fresh_rows), np.nan)]
    preds, models = fit_predict(data, all_rows, y, train, valid)
    save('fresh_predictions.json', {'phase': 'before_any_descendant_CPI_access',
                                    'predictions': preds, 'models': models, 'rows': fresh_rows})
    frozen = sha(OUT / 'fresh_predictions.json')
    status('fresh_predictions_frozen', candidate_union=sum(len(t['candidates']) for t in tests))
    cache, reuse_paths = {}, {}
    for b in oracles:
        cache[b] = {}
        for name in ('initial', 'scalar', 'scalar_guard', 'mscs_guard'):
            path = ROOT / 'mscs_guard_v5/results_full_01' / b / f's42/{name}.json'
            archive = read(path)
            assert archive['checkpoint_sha256'] == sha(CKPT)
            reuse_paths[str(path)] = sha(path)
            for c, value in zip(archive['configs'], archive['cpis']):
                k = key(c)
                if k in cache[b]:
                    assert np.isclose(cache[b][k], value, rtol=3e-6)
                cache[b][k] = float(value)
    counts, labels = Counter(), {}
    for i, r in enumerate(rr):
        b = r['bench']; values = {}
        for c in r['candidates']:
            k = key(c)
            if k in cache[b]:
                value = cache[b][k]; counts['reused_candidate_cpi'] += 1
            else:
                assert counts['new_candidate_cpi'] < 32
                _, value, ids = oracles[b].query(c)
                assert list(ids) == r['slice_ids']
                counts['new_candidate_cpi'] += 1
            values[k] = float(value)
        labels[len(roots)+i] = values
        status('fresh_labels', completed=i+1, counts=dict(counts))
    for r in fresh_rows:
        r['descendant_cpis'] = [labels[r['root_id']][key(c)] for c in r['candidates']]
        c0 = all_roots[r['root_id']]['root_cpi']
        r['gain'] = max(0., 1-min(r['descendant_cpis'])/c0)
    save('fresh_labels.json', fresh_rows)
    result = summarize(fresh_rows, np.array([r['gain'] for r in fresh_rows]), preds)
    physical = sum(o.n_queries for o in oracles.values())
    assert physical == counts['new_candidate_cpi']+4 and physical <= 36
    assert sha(OUT / 'fresh_predictions.json') == frozen and sha(OUT / 'fresh_plan.json') == plan_sha
    SOURCES.update(reuse_paths)
    result.update(phase='confirmation_complete', counts=dict(counts), root_queries=4,
                  physical_forward_configs=physical, predictions_sha256=frozen, plan_sha256=plan_sha)
    save('confirmation.json', result)
    return result


def findings(summary):
    dev = summary['development']
    lines = ['MSCS resource-range interaction diagnostic — seed 42',
             '24 examined roots / 21 configuration groups / 40 complete sets; nested group CV.',
             'Development is exploratory, not a fresh confirmation or a safe pruning bound.',
             'Gain MAE in percentage points:']
    for arm, r in dev['metrics'].items():
        lines.append(f"  {arm}: {100*r['mae']:.6f}; max underprediction {100*r['max_underprediction']:.6f}")
    lines += [f"Development gate: {dev['gate']}", f"Final phase: {summary['phase']}",
              f"Additional confirmation: {summary['confirmation']}",
              'All prior sources/checkpoint preserved. No full DSE, area jobs, gem5, or extra seeds.',
              'No result is a proof that all MSCS information is useless/useful.',
              'Tests and independent result verification are saved separately.']
    (OUT / 'FINDINGS.txt').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(exist_ok=False)
    status('preflight')
    SOURCES.update(sources())
    control = controls(); save('controls.json', control)
    assert control['positive_pass'] and control['negative_pass'], 'Synthetic control failed'
    roots, rows = load_saved()
    data = make_arrays(roots, rows)
    save('plan.json', {'seed': 42, 'source_sha256': dict(SOURCES), 'rows': len(rows), 'roots': len(roots),
                       'groups': len({r['group'] for r in rows}), 'max_new_descendants': 32,
                       'max_root_forward_calls': 4, 'protocol_sha256': sha(HERE / 'PROTOCOL.txt')})
    save('dataset.json', {'roots': roots, 'rows': rows})
    save('features.json', data)
    dev = nested_cv(data, rows, progress=lambda n: status('development_cv', completed_folds=n, total=5))
    save('development.json', dev)
    status('development_gate', gate=dev['gate'])
    # Independent recomputation must pass before any potential inference.
    from validate_result import verify_development
    save('development_validation.json', verify_development(OUT))
    if dev['gate']['pass']:
        follow = confirmation(roots, rows)
    else:
        follow = {'phase': 'not_run_development_gate_failed', 'new_candidate_cpi': 0, 'root_queries': 0}
    assert all(sha(p) == h for p, h in SOURCES.items()), 'Preserved source changed during run'
    summary = {'phase': 'complete', 'development': {k: v for k, v in dev.items() if k not in ('folds', 'predictions')},
               'confirmation': follow, 'source_sha256': dict(SOURCES),
               'preservation_checks_passed': True, 'wall_seconds': time.monotonic()-START}
    save('summary.json', summary); findings(summary)
    status('complete', confirmation_phase=follow['phase'], preservation_checks_passed=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT.exists() and not (OUT / 'summary.json').exists():
            (OUT / 'ERROR.txt').write_text(traceback.format_exc())
            status('failed', error_file=str(OUT / 'ERROR.txt'))
        raise
