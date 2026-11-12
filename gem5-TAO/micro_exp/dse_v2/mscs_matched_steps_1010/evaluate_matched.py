"""Complete only the frozen list's missing labels; never select using outcomes."""
from collections import Counter
import sqlite3
import time
import traceback
import torch
from matched_support import *
from local_support import Inputs, FeatureCache, load_model, exact, contributions
import run_adaptive as area_runner


def historical_labels(plan):
    required = {v['id']: v for v in plan['required_labels']}; known = {}; provenance = {}
    dirs = set(SOURCES)
    for ctx in plan['contexts']:
        dirs.update(Path(p) for p in ctx.get('controls', {}).values())
    for directory in sorted(dirs):
        pp = read(directory/'plan.json')
        assert pp['source_sha256'][str(DEFAULT_CKPT)] == plan['checkpoint_sha256']
        for ctx in pp['contexts']:
            if ctx['bench'] not in {v['bench'] for v in required.values()}:
                continue
            for path in sorted((directory/'walks'/ctx['id']).glob('*/round_*.json')):
                rr = read(path)
                for i, trial in enumerate(rr['trials']):
                    lid = label_id(ctx['bench'], trial['config'])
                    if lid not in required:
                        continue
                    src = dict(file=str(path), sha256=sha(path), trial_index=i,
                               plan=str(directory/'plan.json'), plan_sha256=sha(directory/'plan.json'),
                               inputs_file=str(directory/'inputs'/f"{ctx['id']}.json"))
                    src['inputs_sha256'] = sha(src['inputs_file'])
                    ref_ctx = next(c for c in plan['contexts'] if c['bench'] == ctx['bench'] and
                                   c['root']['l1_size'] == trial['config']['l1_size'])
                    reference = read(Path(ref_ctx['source_directory'])/'inputs'/f"{ref_ctx['id']}.json")
                    assert read(src['inputs_file'])['slice_ids'] == reference['slice_ids']
                    assert ctx['area_sources'] == ref_ctx['area_sources']
                    if lid not in known:
                        known[lid] = dict(required[lid], cpi=trial['cpi'], area=trial['area'],
                            area_error=trial.get('area_error'), origin='historical', pressure_cpi=None)
                    rec = known[lid]
                    np.testing.assert_allclose(rec['cpi'], trial['cpi'], rtol=3e-6, atol=1e-7)
                    if trial['area'] is not None:
                        if rec['area'] is not None:
                            np.testing.assert_allclose(rec['area'], trial['area'], rtol=0, atol=1e-6)
                        rec['area'] = trial['area']; rec['area_error'] = None
                    if trial.get('explanation'):
                        e = trial['explanation']
                        pc = np.array(e['root_pressure_cpi'])+np.array(e['actual_pressure_cpi_change'])
                        if rec['pressure_cpi'] is not None:
                            np.testing.assert_allclose(pc, rec['pressure_cpi'], rtol=5e-5, atol=2e-6)
                        rec['pressure_cpi'] = pc.tolist()
                    provenance.setdefault(lid, []).append(src)
    return known, provenance, dirs


def main():
    start = time.perf_counter(); plan = read(OUT/'plan.json'); frozen = sha(OUT/'plan.json')
    assert all(sha(p) == h for p, h in plan['source_sha256'].items())
    known, provenance, directories = historical_labels(plan)
    save(OUT/'historical_provenance.json', provenance)
    missing = [r for r in plan['required_labels'] if r['id'] not in known]
    save(OUT/'label_inventory_before_queries.json', dict(plan_sha256=frozen,
        historical=len(known), missing=[r['id'] for r in missing]))
    print(json.dumps(dict(phase='frozen_label_inventory', total=len(plan['required_labels']),
                          reused=len(known), new_forwards=len(missing))), flush=True)
    area_runner.OUT = OUT
    area = area_runner.Areas()
    for directory in sorted(directories | {ROOT/'mscs_adaptive_newroots_1010/results_s42_01',
                                          ROOT/'mscs_local_feedback_1010/fresh_s42_01'}):
        p = directory/'area_store/areas.sqlite'
        if p.exists():
            area.old.append(sqlite3.connect(f'file:{p}?mode=ro', uri=True))
    assert all(c['area_sources'] == area.store.sources for c in plan['contexts'])
    model = load_model() if missing else None
    inputs = {}; cache = {}; forwards = 0; labels = []
    for req in plan['required_labels']:
        lid = req['id']; rec = known.get(lid)
        if rec is None:
            key = (req['bench'], req['config']['l1_size'])
            if key not in inputs:
                inputs[key] = Inputs(*key); cache[key] = FeatureCache(inputs[key])
                refs = [c for c in plan['contexts'] if c['bench'] == key[0] and c['root']['l1_size'] == key[1]]
                for ctx in refs:
                    assert inputs[key].ids == read(Path(ctx['source_directory'])/'inputs'/f"{ctx['id']}.json")['slice_ids']
                save(OUT/'inputs'/f'{key[0]}_l1_{key[1]}.json', dict(slice_ids=inputs[key].ids))
            width = compute_w_min(req['config'])
            cpi, m = exact(model, cache[key].get(req['config']), width); forwards += 1
            snapshot = OUT/'new_stacks'/f'{lid}.npz'; snapshot.parent.mkdir(exist_ok=True)
            with snapshot.open('xb') as f:
                np.savez_compressed(f, m=m.numpy())
            rec = dict(req, cpi=cpi, area=None, area_error=None, origin='new_forward',
                       pressure_cpi=contributions(m, width).tolist(),
                       stack_file=str(snapshot), stack_sha256=sha(snapshot))
        else:
            rec = dict(rec)
        needed_area = any(rec['cpi'] < root*(1-1e-6) for root in req['root_cpis'])
        rec['area_required_for_outcome'] = needed_area
        if rec['area'] is None and needed_area:
            rec['area'] = area.get(req['config'])
            rec['area_error'] = area.store.errors.get(area.store.key(req['config']))
        rec['area_status'] = ('known' if rec['area'] is not None else
                              'failed_required' if needed_area else 'irrelevant_nonimprovement')
        labels.append(rec); save(OUT/'labels'/f'{lid}.json', rec)
    assert forwards == len(missing)
    save(OUT/'labels.json', dict(records=labels, new_forward_calls=forwards,
        historical_records=len(known), unique_records=len(labels), area_counts=dict(area.counts),
        area_failures=area.store.n_fail, area_errors=area.store.errors,
        feature_seconds=sum(c.seconds for c in cache.values()), wall_seconds=time.perf_counter()-start))
    assert sha(OUT/'plan.json') == frozen
    from matched_report import report
    report(OUT)
    from validate_matched import validate
    save(OUT/'validation.json', validate(OUT))
    save(OUT/'process_exit.json', dict(returncode=0, wall_seconds=time.perf_counter()-start,
                                      plan_sha256=frozen))
    print(json.dumps(dict(phase='complete', new_forwards=forwards,
                         wall_seconds=time.perf_counter()-start)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save(OUT/'failure.json', dict(error=traceback.format_exc()))
        raise
