"""Evaluate only the frozen union, in a separate process from prediction screening."""
import sqlite3
import time
import torch
from fresh_disagreement_support import *
import run_adaptive as area_runner


def main():
    start = time.perf_counter(); plan = read(OUT/'confirmation_plan.json'); verify_sources(plan)
    frozen = sha(OUT/'confirmation_plan.json')
    exposed = {(r['bench'], config_key(r['config'])) for r in read(OUT/'exclusions.json')['exposed']}
    area_runner.OUT = OUT; area = area_runner.Areas()
    for directory in read(OUT/'exclusions.json')['directories']:
        for p in Path(directory).glob('*/area_store/areas.sqlite'):
            area.old.append(sqlite3.connect(f'file:{p}?mode=ro', uri=True))
    root_checks = []
    for ctx in plan['contexts']:
        assert area.store.sources == ctx['area_sources']
        aa = area.get(ctx['root']); assert aa is not None
        np.testing.assert_allclose(aa, ctx['root_area'], rtol=0, atol=1e-6)
        root_checks.append(dict(context=ctx['id'], area=aa, passed=True))
    model = load_model() if plan['required_labels'] else None
    records = []; forward_seconds = 0.
    for req in plan['required_labels']:
        assert sha(req['feature_file']) == req['feature_sha256']
        x = torch.from_numpy(np.load(req['feature_file']))
        t = time.perf_counter(); cp, m = exact(model, x, compute_w_min(req['config']))
        forward_seconds += time.perf_counter()-t
        needed = any(cp < root*(1-1e-6) for root in req['root_cpis'])
        aa = area.get(req['config']) if needed else None
        error = area.store.errors.get(area.store.key(req['config'])) if needed else None
        stack = OUT/'new_stacks'/f"{req['id']}.npz"; array_save(stack, m=m)
        rec = dict(req, cpi=cp, area=aa, area_error=error,
            area_required_for_outcome=needed,
            area_status='known' if aa is not None else 'failed_required' if needed else 'irrelevant_nonimprovement',
            origin='new_forward', stack_file=str(stack), stack_sha256=sha(stack),
            pressure_cpi=contributions(m, compute_w_min(req['config'])).tolist(),
            in_prior_exposure_inventory=(req['bench'],config_key(req['config'])) in exposed)
        records.append(rec); save(OUT/'labels'/f"{req['id']}.json", rec)
    assert sha(OUT/'confirmation_plan.json') == frozen
    save(OUT/'labels.json', dict(records=records, new_candidate_forwards=len(records),
        reused_candidate_labels=0, candidate_forward_seconds=forward_seconds,
        area_counts=dict(area.counts), area_failures=area.store.n_fail,
        area_errors=area.store.errors, root_checks=root_checks,
        wall_seconds=time.perf_counter()-start))
    from report_disagreement import report
    report(OUT)
    from validate_disagreement import validate
    save(OUT/'validation.json', validate(OUT))
    save(OUT/'process_exit.json', dict(returncode=0, evaluation_and_validation_seconds=time.perf_counter()-start,
                                      confirmation_plan_sha256=frozen))
    print(json.dumps(dict(phase='complete', roots=len(plan['contexts']), cells=len(plan['cells']),
                          candidate_forwards=len(records), elapsed_seconds=time.perf_counter()-start)), flush=True)


if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception:
        save(OUT/'failure.json', dict(traceback=traceback.format_exc()))
        raise
