"""Reviewable fork: only proposal allocation/feedback differ from the old walk."""
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from local_support import (HERE,STEPS,TOPK,config_key,compute_w_min,moves,LocalModel,
    FeatureCache,exact,contributions,LocalAllocator)
from run_adaptive import save,status,array_save,sha,check_root_derivatives
OUT=HERE/'results_s42_01'

def walk(context, arm, model, inputs, area, record_truth):
    prefix = Path('walks')/context['id']/arm
    cache = FeatureCache(inputs)
    cur = dict(context['root']); width = compute_w_min(cur)
    cap, radius = context['area_cap'], 4
    seen = {config_key(cur)}
    current_cpi = context['root_cpi']; current_area = context['root_area']
    initial_cpi = current_cpi
    rounds, counts = [], Counter()
    timing = {k: 0. for k in ('root_forward', 'jacobian', 'ranking', 'candidate_forward', 'area', 'diagnostics')}
    local, local_key = None, None
    start = time.perf_counter()
    initial = None
    allocator = LocalAllocator()
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
        scalar_scores = prediction['scalar_cpi'].cpu().numpy()
        mscs_scores = prediction['mscs_cpi'].cpu().numpy() if prediction['mscs_cpi'] is not None else None
        allocation_before = allocator.snapshot()
        allocation_mode = 'local_anchor'
        order, proposal_sources, local_ranking = allocator.choose(cand, scalar_scores, mscs_scores)
        timing['ranking'] += time.perf_counter()-t
        decision = {'step': step, 'root': cur, 'root_cpi': current_cpi, 'root_area': current_area,
                    'radius': radius, 'candidates': cand, 'predicted_cpis': scores.tolist(),
                    'predicted_cpi_slopes': prediction['cpi_slope'].tolist(), 'chosen': order,
                    'scalar_predicted_cpis': scalar_scores.tolist(),
                    'mscs_predicted_cpis': mscs_scores.tolist() if mscs_scores is not None else None,
                    'proposal_sources': proposal_sources,
                    'allocator_before': allocation_before, 'allocation_mode': allocation_mode,
                    'local_ranking': local_ranking,
                    'phase': 'before_selected_candidate_network_and_area_evaluations'}
        snapshot = prefix/f'predictions_{step:02}.npz'
        array_save(snapshot, root_m=local.m, root_cs=local.cs, root_ell=local.ell,
                   stage_slope=prediction['stage_slope'], logodds_slope=prediction['logodds_slope'],
                   predicted_m=prediction['mscs'])
        decision.update(prediction_file=str(snapshot), prediction_sha256=sha(OUT/snapshot))
        decision_name = prefix/f'decision_{step:02}.json'
        save(decision_name, decision); frozen = sha(OUT/decision_name)
        if step == 0:
            initial = dict(decision)
            if arm != 'scalar_excess':
                t = time.perf_counter()
                save(prefix/'derivative_check.json', check_root_derivatives(model, local, xs[0]-local.x))
                timing['diagnostics'] += time.perf_counter()-t
                counts['diagnostic_jvp_calls'] += 1
        trials = []
        for slot, j in enumerate(order):
            action = cand[j]; c = action['config']; k = config_key(c)
            assert k not in seen
            t = time.perf_counter(); value, mm = exact(model, xs[j], width)
            timing['candidate_forward'] += time.perf_counter()-t
            counts['candidate_forward_calls'] += 1
            t = time.perf_counter(); aa = area.get(c); timing['area'] += time.perf_counter()-t
            counts['area_requests'] += 1; seen.add(k)
            record_truth(c, value, mm)
            trial = dict(action, predicted_cpi=float(scores[j]), cpi=value, area=aa,
                         proposed_by=proposal_sources[slot], scalar_predicted_cpi=float(scalar_scores[j]),
                         mscs_predicted_cpi=float(mscs_scores[j]) if mscs_scores is not None else None,
                         area_error=area.store.errors.get(area.store.key(c)) if aa is None else None,
                         feasible=aa is not None and aa <= cap+1e-9,
                         query=int(counts['candidate_forward_calls']),
                         predicted_gain=(current_cpi-float(scores[j]))/current_cpi,
                         actual_gain=(current_cpi-value)/current_cpi)
            if arm != 'scalar_excess':
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
        assert allocator.snapshot() == allocation_before
        feedback = allocator.observe(current_cpi, [cand[j] for j in order],
            [v['scalar_predicted_cpi'] for v in trials],
            [v['mscs_predicted_cpi'] for v in trials], [v['cpi'] for v in trials])
        accepted = [i for i, v in enumerate(trials) if v['feasible'] and v['cpi'] < current_cpi*(1-1e-6)]
        best = min(accepted, key=lambda i: trials[i]['cpi']) if accepted else None
        for j, trial in enumerate(trials):
            trial['accepted'] = j == best
        rec = dict(step=step, root=cur, root_cpi=current_cpi, root_area=current_area,
                   radius=radius, trials=trials, accepted_trial=best, decision_file=str(decision_name),
                   decision_sha256=frozen, ranking_feedback=feedback,
                   allocator_after=allocator.snapshot())
        if best is not None:
            cur = dict(trials[best]['config']); current_cpi = trials[best]['cpi']; current_area = trials[best]['area']
            radius = 4
        else:
            radius = max(1, radius//2)
        rec.update(next_root=cur, next_cpi=current_cpi, next_area=current_area)
        assert sha(OUT/decision_name) == frozen
        save(prefix/f'round_{step:02}.json', rec); rounds.append(rec)
        status('greedy', context=context['id'], arm=arm, round=step+1,
               verified=counts['candidate_forward_calls'], gain_pct=100*(1-current_cpi/initial_cpi),
               allocation=allocation_mode, bucket_losses=allocator.state['buckets'])
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
              'initial_prediction_file': str(prefix/'predictions_00.npz'),
              'allocator_final': allocator.snapshot()}
    save(prefix/'summary.json', result)
    return result, initial

