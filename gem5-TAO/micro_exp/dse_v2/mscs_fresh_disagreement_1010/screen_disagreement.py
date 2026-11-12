"""Model predictions only: no candidate network forward in this process."""
import time
import torch
from fresh_disagreement_support import *
from core import LocalModel as OriginalLocalModel
from run_adaptive import check_root_derivatives


def main():
    start = time.perf_counter(); plan = read(OUT/'screen_plan.json'); verify_sources(plan)
    model = load_model(); inputs = {}; caches = {}; screened = []; contexts = []; cells = []; required = {}
    timing = dict(root_forward=0., jacobian=0., ranking=0., diagnostics=0.)
    source_hashes = dict(plan['source_sha256'])
    source_hashes[str(OUT/'screen_plan.json')] = sha(OUT/'screen_plan.json')
    for schedule in plan['schedules']:
        for ctx in schedule['roots']:
            c = ctx['root']; key = (ctx['bench'], c['l1_size']); width = compute_w_min(c)
            if key not in inputs:
                inputs[key] = Inputs(*key); caches[key] = FeatureCache(inputs[key])
            cache = caches[key]
            local = LocalModel(model, cache.get(c), width, 'local')
            np.testing.assert_allclose(local.cpi, ctx['root_cpi'], rtol=3e-6)
            timing['root_forward'] += local.forward_seconds; timing['jacobian'] += local.backward_seconds
            candidates = moves(c, radius=4, seen={config_key(c)})
            xs = torch.stack([cache.get(a['config']) for a in candidates])
            t = time.perf_counter(); pred = local.predict(xs); timing['ranking'] += time.perf_counter()-t
            prefix = Path('screened')/ctx['id']; snapshot = prefix/'predictions.npz'
            array_save(OUT/snapshot, root_m=local.m, root_cs=local.cs, root_ell=local.ell,
                       stage_slope=pred['stage_slope'], logodds_slope=pred['logodds_slope'], predicted_m=pred['mscs'])
            d = dict(step=0, root=c, root_cpi=local.cpi, root_area=ctx['root_area'], radius=4,
                candidates=candidates, scalar_predicted_cpis=pred['scalar_cpi'].tolist(),
                mscs_predicted_cpis=pred['mscs_cpi'].tolist(), predicted_cpis=pred['mscs_cpi'].tolist(),
                predicted_cpi_slopes=pred['cpi_slope'].tolist(),
                allocator_before=LocalAllocator().snapshot(), prediction_file=str(snapshot),
                prediction_sha256=sha(OUT/snapshot), phase='before ANY candidate network/area query')
            decision = OUT/prefix/'decision.json'; save(decision, d)
            save(OUT/prefix/'inputs.json', dict(slice_ids=inputs[key].ids))
            source_hashes[str(decision)] = sha(decision); source_hashes[str(OUT/snapshot)] = sha(OUT/snapshot)
            pools = []
            for kind in KINDS:
                ids = pool_indices(d, kind)
                if len(ids) < 2:
                    pools.append(dict(pool=kind, pool_indices=ids, chosen=None, disagreements=[])); continue
                pairs = choices(d, ids); assert pairs['local'] == pairs['hybrid']
                changed = [a for a in ARMS[1:] if set(pairs[a]) != set(pairs['scalar'])]
                pools.append(dict(pool=kind, pool_indices=ids, chosen=pairs, disagreements=changed))
            first = next((p['pool'] for p in pools if p['disagreements']), None)
            rec = dict(context=ctx, decision_file=str(decision), pools=pools,
                       first_disagreement=first, selected=first is not None)
            screened.append(rec)
            print(json.dumps(dict(phase='screen', context=ctx['id'], first_disagreement=first,
                                  elapsed_seconds=time.perf_counter()-start)), flush=True)
            if first is None:
                del local, xs, pred; continue
            contexts.append(dict(ctx, root_cpi=local.cpi))
            t = time.perf_counter()
            # Independent scalar backprop verifies the baseline is not weakened.
            scalar = OriginalLocalModel(model, cache.get(c), width, 'scalar_excess')
            torch.testing.assert_close(scalar.gc, local.gc, rtol=2e-4, atol=2e-6)
            torch.testing.assert_close(scalar.predict(xs)['cpi'], pred['scalar_cpi'], rtol=3e-6, atol=2e-6)
            check = check_root_derivatives(model, local, xs[0]-local.x)
            save(OUT/prefix/'derivative_check.json', dict(check, independent_scalar_backward_outputs=4))
            timing['diagnostics'] += time.perf_counter()-t
            selected_kinds = ['full']+([] if first == 'full' else [first])
            for kind in selected_kinds:
                pool = next(p for p in pools if p['pool'] == kind)
                cell = dict(pool, cell_id=f"{ctx['id']}_{kind}", context=ctx['id'], bench=ctx['bench'],
                    step=0, root=c, root_cpi=local.cpi, radius=4, area_cap=ctx['area_cap'],
                    decision_file=str(decision), primary=kind == 'full', label_ids={})
                for i in sorted(set(sum(pool['chosen'].values(), []))):
                    cn = candidates[i]['config']; lid = label_id(ctx['bench'], cn)
                    cell['label_ids'][str(i)] = lid
                    if lid not in required:
                        feature = OUT/'features'/f'{lid}.npy'; feature.parent.mkdir(exist_ok=True)
                        with feature.open('xb') as f:
                            np.save(f, xs[i].numpy())
                        required[lid] = dict(id=lid, bench=ctx['bench'], config=cn,
                            feature_file=str(feature), feature_sha256=sha(feature), cells=[], root_cpis=[])
                    required[lid]['cells'].append(cell['cell_id']); required[lid]['root_cpis'].append(local.cpi)
                cells.append(cell)
            del local, scalar, xs, pred
            break
    for path in (OUT/'screened').rglob('*.json') if (OUT/'screened').exists() else []:
        source_hashes[str(path)] = sha(path)
    verify_sources(plan)
    save(OUT/'confirmation_plan.json', dict(seed=42, source_sha256=source_hashes,
        screen_plan_sha256=sha(OUT/'screen_plan.json'), contexts=contexts, cells=cells,
        screened=screened, required_labels=list(required.values()),
        target_roots=2, selected_roots=len(contexts), unfilled_root_quota=2-len(contexts),
        root_forward_calls=len(screened), backward_scalar_outputs=12*len(screened),
        diagnostic_root_forwards=len(contexts), diagnostic_backward_outputs=4*len(contexts),
        diagnostic_jvp_calls=len(contexts), candidate_forwards_before_plan=0,
        timing_seconds=dict(timing, feature_build=sum(c.seconds for c in caches.values()),
                            load_inputs=sum(i.load_seconds for i in inputs.values())),
        wall_seconds=time.perf_counter()-start,
        scope=plan['scope'], phase='ALL roots and candidates frozen before any candidate network forward'))
    print(json.dumps(dict(phase='confirmation_frozen', selected_roots=len(contexts),
        screened_roots=len(screened), cells=len(cells), candidate_union=len(required),
        wall_seconds=time.perf_counter()-start)), flush=True)


if __name__ == '__main__':
    main()
