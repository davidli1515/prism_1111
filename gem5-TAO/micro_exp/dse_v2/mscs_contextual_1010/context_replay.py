"""Independent numpy replay of frozen decisions, formulas, and audit outcomes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from mechanism import MODES, TAU, compute_w_min, config_key, legal


def read(p):
    return json.loads(Path(p).read_text())


def close(a, b, rtol=5e-5, atol=3e-6):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def replay(out):
    out = Path(out); plan = read(out/'plan.json'); report = {'passed': True, 'roots': []}
    for p,h in plan['source_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    search_calls = audit_calls = 0
    for ctx in plan['contexts']:
        base = out/'roots'/ctx['id']; pool = read(base/'pool.json'); d = read(base/'decision.json')
        arrays = np.load(base/'predictions.npz'); result = read(base/'summary.json')
        assert hashlib.sha256((base/'decision.json').read_bytes()).hexdigest() == result['decision_sha256']
        c0 = d['root_cpi']; w = compute_w_min(ctx['root'])
        assert len(pool) == len({config_key(c['config']) for c in pool})
        for c in pool:
            assert legal(c['config']) and compute_w_min(c['config']) == w
            changed = {k for k in ctx['root'] if c['config'][k] != ctx['root'][k]}
            if c['kind'] == 'single':
                assert changed == {c['parameter']}
            else:
                u, v = c['parts']; assert len(changed) == 2
                assert changed == {pool[u]['parameter'], pool[v]['parameter']}
                assert pool[u]['sign'] == 1 and pool[v]['sign'] == (1 if c['kind'] == 'up_up' else -1)
                for i in (u, v):
                    name = pool[i]['parameter']; assert c['config'][name] == pool[i]['config'][name]
        lo, hi = 1/w, min(100., 1/(.01*w+1e-8))
        excess = np.maximum(arrays['root_cs']-lo, 1e-10)
        sc = np.clip(lo+excess[None]*np.exp(np.clip(arrays['stage_slope']/excess[None], -80, 80)), lo, hi).mean((1,2))
        ell = arrays['root_ell'][None]+arrays['logodds_slope']
        logits = np.concatenate([np.zeros_like(ell[..., :1]), ell], -1)
        logits -= logits.max(-1, keepdims=True); mm = np.exp(logits); mm /= mm.sum(-1, keepdims=True)
        cm = np.minimum(100., 1/(np.maximum(.01, mm[..., 0])*w+1e-8)).mean((1,2))
        close(sc, arrays['scalar_cpi']); close(cm, arrays['mscs_cpi']); close(mm, arrays['predicted_m'])
        # Replay rankings from saved float32 predictions to preserve exact tie order.
        sc, cm = arrays['scalar_cpi'], arrays['mscs_cpi']
        def protected(cp, k):
            c = pool[k]
            if c['kind'] != 'up_down':
                return False
            u,v = c['parts']
            return 0 <= (cp[v]-c0)/c0 < TAU and (cp[k]-cp[u])/c0 >= TAU
        scalar_order = np.argsort(sc, kind='stable')
        anchor = next(int(k) for k in scalar_order if not protected(sc, k))
        labels = {int(f.stem): read(f) for f in (base/'labels').glob('*.json')}
        search_set = set()
        for mode in MODES:
            sel = d['selections'][mode]; score = sc.copy() if mode == 'scalar' else cm.copy()
            allowed = np.ones(len(pool), dtype=bool)
            for k,c in enumerate(pool):
                if c['kind'] == 'single':
                    continue
                if (c['kind'] == 'up_up' and mode in ('no_decouple','no_context') or
                    c['kind'] == 'up_down' and mode in ('no_protect','no_context')):
                    u,v = c['parts']; score[k] = max(lo, cm[u]+cm[v]-c0)
                if mode == 'scalar':
                    allowed[k] = not protected(sc, k)
                elif mode in ('mscs','no_decouple'):
                    allowed[k] = not protected(cm, k)
            second = next(int(k) for k in np.argsort(score, kind='stable') if k != anchor and allowed[k])
            assert sel['chosen'] == [anchor, second]
            close(sel['scores'], score); assert sel['allowed'] == allowed.tolist()
            search_set.update(sel['chosen']); search_calls += 2
            eligible = [labels[k] for k in sel['chosen'] if labels[k]['area'] is not None
                        and labels[k]['area'] <= ctx['area_cap']+1e-9 and labels[k]['cpi'] < c0*(1-1e-6)]
            best = min(eligible, key=lambda z:z['cpi']) if eligible else None
            rr = result['outcomes'][mode]
            assert rr['accepted_candidate'] == (best['candidate'] if best else None)
            close(rr['final_cpi'], best['cpi'] if best else c0)
            close(rr['gain_pct'], 100*(1-rr['final_cpi']/c0))
            assert rr['logical_candidate_calls'] == 2 and rr['final_area'] <= ctx['area_cap']+1e-9
        for k in search_set:
            assert labels[k]['phase_first_evaluated'] == 'search'
        candidate_dec, candidate_protect = [], []
        for k,c in enumerate(pool):
            if c['kind'] == 'single':
                continue
            u,v = c['parts']; inter = (cm[k]-cm[u]-cm[v]+c0)/c0
            if c['kind'] == 'up_up' and inter < -1e-8 and cm[k] < c0:
                candidate_dec.append((k, inter))
            if protected(cm, k) and cm[u] < c0:
                candidate_protect.append((k, inter))
        expected_witness = {}
        if candidate_dec:
            ids = [k for k,_ in candidate_dec]
            chosen = [k for k in d['selections']['mscs']['chosen'] if k in ids]
            expected_witness['decouple'] = chosen[0] if chosen else min(candidate_dec, key=lambda t:(t[1],t[0]))[0]
        if candidate_protect:
            expected_witness['protect'] = max(candidate_protect, key=lambda t:(t[1],-t[0]))[0]
        assert d['witnesses'] == expected_witness
        for kind,k in d['witnesses'].items():
            u,v = pool[k]['parts']; cp,cu,cv = [labels[i]['cpi'] for i in (k,u,v)]
            wr = result['witness_results'][kind]
            inter = (cp-cu-cv+c0)/c0; alone = (cv-c0)/c0; after = (cp-cu)/c0
            close(wr['actual_interaction'], inter); close(wr['actual_alone_cost'], alone)
            close(wr['actual_after_investment_cost'], after)
            verified = inter < -1e-8 and cp < c0 if kind == 'decouple' else 0 <= alone < TAU and after >= TAU and cu < c0
            assert wr['mechanism_verified'] == verified
            for name, cpred in (('scalar', sc), ('mscs', cm)):
                close(wr['predictions'][name]['interaction'], (cpred[k]-cpred[u]-cpred[v]+c0)/c0)
            if (arrays['root_m'][..., 0] > .01).all() and all(labels[i]['clamped_slice_stages'] == 0 for i in (k,u,v)):
                close(np.array(wr['actual_interaction_components']).sum(-1).mean(), inter*c0, atol=5e-6)
            audit_calls += 3
        assert len(d['witnesses']) <= 2
        assert len(labels) == sum(v for k,v in result['counts'].items() if k.endswith('physical_forward_calls'))
        report['roots'].append({'context': ctx['id'], 'predictions_replayed': True,
                                'choices_and_ablations_replayed': True, 'witnesses_replayed': True})
    assert search_calls == 40 and audit_calls <= 24
    report.update(logical_search_calls=search_calls, logical_audit_calls_upper_bound=audit_calls, sources_unchanged=True)
    return report
