"""Replay priority demotion, area feasibility, labels and the fixed pilot gate."""
import numpy as np
from pathlib import Path
from secant_support import HERE,read,sha,compute_w_min
from matched_support import label_id
from width_priority_validate import cpi


def validate(out):
    plan=read(out/'plan.json');s=read(out/'summary.json');freeze=read(out/'freeze.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    assert plan['arms']==['scalar_raw','scalar_memory','scalar_veto','mscs_veto']
    expected=[(slot,b,step) for slot,src in enumerate(plan['source_directories'])
              for b in read(Path(src)/'plan.json')['benchmarks'] for step in range(6)]
    assert [(r['slot'],r['bench'],r['step']) for r in s['rows']]==expected and len(expected)==54
    assert len(freeze['decision_sha256'])==54
    queried={};logical=0;deltas=[];wins=losses=0
    for row in s['rows']:
        slot,b,step=row['slot'],row['bench'],row['step'];src=Path(plan['source_directories'][slot])
        path=out/'decisions'/f'{slot}__{b}__{step}.json';d=read(path)
        assert sha(path)==freeze['decision_sha256'][str(path)]
        oldpath=src/'walks'/b/'scalar_memory'/f'decision_{step:02}.json'
        assert str(oldpath)==d['source'] and sha(oldpath)==d['source_sha256']
        old=read(oldpath);pp=read(src/'plan.json');assert d['area_cap']==pp['area_cap']
        history=[pp['root']]+[read(oldpath.parent/f'round_{i:02}.json')['trial']['config'] for i in range(step)]
        assert [label_id(b,c) for c in history]==old['history_ids'] and history==old['center_configs']
        pool=old['candidates'];down=[any(a['config'][k]<v for k,v in old['root'].items()) for a in pool]
        assert down==d['withdrawal'];resolved=True
        for arm in plan['arms']:
            scores=old['scalar_scores'] if arm=='scalar_memory' else old['raw_scalar_scores']
            check=old['raw_mscs_scores'] if arm=='mscs_veto' else old['scalar_scores']
            blocked=[bool(arm.endswith('_veto') and w and c>=old['root_cpi']*(1-1e-6)) for w,c in zip(down,check)]
            assert blocked==d['blocked'][arm]
            order=sorted(range(len(pool)),key=lambda i:(blocked[i],scores[i],i));assert order==d['orders'][arm]
            # Increases cannot be demoted, and their mutual scalar order stays exact.
            assert not any(bb for bb,w in zip(blocked,down) if not w)
            prefix=d['area_prefixes'][arm];assert [p['index'] for p in prefix]==order[:len(prefix)]
            choice=None
            for j,p in enumerate(prefix):
                if p['area'] is None:assert p['error'] and j==len(prefix)-1;resolved=False;break
                if p['area']<=pp['area_cap']+1e-9:assert j==len(prefix)-1;choice=p['index'];break
            assert choice==d['choices'][arm];resolved &= choice is not None
        assert d['choices']['scalar_memory']==old['choice']
        assert resolved==d['resolved']==row['resolved']
        if not resolved:assert not row['arms'];continue
        for arm,rec in row['arms'].items():
            i=d['choices'][arm];config=pool[i]['config'];lid=label_id(b,config)
            assert rec['index']==i and rec['config']==config and rec['label_id']==lid
            assert lid not in old['history_ids'];assert rec['blocked_fallback']==d['blocked'][arm][i]
            with np.load(out/'truth'/b/f'{lid}.npz') as z:pred=float(cpi(z['m'],compute_w_min(config)).mean())
            np.testing.assert_allclose(pred,rec['cpi'],rtol=4e-6)
            np.testing.assert_allclose(max(0.,(old['root_cpi']-rec['cpi'])/old['root_cpi']),rec['gain'],atol=1e-12)
            logical+=1;queried[(b,lid)]=config
        arms=row['arms'];envelope=max(arms[k]['gain'] for k in ('scalar_raw','scalar_memory','scalar_veto'))
        delta=arms['mscs_veto']['gain']-envelope;deltas.append(delta);wins+=delta>1e-6;losses+=delta< -1e-6
    physical=reused=0
    for (b,lid),config in queried.items():
        rec=read(out/'truth'/b/f'{lid}.json');assert rec['config']==config
        if rec['origin']=='new_forward':physical+=1;assert rec['source'] is None
        else:
            assert rec['origin']=='old_label_after_freeze';reused+=1
            assert rec['source'] in [str(Path(src)/'truth'/b/f'{lid}.npz') for src in plan['source_directories']]
            with np.load(rec['source']) as a,np.load(out/'truth'/b/f'{lid}.npz') as z:np.testing.assert_array_equal(a['m'],z['m'])
    assert physical==s['new_candidate_forwards'] and reused==s['reused_old_labels'] and logical==s['logical_candidate_queries']
    assert s['new_ranking_model_forwards']==s['new_root_jacobians']==0
    gate=all(r['resolved'] for r in s['rows']) and wins>=2 and losses==0 and np.mean(deltas)>1e-6
    assert bool(gate)==s['pilot_gate_passed'] and wins==s['wins'] and losses==s['losses']
    np.testing.assert_allclose(np.mean(deltas),s['mean_advantage_vs_scalar_envelope'],atol=1e-12)
    return dict(passed=True,all54_private_contexts_replayed=True,scalar_memory_control_reproduced=True,
        only_withdrawals_demoted=True,all_choices_frozen_before_labels=True,areas_labels_and_budgets_replayed=True,
        sources_preserved=True)


if __name__=='__main__':
    import json
    print(json.dumps(validate(HERE/'reclaim_veto_s42_01'),indent=2))
