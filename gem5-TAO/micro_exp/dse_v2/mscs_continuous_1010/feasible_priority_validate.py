"""Replay every area-prefix stop and selected performance outcome independently."""
from pathlib import Path
import numpy as np
from secant_support import HERE,read,sha,config_key,compute_w_min
from matched_support import label_id
from adaptive_validate import close,stage_cpi


def validate(out):
    plan=read(out/'plan.json');summary=read(out/'summary.json');frozen=read(out/'frozen_choices.json');labels=read(out/'labels.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items());assert frozen['candidate_cpi_labels_observed']==0
    logical=0;required=set();counts=dict(win=0,tie=0,loss=0,unresolved=0)
    for ctx,rec,row in zip(plan['contexts'],frozen['contexts'],summary['rows']):
        assert ctx['id']==rec['context']==row['context'];d=read(rec['decision_file'])
        failures=read(HERE/'reclaim_supported_s42_01/roots'/ctx['id']/'decision.json')['unsupported_restrictions']
        excluded={config_key(r['action']['config']) for r in failures}
        assert excluded=={config_key(c) for c in rec['excluded']}
        allowed=[i for i,a in enumerate(d['candidates']) if config_key(a['config']) not in excluded]
        valid=True
        for arm in ('scalar','mscs'):
            order=sorted(allowed,key=lambda i:(d[arm+'_predicted_cpis'][i],i));prefix=rec['prefixes'][arm];choice=None
            assert [r['index'] for r in prefix]==order[:len(prefix)]
            for j,r in enumerate(prefix):
                assert r['config']==d['candidates'][r['index']]['config']
                if r['area'] is None:
                    assert j==len(prefix)-1 and r['error'];valid=False
                elif r['area']<=ctx['area_cap']+1e-9:
                    assert j==len(prefix)-1;choice=r['index']
            assert choice==rec['choices'][arm]
            if choice is None:valid=False
        assert valid==rec['resolved']==row['resolved']
        if not valid:counts['unresolved']+=1;continue
        accepted={}
        for arm,i in rec['choices'].items():
            lid=label_id(ctx['bench'],d['candidates'][i]['config']);required.add(lid);logical+=1;r=labels[lid]
            assert r['area']==rec['prefixes'][arm][-1]['area']<=ctx['area_cap']+1e-9
            cp=r['cpi'] if r['cpi']<ctx['root_cpi']*(1-1e-6) else ctx['root_cpi'];accepted[arm]=cp
            v=row['outcomes'][arm];assert v['resolved'] and v['best_cpi']==cp
            close(v['gain_pct'],100*(1-cp/ctx['root_cpi']))
        gap=(accepted['scalar']-accepted['mscs'])/ctx['root_cpi'];status='win' if gap>1e-6 else 'loss' if gap< -1e-6 else 'tie'
        close(gap,row['normalized_gap']);assert status==row['status'];counts[status]+=1
    assert required==set(labels) and logical==summary['logical_candidate_queries'] and counts==summary['counts']
    new=0;reused=0
    for lid,r in labels.items():
        assert lid==label_id(r['bench'],r['config'])
        with np.load(out/'labels'/f'{lid}.npz') as z:
            close(stage_cpi(z['m'],compute_w_min(r['config'])).mean(),r['cpi'])
            if r['origin']=='reused_prior_paid_label':
                reused+=1
                with np.load(r['source_file']) as old:np.testing.assert_array_equal(old['m'],z['m'])
            else:assert r['origin']=='new_forward';new+=1
    assert new==summary['new_candidate_forwards'] and reused==summary['reused_candidate_labels']
    return dict(passed=True,rank_prefixes_and_exact_area_stops_replayed=True,known_unsupported_configs_explicit=True,
        all_candidate_outcomes_replayed=True,sources_unchanged=True,new_candidate_forwards=new,reused_candidate_labels=reused,
        logical_candidate_queries=logical,scope='same CPI-query budget; area costs separately counted')


if __name__=='__main__':
    import json
    print(json.dumps(validate(HERE/'feasible_priority_s42_01'),indent=2))
