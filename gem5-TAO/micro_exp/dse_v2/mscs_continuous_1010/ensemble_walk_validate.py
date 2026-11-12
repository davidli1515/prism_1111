"""Independent root selection, numeric ranking, local feedback and budget replay."""
from pathlib import Path
import numpy as np
from secant_support import read,sha,config_key,compute_w_min,moves,ARMS,HISTORY
from matched_support import label_id
from adaptive_validate import prediction_replay,stage_cpi,softmax,close

from quadratic_validate_replay_v2 import ResponseView,response_replay


def validate(out):
    plan=read(out/'plan.json');summary=read(out/'summary.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    records=[];total=0;all_results=[];physical_ids=set()
    for ctx in plan['contexts']:
        width=compute_w_min(ctx['root']);lo=1/width;hi=min(100.,1/(.01*width+1e-8))
        root_id=label_id(ctx['bench'],ctx['root'])
        for arm in plan['arms']:
            path=out/'walks'/ctx['id']/arm;current=ctx['root'];cp=ctx['root_cpi'];area=ctx['root_area'];radius=4
            seen={config_key(current)};history=[root_id];q=0
            decisions=sorted(path.glob('decision_*.json'))
            assert read(path/'derivative_check.json')['passed']
            for file in decisions:
                d=read(file);step=d['step'];r=read(path/f'round_{step:02}.json')
                assert d['root']==current and d['radius']==radius;close(d['root_cpi'],cp)
                assert sha(file)==r['decision_sha256']==read(out/'batches'/ctx['id']/f'freeze_{step:02}.json')['decision_sha256'][arm]
                assert d['phase']=='before ANY method current-round candidate labels'
                assert d['candidates']==moves(current,radius,seen)
                assert d['history_ids']==history[-HISTORY:]
                arr=prediction_replay(out,d,'local',width)
                anchor=response_replay(ResponseView(arr,'scalar'),'scalar',ctx,d['history_ids'],width,out)
                close(anchor,d['scalar_corrected_cpis'],rtol=1e-4,atol=3e-6)
                kind='scalar' if arm=='ensemble_scalar' else 'mscs'
                close(d['scores'],d[kind+'_predicted_cpis']);arr.close()
                first=int(np.argmin(anchor));chosen=[first]+[i for i in np.argsort(d['scores'],kind='stable').tolist() if i!=first][:1]
                assert chosen==d['chosen']
                assert len(r['trials'])==2
                for i,t in zip(d['chosen'],r['trials']):
                    action=d['candidates'][i];assert all(t[k]==v for k,v in action.items())
                    lid=label_id(ctx['bench'],t['config']);assert t['label_id']==lid
                    assert config_key(t['config']) not in seen
                    with np.load(out/'truth'/ctx['id']/f'{lid}.npz') as z:close(stage_cpi(z['m'],width).mean(),t['cpi'])
                    physical_ids.add((ctx['id'],lid))
                    close(t['scalar_predicted_cpi'],d['scalar_predicted_cpis'][i]);close(t['mscs_predicted_cpi'],d['mscs_predicted_cpis'][i])
                    seen.add(config_key(t['config']));history.append(lid);q+=1
                valid=[t for t in r['trials'] if t['area'] is not None and t['area']<=ctx['area_cap']+1e-9 and t['cpi']<cp*(1-1e-6)]
                best=min(valid,key=lambda t:t['cpi'],default=None)
                unknown=[t for t in r['trials'] if t['area'] is None and t['cpi']<cp*(1-1e-6)]
                high=cp if best is None else best['cpi'];low=min([high]+[t['cpi'] for t in unknown])
                assert r['outcome']['cpi_lower']==low and r['outcome']['cpi_upper']==high
                assert r['resolved']==(low==high) and r['queries']==q
                if low!=high:
                    assert file==decisions[-1]
                elif best is None:radius=max(1,radius//2)
                else:current,cp,area,radius=best['config'],best['cpi'],best['area'],4
                assert r['next_root']==current and r['next_radius']==radius;close(cp,r['next_cpi'])
            result=read(path/'summary.json');assert q==result['logical_queries']<=12
            assert current==result['final_config'];close(cp,result['final_cpi'])
            close(result['gain_pct'],100*(1-cp/ctx['root_cpi']))
            all_results.append(result)
            total+=q;records.append(dict(context=ctx['id'],arm=arm,queries=q,passed=True))
    assert total==summary['logical_queries']
    assert len(physical_ids)==summary['new_candidate_forwards']+summary['reused_candidate_labels']
    reused=0
    for ctxid,lid in physical_ids:
        meta=read(out/'truth'/ctxid/f'{lid}.json')
        if meta['origin']=='reused_prior_paid_label':
            reused+=1
            with np.load(meta['source_file']) as a, np.load(out/'truth'/ctxid/f'{lid}.npz') as b:np.testing.assert_array_equal(a['m'],b['m'])
        else:assert meta['origin']=='new_forward'
    assert reused==summary['reused_candidate_labels']
    assert all_results==summary['walks']
    return dict(passed=True,method_local_history_replayed=True,numeric_rankings_match=True,
        scalar_correction_independently_replayed=True,labels_reused_only_after_choice=True,
        reused_labels=reused,new_candidate_forwards=summary['new_candidate_forwards'],
        sources_unchanged=True,logical_queries=total,walks=records)



if __name__=='__main__':
    import json
    from secant_support import HERE
    print(json.dumps(validate(HERE/'ensemble_walk_s42_01'),indent=2))
