"""Independent root selection, numeric ranking, local feedback and budget replay."""
from pathlib import Path
import numpy as np
from secant_support import read,sha,config_key,compute_w_min,moves,ARMS,HISTORY
from matched_support import label_id
from adaptive_validate import prediction_replay,stage_cpi,softmax,close

from quadratic_validate_replay_v2 import ResponseView,response_replay
from validate_disagreement import inventory_again
from validate_matched import independent_pairs,advance_feedback
from fresh_disagreement_support import legal


def selection_replay(out):
    rp=read(out/'root_plan.json');ex=read(out/'exclusions.json')
    assert sha(out/'exclusions.json')==rp['exclusions_sha256']
    found,unknown=inventory_again(ex['source_sha256'])
    benches=('602.gcc_s','605.mcf_s','623.xalancbmk_s','641.leela_s')
    found.update((b,c) for b in benches for c in unknown)
    assert found=={(r['bench'],config_key(r['config'])) for r in ex['exposed']}
    assert unknown=={config_key(c) for c in ex['unknown_context_configs']}
    expected=[]
    assert [r['bench'] for r in rp['selection']]==list(benches)
    for row in rp['selection']:
        archive=read(row['archive']);points=np.c_[archive['cpis'],archive['areas']]
        front=[]
        for i,v in enumerate(points):
            if not (np.all(points<=v,axis=1)&np.any(points<v,axis=1)).any() and not np.any(np.all(points[:i]==v,axis=1)):
                front.append(i)
        front=sorted(front,key=lambda i:points[i,1])
        target=(len(front)-1)//2
        eligible=[legal(archive['configs'][i]) and (row['bench'],config_key(archive['configs'][i])) not in found for i in front]
        rank=min((j for j,x in enumerate(eligible) if x),key=lambda j:(abs(j-target),j),default=None)
        assert row['front']==front and row['target']==target and row['eligible']==eligible and row['selected_rank']==rank
        if rank is None:continue
        c=rp['contexts'][len(expected)];idx=front[rank]
        assert c['bench']==row['bench'] and c['root']==archive['configs'][idx] and c['archive_index']==idx and c['area_rank']==rank
        close(c['root_cpi'],archive['cpis'][idx]);close(c['root_area'],archive['areas'][idx]);close(c['area_cap'],1.10*c['root_area'])
        expected.append(c)
    assert expected==rp['contexts']
    return expected


def validate(out):
    plan=read(out/'plan.json');summary=read(out/'summary.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    assert plan['contexts']==selection_replay(out)
    records=[];total=0;all_results=[];physical_ids=set()
    for ctx in plan['contexts']:
        width=compute_w_min(ctx['root']);lo=1/width;hi=min(100.,1/(.01*width+1e-8))
        root_id=label_id(ctx['bench'],ctx['root'])
        for arm in plan['arms']:
            path=out/'walks'/ctx['id']/arm;current=ctx['root'];cp=ctx['root_cpi'];area=ctx['root_area'];radius=4
            seen={config_key(current)};history=[root_id];q=0
            allocator=dict(completed_rounds=0,buckets={})
            decisions=sorted(path.glob('decision_*.json'))
            assert read(path/'derivative_check.json')['passed']
            for file in decisions:
                d=read(file);step=d['step'];r=read(path/f'round_{step:02}.json')
                assert d['root']==current and d['radius']==radius;close(d['root_cpi'],cp)
                assert sha(file)==r['decision_sha256']==read(out/'batches'/ctx['id']/f'freeze_{step:02}.json')['decision_sha256'][arm]
                assert d['phase']=='before ANY method current-round candidate labels'
                assert d['candidates']==moves(current,radius,seen)
                assert d['history_ids']==history[-HISTORY:]
                arr=prediction_replay(out,d,'local',width);space='mscs' if arm=='local' else arm.split('_')[0]
                if arm=='hybrid_secant':
                    for ss in ('scalar','mscs'):
                        predicted=response_replay(ResponseView(arr,ss),ss,ctx,d['history_ids'],width,out)
                        close(predicted,d[ss+'_corrected_cpis'],rtol=1e-4,atol=3e-6)
                        if ss=='scalar':independent_scalar_scores=predicted
                        else:independent_mscs_scores=predicted
                    close(d['scores'],d['mscs_corrected_cpis'])
                elif arm.endswith('_secant'):
                    predicted=response_replay(arr,space,ctx,d['history_ids'],width,out)
                    close(predicted,d['scores'],rtol=1e-4,atol=3e-6)
                    assert np.argsort(predicted,kind='stable')[:2].tolist()==d['chosen']
                else:close(d['scores'],d[space+'_predicted_cpis'])
                arr.close()
                if arm=='hybrid_secant':
                    first=int(np.argmin(independent_scalar_scores))
                    independent=[first]+[i for i in np.argsort(independent_mscs_scores,kind='stable').tolist() if i!=first][:1]
                    assert independent==d['chosen']
                    first=int(np.argmin(d['scalar_corrected_cpis']))
                    chosen=[first]+[i for i in np.argsort(d['mscs_corrected_cpis'],kind='stable').tolist() if i!=first][:1]
                elif arm=='local':
                    assert d['allocator_before']==allocator
                    chosen=independent_pairs(d,list(range(len(d['candidates']))))['local']
                else:chosen=np.argsort(d['scores'],kind='stable')[:2].tolist()
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
                if arm=='local':
                    advance_feedback(allocator,r);assert allocator==r['allocator_after']
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
    assert len(physical_ids)==summary['new_candidate_forwards']
    assert all_results==summary['walks']
    for candidate in ('hybrid_secant','local'):
        gate=summary['gates'][candidate];differences=[];wins=0;otherwins=0;resolved=True
        for row,ctx in zip(gate['rows'],plan['contexts']):
            rr={r['arm']:r for r in all_results if r['context']==ctx['id']}
            diff=(min(rr[a]['final_cpi'] for a in ('scalar','scalar_secant'))-rr[candidate]['final_cpi'])/ctx['root_cpi']
            differences.append(diff);wins+=diff>1e-6;otherwins+=diff>1e-6 and ctx['bench']!='602.gcc_s'
            resolved &= all(r['resolved'] for r in rr.values())
            assert row['status']==('win' if diff>1e-6 else 'loss' if diff< -1e-6 else 'tie')
            close(row['normalized_cpi_advantage'],diff)
        assert gate['wins']==wins and gate['non_gcc_wins']==otherwins
        close(gate['mean_normalized_cpi_advantage'],np.mean(differences))
        assert gate['passed']==bool(len(differences)==4 and resolved and wins>=2 and otherwins>=1 and np.mean(differences)>1e-6)
    assert summary['gate_passed']==summary['gates']['hybrid_secant']['passed']
    return dict(passed=True,method_local_history_replayed=True,local_feedback_replayed=True,
        numeric_rankings_match=True,validated_recorded_rhs_used=True,sources_unchanged=True,
        root_selection_replayed=True,all_gates_replayed=True,logical_queries=total,walks=records)



if __name__=='__main__':
    import json
    from secant_support import HERE
    print(json.dumps(validate(HERE/'confirmation_s42_01'),indent=2))
