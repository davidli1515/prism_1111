"""Independent local selection/feedback replay; old numerical replay reused."""
from pathlib import Path
import numpy as np
from local_support import PARAMS,config_key,compute_w_min,moves,legal,COMPARISON_ARMS
from adaptive_validate import prediction_replay,close,sha,read


def validate(out):
    out=Path(out);plan=read(out/'plan.json');summary=read(out/'summary.json')
    for p,h in plan['source_sha256'].items():assert sha(p)==h,p
    checks=[];queries_total=0;no_worse=True;strict=False
    for ctx in plan['contexts']:
        base=out/'walks'/ctx['id']/'local';result=read(base/'summary.json')
        cur=ctx['root'];cp=ctx['root_cpi'];area=ctx['root_area'];width=compute_w_min(cur)
        radius=4;seen={config_key(cur)};roots=set();queries=0;accepted_count=0
        state=dict(completed_rounds=0,buckets={})
        files=sorted(base.glob('round_*.json'));assert len(files)==result['rounds']<=6
        for step,file in enumerate(files):
            r=read(file);d=read(out/r['decision_file'])
            assert sha(out/r['decision_file'])==r['decision_sha256']
            assert r['step']==d['step']==step and cur==r['root']==d['root']
            assert d['radius']==r['radius']==radius
            close(cp,d['root_cpi']);close(cp,r['root_cpi']);close(area,r['root_area']);close(area,d['root_area'])
            assert d['candidates']==moves(cur,radius=radius,seen=seen)
            arr=prediction_replay(out,d,'local',width)
            roots.add(config_key(cur))
            assert d['allocator_before']==state and d['allocation_mode']=='local_anchor'
            scores=[];models=[]
            for j,action in enumerate(d['candidates']):
                name=f"{action['parameter']}:{action['sign']:+d}"
                losses=state['buckets'].get(name,{}).get('loss2',dict(scalar=0,mscs=0))
                model='mscs' if losses['mscs']<losses['scalar'] else 'scalar'
                models.append(model);scores.append(d[model+'_predicted_cpis'][j])
            close(scores,d['local_ranking']['anchor_scores'],rtol=0,atol=0)
            assert models==d['local_ranking']['anchor_models']
            anchor=min(range(len(scores)),key=lambda j:(scores[j],j))
            m_order=sorted(range(len(scores)),key=lambda j:(d['mscs_predicted_cpis'][j],j))
            chosen=[anchor]+[j for j in m_order if j!=anchor][:1]
            sources=['local_'+models[anchor]+'_anchor']+['mscs_alternative']*(len(chosen)-1)
            assert chosen==d['chosen'] and sources==d['proposal_sources']
            assert len(chosen)==len(r['trials'])<=2
            if step==0:
                assert d['candidates']==ctx['initial_candidates']
                assert read(base/'derivative_check.json')['passed']
                path=Path(ctx['controls']['scalar_excess'])/'walks'/ctx['id']/'scalar_excess/predictions_00.npz'
                with np.load(path) as old:
                    close(arr['root_m'],old['root_m'])
                    close(arr['stage_slope'],old['stage_slope'],rtol=2e-4,atol=3e-6)
            groups={};accepted=[]
            for i,(j,t) in enumerate(zip(chosen,r['trials'])):
                action=d['candidates'][j];assert all(t[k]==v for k,v in action.items())
                c=t['config'];key=config_key(c)
                assert key not in seen and legal(c) and compute_w_min(c)==width
                assert [p for p in cur if c[p]!=cur[p]]==[t['parameter']]
                assert t['parameter'] in PARAMS and c['l1_size']==cur['l1_size'] and c['bp_type']==cur['bp_type']
                seen.add(key);queries+=1;assert t['query']==queries
                for field,series in (('predicted_cpi','predicted_cpis'),('scalar_predicted_cpi','scalar_predicted_cpis'),('mscs_predicted_cpi','mscs_predicted_cpis')):
                    close(t[field],d[series][j])
                assert t['proposed_by']==sources[i]
                close(t['actual_gain'],1-t['cpi']/cp);close(t['predicted_gain'],1-t['predicted_cpi']/cp)
                feasible=t['area'] is not None and t['area']<=ctx['area_cap']+1e-9
                assert feasible==t['feasible']
                if feasible and t['cpi']<cp*(1-1e-6):accepted.append(i)
                groups.setdefault(f"{t['parameter']}:{t['sign']:+d}",[]).append(i)
                e=t['explanation']
                for field in ('predicted_share_change','actual_share_change'):
                    assert np.shape(e[field])==(4,4);close(np.sum(e[field],axis=-1),0,atol=3e-7)
                for label in ('predicted','actual'):
                    values=np.asarray(e[label+'_pressure_cpi_change']);assert values.shape==(4,3)
                    if e['root_clamped_slices_stages']==0 and e[label+'_clamped_slices_stages']==0:
                        value=t['predicted_cpi'] if label=='predicted' else t['cpi']
                        close(values.sum(-1).mean(),value-cp,atol=5e-6)
            assert set(r['ranking_feedback'])==set(groups)
            for name,ids in groups.items():
                feedback=r['ranking_feedback'][name];assert feedback['trial_indices']==ids
                truth=[cp]+[r['trials'][i]['cpi'] for i in ids]
                predictions=[[cp]+[r['trials'][i][field] for i in ids]
                             for field in ('scalar_predicted_cpi','mscs_predicted_cpi')]
                tol=1e-6*cp;total=[0,0];comparisons=[]
                for i in range(len(truth)):
                    for j in range(i+1,len(truth)):
                        if abs(truth[i]-truth[j])<=tol:continue
                        actual_order=int(np.sign(truth[i]-truth[j]));orders=[];loss=[]
                        for values in predictions:
                            gap=values[i]-values[j];order=0 if abs(gap)<=tol else int(np.sign(gap))
                            orders.append(order);loss.append(1 if order==0 else 2*int(order!=actual_order))
                        total=[a+b for a,b in zip(total,loss)]
                        comparisons.append(dict(pair=[i,j],actual_order=actual_order,predicted_order=orders,loss2=loss))
                rr=feedback['feedback'];informative=sum(c['loss2'][0]!=c['loss2'][1] for c in comparisons)
                assert rr['comparisons']==comparisons and rr['loss2']==dict(zip(('scalar','mscs'),total))
                assert rr['discriminating_comparisons']==informative
                close(rr['tolerance'],tol,rtol=1e-12,atol=0)
                entry=state['buckets'].setdefault(name,dict(loss2={'scalar':0,'mscs':0},comparisons=0,
                    discriminating_comparisons=0,observations=0))
                for label,value in zip(('scalar','mscs'),total):entry['loss2'][label]+=value
                entry['comparisons']+=len(comparisons);entry['discriminating_comparisons']+=informative
                entry['observations']+=len(ids)
            state['completed_rounds']+=1;assert r['allocator_after']==state
            best=min(accepted,key=lambda i:r['trials'][i]['cpi']) if accepted else None
            assert best==r['accepted_trial']
            assert [t['accepted'] for t in r['trials']]==[i==best for i in range(len(chosen))]
            if best is not None:
                t=r['trials'][best];cur=t['config'];cp=t['cpi'];area=t['area'];radius=4;accepted_count+=1
            else:radius=max(1,radius//2)
            assert cur==r['next_root'];close(cp,r['next_cpi']);close(area,r['next_area'])
        assert queries==result['counts']['candidate_forward_calls']==result['counts']['area_requests']<=12
        assert len(roots)==result['counts']['root_forward_calls']
        assert result['counts']['backward_scalar_outputs']==12*len(roots)
        assert result['seen_configurations']==len(seen) and result['accepted_steps']==accepted_count
        assert result['allocator_final']==state and result['final_config']==cur
        close(cp,result['final_cpi']);close(area,result['final_area']);assert area<=ctx['area_cap']+1e-9
        close(1-cp/ctx['root_cpi'],result['gain'])
        pair=next(p for p in summary['paired'] if p['context']==ctx['id'])
        for arm in COMPARISON_ARMS:
            saved=result if arm=='local' else read(Path(ctx['controls'][arm])/'walks'/ctx['id']/arm/'summary.json')
            close(pair['gain_pct'][arm],100*saved['gain']);close(pair['final_cpi'][arm],saved['final_cpi'])
        tol=1e-6*ctx['root_cpi']
        for arm in ('hybrid','adaptive'):
            no_worse &= cp<=pair['final_cpi'][arm]+tol;strict |= cp<pair['final_cpi'][arm]-tol
        queries_total+=queries;checks.append(dict(context=ctx['id'],queries=queries,rounds=len(files),passed=True))
    assert queries_total<=plan['max_candidate_verifications']
    assert queries_total==sum(w['counts']['candidate_forward_calls'] for w in summary['walks'])
    assert summary['gate_passed']==bool(no_worse and strict)
    return dict(passed=True,sources_unchanged=True,walks=checks,total_candidate_queries=queries_total,
                local_feedback_and_selection_replayed=True,extension_gate_replayed=True)
