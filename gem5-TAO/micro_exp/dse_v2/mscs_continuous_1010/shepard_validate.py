"""Independent multi-center response and private accepted-root history replay."""
from pathlib import Path
import numpy as np
from secant_support import HERE,read,sha,config_key,compute_w_min,moves,HISTORY
from matched_support import label_id
from adaptive_validate import prediction_replay,stage_cpi,softmax,close
from quadratic_validate_replay_v2 import ResponseView


def response_replay(z,space,ctx,ids,width,out):
    observed=[];lo=1/width;hi=min(100.,1/(.01*width+1e-8))
    for lid in ids:
        with np.load(out/'truth'/ctx['id']/f'{lid}.npz') as old:
            m=old['m'];value=np.log(np.maximum(stage_cpi(m,width)-lo,1e-10)) if space=='scalar' else np.log(m[...,1:])-np.log(m[...,:1])
            observed.append(value)
    close(observed,z['center_response'],rtol=1e-4,atol=3e-6)
    distances=z['distance_squared'];assert (distances>=0).all() and len(distances)==len(ids)
    zeros=distances==0;safe=np.maximum(distances,np.finfo(distances.dtype).tiny)
    nearest=safe.min(0,keepdims=True);ratios=nearest/safe
    weights=ratios/ratios.sum(0,keepdims=True)
    exact=zeros/np.maximum(1,zeros.sum(0,keepdims=True));weights=np.where(zeros.any(0,keepdims=True),exact,weights)
    close(weights,z['weights'],rtol=1e-12,atol=1e-12)
    models=z['center_response'][:,None]+z['center_slope']
    value=(weights.reshape(*weights.shape,*([1]*(models.ndim-weights.ndim)))*models).sum(0).astype(np.float32)
    close(value,z['blended_response'],rtol=1e-5,atol=3e-6)
    if space=='scalar':
        ex=np.maximum(z['root_cs']-lo,1e-10)
        ratio=z['stage_slope']/ex[None] if len(ids)==1 else value-z['center_response'][-1][None]
        stages=np.clip(lo+ex[None]*np.exp(np.clip(ratio,-80,80)),lo,hi)
    else:
        if len(ids)==1:mm=z['predicted_m']
        else:mm=softmax(np.concatenate([np.zeros_like(value[...,:1]),value],-1))
        close(mm,z['blended_m'],rtol=1e-4,atol=3e-6);stages=stage_cpi(mm,width)
    close(stages,z['blended_stage_cpi'],rtol=1e-4,atol=3e-6)
    return stages.mean((1,2))


class View(ResponseView):
    def __getitem__(self,key):
        return self.arr[key] if key in ('root_m','root_cs','root_ell','stage_slope','logodds_slope','predicted_m') else self.arr[self.prefix+key]


def validate(out):
    plan=read(out/'plan.json');summary=read(out/'summary.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    total=0;physical=set();results=[];records=[]
    for ctx in plan['contexts']:
        width=compute_w_min(ctx['root']);root_id=label_id(ctx['bench'],ctx['root'])
        for arm in plan['arms']:
            path=out/'walks'/ctx['id']/arm;current=ctx['root'];cp=ctx['root_cpi'];area=ctx['root_area'];radius=4
            seen={config_key(current)};history=[root_id];centers=[root_id];q=0
            assert read(path/'derivative_check.json')['passed']
            for file in sorted(path.glob('decision_*.json')):
                d=read(file);step=d['step'];r=read(path/f'round_{step:02}.json');cur_id=label_id(ctx['bench'],current)
                if centers[-1]!=cur_id:centers.append(cur_id)
                assert d['root']==current and d['radius']==radius;close(d['root_cpi'],cp)
                assert d['history_ids']==history[-HISTORY:] and d['center_ids']==centers[-HISTORY:]
                assert all(lid in history for lid in d['center_ids'])
                assert d['candidates']==moves(current,radius,seen)
                assert sha(file)==r['decision_sha256']==read(out/'batches'/ctx['id']/f'freeze_{step:02}.json')['decision_sha256'][arm]
                z=prediction_replay(out,d,'local',width)
                if arm=='hybrid_shepard':
                    scores={}
                    for space in ('scalar','mscs'):
                        scores[space]=response_replay(View(z,space),space,ctx,d['center_ids'],width,out)
                        close(scores[space],d[space+'_blended_cpis'],rtol=1e-4,atol=3e-6)
                    first=int(np.argmin(scores['scalar']));chosen=[first]+[i for i in np.argsort(scores['mscs'],kind='stable').tolist() if i!=first][:1]
                else:
                    scores=response_replay(z,arm.split('_')[0],ctx,d['center_ids'],width,out)
                    close(scores,d['scores'],rtol=1e-4,atol=3e-6);chosen=np.argsort(scores,kind='stable')[:2].tolist()
                assert chosen==d['chosen'];z.close()
                assert len(r['trials'])==2
                for i,t in zip(d['chosen'],r['trials']):
                    action=d['candidates'][i];assert all(t[k]==v for k,v in action.items())
                    lid=label_id(ctx['bench'],t['config']);assert lid==t['label_id'] and config_key(t['config']) not in seen
                    with np.load(out/'truth'/ctx['id']/f'{lid}.npz') as a:close(stage_cpi(a['m'],width).mean(),t['cpi'])
                    physical.add((ctx['id'],lid));seen.add(config_key(t['config']));history.append(lid);q+=1
                valid=[t for t in r['trials'] if t['area'] is not None and t['area']<=ctx['area_cap']+1e-9 and t['cpi']<cp*(1-1e-6)]
                best=min(valid,key=lambda t:t['cpi'],default=None)
                unknown=[t['cpi'] for t in r['trials'] if t['area'] is None and t['cpi']<cp*(1-1e-6)]
                high=cp if best is None else best['cpi'];low=min([high]+unknown)
                assert r['outcome']['cpi_lower']==low and r['outcome']['cpi_upper']==high
                assert r['resolved']==(low==high) and r['queries']==q
                if low==high:
                    if best is None:radius=max(1,radius//2)
                    else:current,cp,area,radius=best['config'],best['cpi'],best['area'],4
                assert r['next_root']==current and r['next_radius']==radius;close(cp,r['next_cpi'])
            result=read(path/'summary.json');assert q==result['logical_queries']<=12
            assert current==result['final_config'];close(cp,result['final_cpi']);close(result['gain_pct'],100*(1-cp/ctx['root_cpi']))
            results.append(result);total+=q;records.append(dict(context=ctx['id'],arm=arm,queries=q,passed=True))
    reused=0
    for ctxid,lid in physical:
        meta=read(out/'truth'/ctxid/f'{lid}.json')
        if meta['origin']=='reused_prior_paid_label':
            reused+=1
            with np.load(meta['source_file']) as a,np.load(out/'truth'/ctxid/f'{lid}.npz') as b:np.testing.assert_array_equal(a['m'],b['m'])
        else:assert meta['origin']=='new_forward'
    assert reused==summary['reused_candidate_labels'] and len(physical)-reused==summary['new_candidate_forwards']
    assert total==summary['logical_queries'] and results==summary['walks']
    old=read(HERE/'confirmation_s42_01/summary.json');wins=0;other=0;gaps=[]
    for ctx,row in zip(plan['contexts'],summary['gate_rows']):
        vals=[r['final_cpi'] for r in old['walks'] if r['context']==ctx['id'] and r['arm'] in ('scalar','scalar_secant')]
        vals += [r['final_cpi'] for r in results if r['context']==ctx['id'] and r['arm']=='scalar_shepard']
        m=next(r for r in results if r['context']==ctx['id'] and r['arm']=='hybrid_shepard')
        gap=(min(vals)-m['final_cpi'])/ctx['root_cpi'];gaps.append(gap);wins+=gap>1e-6;other+=gap>1e-6 and ctx['bench']!='602.gcc_s'
        close(gap,row['normalized_cpi_advantage']);assert row['status']==('win' if gap>1e-6 else 'loss' if gap< -1e-6 else 'tie')
    assert summary['gate_passed']==bool(all(r['resolved'] for r in results) and wins>=2 and other>=1 and np.mean(gaps)>1e-6)
    return dict(passed=True,sources_unchanged=True,independent_weights_and_responses=True,numeric_rankings_match=True,
        private_accepted_root_centers=True,no_rejected_or_other_arm_derivatives=True,
        logical_queries=total,reused_labels=reused,new_candidate_forwards=len(physical)-reused,gate_replayed=True,walks=records)


if __name__=='__main__':
    import json
    print(json.dumps(validate(HERE/'shepard_development_s42_01'),indent=2))
