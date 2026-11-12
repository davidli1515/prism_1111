"""Independent numpy replay of multisecant fits and method-local information."""
from pathlib import Path
import numpy as np
from secant_support import read,sha,config_key,compute_w_min,moves,ARMS,HISTORY
from matched_support import label_id
from adaptive_validate import prediction_replay,stage_cpi,softmax,close



class ResponseView:
    def __init__(self,arr,space):self.arr,self.prefix=arr,space+'__'
    def __getitem__(self,key):
        return self.arr[key] if key in ('root_m','root_cs','root_ell','stage_slope','logodds_slope') else self.arr[self.prefix+key]


def response_replay(arr,space,ctx,history_ids,width,out):
    lo=1/width;hi=min(100.,1/(.01*width+1e-8))
    close(arr['gram'],arr['gram_linear']**2,rtol=1e-12,atol=0)
    close(arr['cross'],arr['cross_linear']**2,rtol=1e-12,atol=0)
    # Each observed stack must have been purchased by this arm BEFORE this batch.
    observed=[]
    for lid in history_ids:
        with np.load(out/'truth'/ctx['id']/f'{lid}.npz') as old:
            m=old['m']
            observed.append(np.log(np.maximum(stage_cpi(m,width)-lo,1e-10)) if space=='scalar' else
                np.log(np.maximum(m[...,1:],1e-30))-np.log(np.maximum(m[...,:1],1e-30)))
    observed=np.stack(observed);close(observed,arr['history_response'])
    anchor=np.log(np.maximum(arr['root_cs']-lo,1e-10)) if space=='scalar' else arr['root_ell']
    residual=observed-anchor[None]-arr['history_slope'];close(residual,arr['residual'])
    eig,vec=np.linalg.eigh(arr['gram'])
    cutoff=1e-6*np.abs(eig).max(axis=-1,keepdims=True)
    inv=np.zeros_like(eig);mask=np.abs(eig)>cutoff;inv[mask]=1/eig[mask]
    inverse=np.einsum('nhi,ni,nki->nhk',vec,inv,vec)
    weights=np.einsum('nch,nhk->nck',arr['cross'],inverse)
    close(weights,arr['weights'],rtol=1e-4,atol=2e-5)
    # First validate transformed observations and residual construction above.
    # Then solve the recorded linear system, avoiding amplification of cross-library
    # log() rounding differences into a DIFFERENT right-hand side.
    rr=np.moveaxis(arr['residual'],0,1);flat=rr.reshape(rr.shape[0],rr.shape[1],-1)
    correction=np.einsum('nch,nhv->ncv',weights,flat).transpose(1,0,2).reshape(arr['correction'].shape)
    close(correction,arr['correction'],rtol=1e-4,atol=3e-6)
    if space=='scalar':
        excess=np.maximum(arr['root_cs']-lo,1e-10)
        stages=np.clip(lo+excess[None]*np.exp(np.clip(arr['stage_slope']/excess[None]+correction,-80,80)),lo,hi)
    else:
        ell=arr['root_ell'][None]+arr['logodds_slope']+correction
        mm=softmax(np.concatenate([np.zeros_like(ell[...,:1]),ell],-1))
        close(mm,arr['corrected_m'],rtol=1e-4,atol=3e-6);stages=stage_cpi(mm,width)
    close(stages,arr['corrected_stage_cpi'],rtol=1e-4,atol=3e-6)
    return stages.mean((1,2))


def validate(out):
    plan=read(out/'plan.json');summary=read(out/'summary.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    records=[];total=0
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
                assert d['candidates']==moves(current,radius,seen)
                assert d['history_ids']==history[-HISTORY:]
                arr=prediction_replay(out,d,'local',width);space=arm.split('_')[0]
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
                else:chosen=np.argsort(d['scores'],kind='stable')[:2].tolist()
                assert chosen==d['chosen']
                assert len(r['trials'])==2
                for i,t in zip(d['chosen'],r['trials']):
                    action=d['candidates'][i];assert all(t[k]==v for k,v in action.items())
                    lid=label_id(ctx['bench'],t['config']);assert t['label_id']==lid
                    assert config_key(t['config']) not in seen
                    with np.load(out/'truth'/ctx['id']/f'{lid}.npz') as z:close(stage_cpi(z['m'],width).mean(),t['cpi'])
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
            if arm=='scalar':
                old=read(Path(ctx['controls']['scalar_excess'])/'walks'/ctx['id']/'scalar_excess/summary.json')
                close(cp,old['final_cpi'],rtol=3e-6)
                assert q==old['counts']['candidate_forward_calls']
            if arm in ('scalar','mscs'):
                previous=read(Path(__file__).resolve().parent/'secant_development_s42_01/walks'/ctx['id']/arm/'summary.json')
                close(cp,previous['final_cpi'],rtol=3e-6)
            total+=q;records.append(dict(context=ctx['id'],arm=arm,queries=q,passed=True))
    assert total==summary['logical_queries']
    no_worse=True;strict=False;primary=[];control=[]
    for ctx in plan['contexts']:
        results={r['arm']:r for r in summary['walks'] if r['context']==ctx['id']}
        tol=1e-6*ctx['root_cpi'];a,b=results['hybrid_secant'],results['scalar_secant']
        no_worse &= all(r['resolved'] for r in results.values()) and a['final_cpi']<=b['final_cpi']+tol
        strict |= a['final_cpi']<b['final_cpi']-tol
        primary.append(a['gain_pct']);control.append(results['scalar']['gain_pct'])
    assert summary['gate_passed']==bool(no_worse and strict and np.mean(primary)>=np.mean(control))
    return dict(passed=True,method_local_history_replayed=True,scalar_controls_match_previous=True,
        numpy_multisecant_replayed=True,independent_numeric_rankings_match=True,validated_recorded_rhs_used=True,sources_unchanged=True,logical_queries=total,walks=records)


if __name__=='__main__':
    from secant_support import HERE
    from secant_support import save
    import json
    out=HERE/'quadratic_development_s42_01'
    result=validate(out)
    result['verifier_sha256']=sha(__file__)
    result['original_verifier_preserved']=sha(HERE/'quadratic_validate.py')
    save(out/'validation_replay_v2.json',result)
    print(json.dumps(result,indent=2))
