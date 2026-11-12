"""Independent numpy replay of multisecant fits and method-local information."""
from pathlib import Path
import numpy as np
from secant_support import read,sha,config_key,compute_w_min,moves,ARMS,HISTORY
from matched_support import label_id
from adaptive_validate import prediction_replay,stage_cpi,softmax,close


def validate(out):
    plan=read(out/'plan.json');summary=read(out/'summary.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    records=[];total=0
    for ctx in plan['contexts']:
        width=compute_w_min(ctx['root']);lo=1/width;hi=min(100.,1/(.01*width+1e-8))
        root_id=label_id(ctx['bench'],ctx['root'])
        for arm in ARMS:
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
                if arm.endswith('_secant'):
                    # Each observed stack must have been purchased by this arm BEFORE this batch.
                    observed=[]
                    for lid in d['history_ids']:
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
                    rr=np.moveaxis(residual,0,1);flat=rr.reshape(rr.shape[0],rr.shape[1],-1)
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
                    close(stages.mean((1,2)),d['scores'],rtol=1e-4,atol=3e-6)
                else:close(d['scores'],d[space+'_predicted_cpis'])
                arr.close()
                assert np.argsort(d['scores'],kind='stable')[:2].tolist()==d['chosen']
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
            total+=q;records.append(dict(context=ctx['id'],arm=arm,queries=q,passed=True))
    assert total==summary['logical_queries']
    return dict(passed=True,method_local_history_replayed=True,scalar_controls_match_previous=True,
        numpy_multisecant_replayed=True,sources_unchanged=True,logical_queries=total,walks=records)


if __name__=='__main__':
    from secant_support import HERE
    print(__import__('json').dumps(validate(HERE/'secant_development_s42_01'),indent=2))
