"""Full cold-start ensemble ablations; reuse labels only after frozen selection."""
import time
import sqlite3
import torch
import numpy as np
from quadratic_support import *
from matched_support import label_id,outcome
from run_adaptive import check_root_derivatives
import run_adaptive as area_runner

ARMS=('ensemble_scalar','ensemble_mscs')
SRC=HERE/'confirmation_s42_01'
OUT=HERE/'ensemble_walk_s42_01'


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    previous=read(SRC/'plan.json');assert read(SRC/'validation.json')['passed']
    sources=dict(previous['source_sha256'])
    for file in list(SRC.rglob('*.json'))+list(SRC.rglob('*.npz'))+list(HERE.glob('ensemble_walk*.py'))+[HERE/'ensemble_walk_protocol.txt']:
        sources[str(file)]=sha(file)
    assert all(sha(p)==h for p,h in sources.items())
    plan=dict(seed=42,contexts=previous['contexts'],arms=ARMS,history=HISTORY,steps=6,topk=2,
        source_sha256=sources,kernel='squared_inner_product',
        scope='two causal ablations on four already inspected roots; not heldout evidence')
    save(OUT/'plan.json',plan)
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    model=load_model();summaries=[];physical=0;root_forwards=0;total_features=0;feature_seconds=0.;reused=0
    for ctx in plan['contexts']:
        bench=ctx['bench'];cache=FeatureCache(Inputs(bench,ctx['root']['l1_size']))
        locals_={};truth={};root_id=label_id(bench,ctx['root']);width=compute_w_min(ctx['root'])
        def getlocal(c):
            nonlocal root_forwards
            key=config_key(c)
            if key not in locals_:
                locals_[key]=LocalModel(model,cache.get(c),width,'local');root_forwards+=1
            return locals_[key]
        initial=getlocal(ctx['root']);np.testing.assert_allclose(initial.cpi,ctx['root_cpi'],rtol=3e-6)
        array_save(OUT/'truth'/ctx['id']/f'{root_id}.npz',m=initial.m)
        truth[root_id]=dict(config=ctx['root'],cpi=initial.cpi,m=initial.m,area=ctx['root_area'])
        states={a:dict(root=ctx['root'],cpi=ctx['root_cpi'],area=ctx['root_area'],radius=4,
            seen={config_key(ctx['root'])},history=[root_id],queries=0,active=True) for a in ARMS}
        for step in range(6):
            frozen={}
            for arm,state in states.items():
                if not state['active']:continue
                local=getlocal(state['root']);np.testing.assert_allclose(local.cpi,state['cpi'],rtol=3e-6)
                cand=moves(state['root'],state['radius'],state['seen'])
                xs=torch.stack([cache.get(a['config']) for a in cand]);base=local.predict(xs)
                history=state['history'][-HISTORY:]
                arrays=dict(root_m=local.m,root_cs=local.cs,root_ell=local.ell,
                    stage_slope=base['stage_slope'],logodds_slope=base['logodds_slope'],predicted_m=base['mscs'])
                hx=torch.stack([cache.get(truth[k]['config']) for k in history])
                hy=torch.stack([response(truth[k]['m'],width,'scalar') for k in history])
                anchor,_,info=predict(local,xs,base,'scalar',hx,hy)
                arrays.update({'scalar__'+k:v for k,v in info.items()})
                scores=base[('scalar' if arm=='ensemble_scalar' else 'mscs')+'_cpi']
                first=int(torch.argmin(anchor))
                order=[first]+[i for i in np.argsort(scores.detach().numpy(),kind='stable').tolist() if i!=first][:1]
                prefix=Path('walks')/ctx['id']/arm;snap=prefix/f'predictions_{step:02}.npz'
                array_save(OUT/snap,**arrays)
                d=dict(step=step,arm=arm,root=state['root'],root_cpi=state['cpi'],root_area=state['area'],
                    radius=state['radius'],candidates=cand,chosen=order,history_ids=history,
                    scalar_predicted_cpis=base['scalar_cpi'].tolist(),mscs_predicted_cpis=base['mscs_cpi'].tolist(),
                    predicted_cpis=base['mscs_cpi'].tolist(),predicted_cpi_slopes=base['cpi_slope'].tolist(),
                    scores=scores.tolist(),prediction_file=str(snap),prediction_sha256=sha(OUT/snap),
                    phase='before ANY method current-round candidate labels')
                d['scalar_corrected_cpis']=anchor.tolist()
                save(OUT/prefix/f'decision_{step:02}.json',d)
                if step==0:
                    save(OUT/prefix/'derivative_check.json',check_root_derivatives(model,local,xs[0]-local.x))
                frozen[arm]=(d,xs)
            # All arms are now frozen. The shared truth cache is inaccessible to selectors except each own history.
            save(OUT/'batches'/ctx['id']/f'freeze_{step:02}.json',dict(
                decision_sha256={a:sha(OUT/'walks'/ctx['id']/a/f'decision_{step:02}.json') for a in frozen}))
            requests={}
            for arm,(d,xs) in frozen.items():
                for i in d['chosen']:
                    c=d['candidates'][i]['config'];lid=label_id(bench,c)
                    r=requests.setdefault(lid,dict(config=c,x=xs[i],roots=[]));r['roots'].append(d['root_cpi'])
            for lid,req in requests.items():
                if lid not in truth:
                    oldfile=SRC/'truth'/ctx['id']/f'{lid}.npz'
                    if oldfile.exists():
                        with np.load(oldfile) as z:mm=torch.from_numpy(z['m'].copy())
                        from core import cpis
                        cp=float(cpis(mm,width).mean());reused+=1;origin='reused_prior_paid_label'
                    else:
                        cp,mm=exact(model,req['x'],width);physical+=1;origin='new_forward'
                    truth[lid]=dict(config=req['config'],cpi=cp,m=mm,area=None,origin=origin)
                    array_save(OUT/'truth'/ctx['id']/f'{lid}.npz',m=mm)
                    save(OUT/'truth'/ctx['id']/f'{lid}.json',dict(config=req['config'],origin=origin,
                        source_file=str(oldfile) if oldfile.exists() else None))
                r=truth[lid]
                if r['area'] is None and any(r['cpi']<cp*(1-1e-6) for cp in req['roots']):
                    r['area']=area.get(r['config'])
            for arm,(d,_) in frozen.items():
                state=states[arm];trials=[]
                for i in d['chosen']:
                    action=d['candidates'][i];lid=label_id(bench,action['config']);r=truth[lid]
                    trials.append(dict(action,label_id=lid,cpi=r['cpi'],area=r['area'],score=d['scores'][i],
                        scalar_predicted_cpi=d['scalar_predicted_cpis'][i],mscs_predicted_cpi=d['mscs_predicted_cpis'][i],
                        area_error=area.store.errors.get(area.store.key(action['config'])) if r['area'] is None else None))
                    state['seen'].add(config_key(action['config']));state['history'].append(lid);state['queries']+=1
                value=outcome(state['cpi'],ctx['area_cap'],trials)
                best=min((t for t in trials if t['area'] is not None and t['area']<=ctx['area_cap']+1e-9
                    and t['cpi']<state['cpi']*(1-1e-6)),key=lambda t:t['cpi'],default=None)
                if not value['resolved']:
                    state['active']=False
                elif best is not None:
                    state.update(root=best['config'],cpi=best['cpi'],area=best['area'],radius=4)
                else:state['radius']=max(1,state['radius']//2)
                save(OUT/'walks'/ctx['id']/arm/f'round_{step:02}.json',dict(step=step,trials=trials,
                    outcome=value,accepted_label=None if best is None else best['label_id'],next_root=state['root'],
                    next_cpi=state['cpi'],next_area=state['area'],next_radius=state['radius'],
                    queries=state['queries'],resolved=value['resolved'],root_cpi=d['root_cpi'],
                    decision_sha256=sha(OUT/'walks'/ctx['id']/arm/f'decision_{step:02}.json')))
        for arm,s in states.items():
            result=dict(context=ctx['id'],arm=arm,final_cpi=s['cpi'],final_area=s['area'],final_config=s['root'],
                gain_pct=100*(1-s['cpi']/ctx['root_cpi']),logical_queries=s['queries'],resolved=s['active'])
            save(OUT/'walks'/ctx['id']/arm/'summary.json',result);summaries.append(result)
        total_features+=cache.builds;feature_seconds+=cache.seconds
        print(__import__('json').dumps(dict(context=ctx['id'],gains={a:100*(1-s['cpi']/ctx['root_cpi'])
            for a,s in states.items()},physical_candidate_forwards_so_far=physical)),flush=True)
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(walks=summaries,new_candidate_forwards=physical,reused_candidate_labels=reused,
        root_forwards=root_forwards,shared_backward_outputs=12*root_forwards,
        logical_queries=sum(r['logical_queries'] for r in summaries),area_counts=dict(area.counts),
        area_failures=area.store.n_fail,feature_builds=total_features,feature_seconds=feature_seconds,
        wall_seconds=time.perf_counter()-start))
    from ensemble_walk_validate import validate
    save(OUT/'validation.json',validate(OUT))


if __name__=='__main__':main()
