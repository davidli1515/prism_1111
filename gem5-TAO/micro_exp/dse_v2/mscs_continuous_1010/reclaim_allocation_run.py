"""One area-reclaim decision under the same predicted and measured CPI cap."""
import time
import sqlite3
import numpy as np
import torch
from quadratic_support import *
from core import PARAMS,legal
from matched_support import label_id
from joint_allocation_support import objective
from reclaim_allocation_support import allocation,round_configs,area_estimate,outcome,TOLERANCE
from run_adaptive import check_root_derivatives
import run_adaptive as area_runner

SRC=HERE/'confirmation_s42_01';OUT=HERE/'reclaim_allocation_s42_01'


def gradient_check(kind,local,stage,odds):
    n=len(stage);w=np.full(n,.37);args=(kind,local.cs.numpy().astype(float),local.ell.numpy().astype(float),stage,odds,local.width)
    value,analytic=objective(w,*args);numeric=[]
    for i in range(n):
        plus=w.copy();minus=w.copy();plus[i]+=1e-6;minus[i]-=1e-6
        numeric.append((objective(plus,*args)[0]-objective(minus,*args)[0])/2e-6)
    np.testing.assert_allclose(analytic,numeric,rtol=2e-4,atol=2e-7)
    return dict(passed=True,at=w.tolist(),analytic=analytic.tolist(),numeric=numeric,
                max_abs_error=float(np.max(np.abs(analytic-numeric))))


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    sourceplan=read(SRC/'plan.json');sources=dict(sourceplan['source_sha256'])
    for p in list(SRC.rglob('*.json'))+list(SRC.rglob('*.npz'))+list(HERE.glob('reclaim_allocation*.py'))+[HERE/'reclaim_allocation_protocol.txt']:
        sources[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in sources.items())
    for p in HERE.glob('joint_allocation*.py'):sources[str(p)]=sha(p)
    contexts=[{k:v for k,v in c.items() if k!='area_cap'} for c in sourceplan['contexts']]
    for c in contexts:c['cpi_cap']=c['root_cpi']*(1+TOLERANCE)
    plan=dict(contexts=contexts,arms=['scalar','mscs'],seed=42,source_sha256=sources,
              scope='four already inspected roots; one greedy proposal-generation comparison, not heldout')
    save(OUT/'plan.json',plan)
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    model=load_model();requests={};decisions=[];feature_builds=0;feature_seconds=0.;basis_area_requests=0
    for ctx in plan['contexts']:
        c=ctx['root'];width=compute_w_min(c);cache=FeatureCache(Inputs(ctx['bench'],c['l1_size']))
        local=LocalModel(model,cache.get(c),width,'local')
        np.testing.assert_allclose(local.cpi,ctx['root_cpi'],rtol=3e-6)
        measured=area.get(c);basis_area_requests+=1
        assert measured is not None;np.testing.assert_allclose(measured,ctx['root_area'],rtol=1e-5,atol=1e-4)
        singles=[a for a in moves(c,4,{config_key(c)}) if a['sign']==-1];xs=torch.stack([cache.get(a['config']) for a in singles]);base=local.predict(xs)
        basis=[];basis_indices=[];basis_areas=[]
        for param in PARAMS:
            ids=[i for i,a in enumerate(singles) if a['parameter']==param and a['sign']==-1]
            if not ids:continue
            i=min(ids,key=lambda j:singles[j]['config'][param]);basis.append(singles[i]);basis_indices.append(i)
            aa=area.get(singles[i]['config']);basis_area_requests+=1
            assert aa is not None,'Area basis failed; stop without treating unknown area as infeasible.'
            basis_areas.append(aa)
        delta_area=np.array(basis_areas)-ctx['root_area'];assert basis
        single_areas=[area_estimate(c,basis,delta_area,ctx['root_area'],a['config'])[0] for a in singles]
        def safe_single_indices(kind):
            return [i for i,v in enumerate(base[kind+'_cpi']) if float(v)<=ctx['cpi_cap'] and single_areas[i]<ctx['root_area']]
        safe=safe_single_indices('scalar')
        anchor_id=min(safe,key=lambda i:(single_areas[i],i)) if safe else int(torch.argmin(base['scalar_cpi']))
        anchor=singles[anchor_id]
        stage=base['stage_slope'][basis_indices].numpy().astype(float)
        odds=base['logodds_slope'][basis_indices].numpy().astype(float)
        checks={k:gradient_check(k,local,stage,odds) for k in ('scalar','mscs')}
        gs=objective(np.zeros(len(basis)),'scalar',local.cs.numpy().astype(float),local.ell.numpy().astype(float),stage,odds,width)[1]
        gm=objective(np.zeros(len(basis)),'mscs',local.cs.numpy().astype(float),local.ell.numpy().astype(float),stage,odds,width)[1]
        np.testing.assert_allclose(gs,gm,rtol=4e-4,atol=2e-7)
        checks['root_jvp']=check_root_derivatives(model,local,xs[basis_indices[0]]-local.x)
        proposals={};details={};prefix=Path('roots')/ctx['id']
        for kind in ('scalar','mscs'):
            sol=allocation(kind,c,basis,delta_area,ctx['root_area'],local.cs.numpy().astype(float),local.ell.numpy().astype(float),stage,odds,width)
            rounded=round_configs(c,basis,sol['fractions'],delta_area,ctx['root_area'])
            for record in rounded:
                x=cache.get(record['config']);pr=local.predict(x[None]);weights=torch.tensor(record['fractions'],dtype=xs.dtype)
                chord=torch.einsum('r,rnd->nd',weights,xs[basis_indices]-local.x[None])
                actual=x-local.x
                record.update(scalar_predicted_cpi=float(pr['scalar_cpi'][0]),mscs_predicted_cpi=float(pr['mscs_cpi'][0]),
                    feature_chord_error_norm=float(torch.linalg.norm(actual-chord)),
                    actual_displacement_norm=float(torch.linalg.norm(actual)))
            for record in rounded:record['predicted_safe']=record[kind+'_predicted_cpi']<=ctx['cpi_cap']
            choices=[r for r in rounded if r['config']!=anchor['config'] and r['predicted_safe']]
            if choices:
                selected=min(choices,key=lambda r:r['predicted_area']);proposal=dict(config=selected['config'],kind='joint_rounded',rounding=selected['rounding'])
            else:
                eligible=[i for i in safe_single_indices(kind) if i!=anchor_id]
                index=min(eligible,key=lambda i:(single_areas[i],i)) if eligible else next(i for i in np.argsort(base[kind+'_cpi'].numpy(),kind='stable') if i!=anchor_id)
                proposal=dict(config=singles[index]['config'],kind='single_fallback',action=singles[index],unsafe_fallback=not bool(eligible))
            proposals[kind]=proposal;details[kind]=dict(solution=sol,rounded=rounded)
        pool=[dict(config=anchor['config'],kind='shared_scalar_anchor')]
        for p in proposals.values():
            if not any(v['config']==p['config'] for v in pool):pool.append(p)
        poolxs=torch.stack([cache.get(p['config']) for p in pool]);pred=local.predict(poolxs)
        snap=prefix/'predictions.npz'
        array_save(OUT/snap,root_m=local.m,root_cs=local.cs,root_ell=local.ell,stage_slope=pred['stage_slope'],
            logodds_slope=pred['logodds_slope'],predicted_m=pred['mscs'],basis_stage_slope=stage,basis_logodds_slope=odds,
            single_stage_slope=base['stage_slope'],single_logodds_slope=base['logodds_slope'],single_predicted_m=base['mscs'])
        chosen={k:[0,next(i for i,v in enumerate(pool) if v['config']==p['config'])] for k,p in proposals.items()}
        assert all(len(set(ids))==2 for ids in chosen.values())
        d=dict(context=ctx['id'],root=c,root_cpi=local.cpi,root_area=ctx['root_area'],cpi_cap=ctx['cpi_cap'],
            singles=singles,single_predicted_areas=single_areas,anchor_index=anchor_id,anchor_unsafe=not bool(safe),
            single_scalar_predicted_cpis=base['scalar_cpi'].tolist(),single_mscs_predicted_cpis=base['mscs_cpi'].tolist(),
            basis=basis,basis_areas=basis_areas,delta_area=delta_area.tolist(),details=details,checks=checks,
            candidates=pool,chosen=chosen,scalar_predicted_cpis=pred['scalar_cpi'].tolist(),
            mscs_predicted_cpis=pred['mscs_cpi'].tolist(),predicted_cpis=pred['mscs_cpi'].tolist(),
            predicted_cpi_slopes=pred['cpi_slope'].tolist(),prediction_file=str(snap),prediction_sha256=sha(OUT/snap),
            cross_ranking={k:np.argsort(pred[k+'_cpi'].numpy(),kind='stable').tolist() for k in ('scalar','mscs')},
            phase='all choices before any candidate full network label')
        save(OUT/prefix/'decision.json',d);decisions.append(d)
        for i,p in enumerate(pool):
            lid=label_id(ctx['bench'],p['config']);req=requests.setdefault(lid,dict(config=p['config'],bench=ctx['bench'],x=poolxs[i],contexts=[]))
            req['contexts'].append(ctx['id'])
        feature_builds+=cache.builds;feature_seconds+=cache.seconds
        print(__import__('json').dumps(dict(phase='frozen',context=ctx['id'],solver={k:v['solution']['success'] for k,v in details.items()},
            proposals={k:{p:pp['config'][p]-c[p] for p in c if pp['config'][p]!=c[p]} for k,pp in proposals.items()})),flush=True)
    save(OUT/'frozen_batch.json',dict(decision_sha256={d['context']:sha(OUT/'roots'/d['context']/'decision.json') for d in decisions},
        candidate_labels_observed=0,area_basis_requests=basis_area_requests))
    labels={};new=0;reused=0
    for lid,req in requests.items():
        old=next((SRC/'truth'/ctxid/f'{lid}.npz' for ctxid in req['contexts'] if (SRC/'truth'/ctxid/f'{lid}.npz').exists()),None)
        if old is not None:
            with np.load(old) as z:mm=torch.from_numpy(z['m'].copy())
            from core import cpis
            cp=float(cpis(mm,compute_w_min(req['config'])).mean());origin='reused_prior_paid_label';reused+=1
        else:cp,mm=exact(model,req['x'],compute_w_min(req['config']));origin='new_forward';new+=1
        needs=any(cp<=d['cpi_cap'] for d in decisions if d['context'] in req['contexts'])
        aa=area.get(req['config']) if needs else None
        array_save(OUT/'labels'/f'{lid}.npz',m=mm)
        labels[lid]=dict(config=req['config'],bench=req['bench'],cpi=cp,area=aa,origin=origin,contexts=req['contexts'],
            source_file=str(old) if old is not None else None,area_required=needs,
            area_error=area.store.errors.get(area.store.key(req['config'])) if needs and aa is None else None)
    save(OUT/'labels.json',labels)
    rows=[]
    for d in decisions:
        ctx=next(c for c in plan['contexts'] if c['id']==d['context']);trials=[labels[label_id(ctx['bench'],p['config'])] for p in d['candidates']]
        results={k:outcome(ctx['root_area'],ctx['cpi_cap'],[trials[i] for i in ids]) for k,ids in d['chosen'].items()}
        row=dict(context=d['context'],outcomes=results,trials=trials);rows.append(row)
        print(__import__('json').dumps(dict(phase='outcome',context=d['context'],area_savings={k:v['saving_pct'] for k,v in results.items()})),flush=True)
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,new_candidate_forwards=new,reused_candidate_labels=reused,logical_queries=4*len(decisions),
        root_forwards=len(decisions),backward_outputs=12*len(decisions),area_counts=dict(area.counts),basis_area_requests=basis_area_requests,
        area_failures=area.store.n_fail,feature_builds=feature_builds,feature_seconds=feature_seconds,wall_seconds=time.perf_counter()-start))
    from reclaim_allocation_validate import validate
    save(OUT/'validation.json',validate(OUT))


if __name__=='__main__':main()
