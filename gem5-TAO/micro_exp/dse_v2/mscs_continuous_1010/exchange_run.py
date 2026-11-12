"""Common-pool area-neutral exchanges; all choices freeze before labels."""
from pathlib import Path
import json,sqlite3,time
import numpy as np
import torch
from secant_support import HERE,ROOT,read,sha,save,array_save,Inputs,FeatureCache,LocalModel,load_model,exact,config_key,compute_w_min
from core import PARAMS,cpis
from matched_support import label_id,outcome
from width_priority_support import predict
from exchange_support import solve,rounding,response
import run_adaptive as area_runner

SRC=HERE/'confirmation_s42_01';UP=HERE/'joint_allocation_s42_01';DOWN=HERE/'reclaim_supported_s42_01';WIDE=HERE/'width_priority_s42_01';OUT=HERE/'exchange_s42_01'


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False);sourceplan=read(SRC/'plan.json');contexts=sourceplan['contexts']
    sources=dict(sourceplan['source_sha256'])
    for p in [*HERE.glob('exchange*.py'),HERE/'exchange_protocol.txt',HERE/'width_priority_support.py']:
        sources[str(p)]=sha(p)
    for folder in (UP,DOWN,WIDE):
        assert read(folder/'validation.json')['passed'];sources[str(folder/'validation.json')]=sha(folder/'validation.json')
        for ctx in contexts:
            for p in (folder/'roots'/ctx['id']).glob('*'):
                if p.is_file():sources[str(p)]=sha(p)
    save(OUT/'plan.json',dict(contexts=contexts,source_sha256=sources,seed=42,area_cap_rule='root_area',scope='common supported signed boxes; inspected development roots'))
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    model=load_model();decisions=[];requests={};areas={};builds=0;feature_seconds=0.;basis_count=0
    for ctx in contexts:
        root=ctx['root'];width=compute_w_min(root);cache=FeatureCache(Inputs(ctx['bench'],root['l1_size']))
        local=LocalModel(model,cache.get(root),width,'local');np.testing.assert_allclose(local.cpi,ctx['root_cpi'],rtol=3e-6)
        du=read(UP/'roots'/ctx['id']/'decision.json');dd=read(DOWN/'roots'/ctx['id']/'decision.json');dw=read(WIDE/'roots'/ctx['id']/'decision.json')
        basis=du['basis']+dd['basis'];basis_areas=du['basis_areas']+dd['basis_areas'];basis_count+=len(basis)
        names=[p for p in PARAMS if any(a['parameter']==p for a in basis)]
        with np.load(UP/du['prediction_file']) as z1,np.load(DOWN/dd['prediction_file']) as z2:
            np.testing.assert_allclose(z1['root_ell'],local.ell.numpy(),rtol=2e-6,atol=2e-6)
            np.testing.assert_array_equal(z1['root_ell'],z2['root_ell'])
            dirs=np.concatenate([z1['basis_logodds_slope'],z2['basis_logodds_slope']],0).astype(float)
        ell=local.ell.numpy().astype(float)
        low=dict(root);high=dict(root)
        for a in basis:
            k=a['parameter'];low[k]=min(low[k],a['config'][k]);high[k]=max(high[k],a['config'][k])
        def supported(c):return all(low[k]<=c[k]<=high[k] for k in root)
        pool=[dict(a) for a in dw['candidates'] if a['kind']=='single_resource' and supported(a['config'])]
        # Reuse ONLY old root forecasts here, never selected candidate labels.
        ids=[next(i for i,a in enumerate(dw['candidates']) if a['config']==b['config']) for b in pool]
        with np.load(dw['prediction_file']) as z:single_odds=torch.from_numpy(z['logodds_slope'][ids].copy())
        solutions={};proposal_records=[]
        for kind in ('scalar','mscs'):
            solutions[kind]=solve(kind,root,basis,names,basis_areas,ctx['root_area'],ell,dirs,width,ctx['root_cpi'])
            for start_id,sol in enumerate(solutions[kind]):
                if not sol['admissible']:continue
                for proposal in rounding(root,basis,names,sol['signed_fractions']):
                    assert supported(proposal['config'])
                    record=dict(proposal,generator=kind,start_id=start_id);proposal_records.append(record)
                    if any(a['config']==proposal['config'] for a in pool):continue
                    pool.append(dict(proposal,kind='joint_exchange',generator=kind,start_id=start_id))
        n_single=len(ids);extra=pool[n_single:]
        if extra:
            xs=torch.stack([cache.get(a['config']) for a in extra]);new_odds=torch.einsum('nskd,cnd->cnsk',local.gell,xs-local.x[None])
            odds=torch.cat([single_odds,new_odds])
        else:odds=single_odds
        pred=predict(local.ell,odds,[width]*len(pool));choices={};prefixes={};resolved=True
        for kind in ('scalar','mscs'):
            order=sorted(range(len(pool)),key=lambda i:(float(pred[kind+'_cpi'][i]),i));prefix=[];choice=None
            for i in order:
                c=pool[i]['config'];key=config_key(c)
                if key not in areas:areas[key]=area.get(c)
                aa=areas[key];prefix.append(dict(index=i,config=c,area=aa,error=area.store.errors.get(area.store.key(c)) if aa is None else None))
                if aa is None:resolved=False;break
                if aa<=ctx['root_area']+1e-9:choice=i;break
            choices[kind]=choice;prefixes[kind]=prefix
        if any(v is None for v in choices.values()):resolved=False
        snap=OUT/'roots'/ctx['id']/'predictions.npz'
        array_save(snap,root_ell=local.ell,root_m=local.m,basis_logodds_slope=dirs,logodds_slope=odds,
            predicted_m=pred['predicted_m'],scalar_stage=pred['scalar_stage'],mscs_stage=pred['mscs_stage'])
        d=dict(context=ctx['id'],root=root,basis=basis,basis_areas=basis_areas,names=names,solutions=solutions,
            proposal_records=proposal_records,candidates=pool,n_single=n_single,supported_low=low,supported_high=high,
            choices=choices,prefixes=prefixes,resolved=resolved,area_cap=ctx['root_area'],root_cpi=local.cpi,
            scalar_predicted_cpis=pred['scalar_cpi'].tolist(),mscs_predicted_cpis=pred['mscs_cpi'].tolist(),
            prediction_file=str(snap),prediction_sha256=sha(snap),candidate_labels_observed=0)
        save(OUT/'roots'/ctx['id']/'decision.json',d);decisions.append(d)
        if resolved:
            for i in set(choices.values()):
                c=pool[i]['config'];lid=label_id(ctx['bench'],c);requests[lid]=dict(config=c,bench=ctx['bench'],context=ctx['id'],x=cache.get(c))
        builds+=cache.builds;feature_seconds+=cache.seconds
        print(json.dumps(dict(phase='frozen',context=ctx['id'],joint_proposals=len(extra),resolved=resolved,
            chosen_edits={a:pool[i]['edits'] if i is not None else None for a,i in choices.items()},
            solver_successes={a:sum(s['success'] for s in vv) for a,vv in solutions.items()})),flush=True)
    save(OUT/'frozen_batch.json',dict(candidate_labels_observed=0,decision_sha256={d['context']:sha(OUT/'roots'/d['context']/'decision.json') for d in decisions}))
    labels={};new=0;reused=0
    for lid,r in requests.items():
        old=SRC/'truth'/r['context']/f'{lid}.npz'
        if old.exists():
            with np.load(old) as z:mm=torch.from_numpy(z['m'].copy())
            cp=float(cpis(mm,compute_w_min(r['config'])).mean());reused+=1;origin='reused_prior_paid_label'
        else:cp,mm=exact(model,r['x'],compute_w_min(r['config']));new+=1;origin='new_forward'
        array_save(OUT/'labels'/f'{lid}.npz',m=mm)
        labels[lid]=dict(config=r['config'],bench=r['bench'],cpi=cp,area=areas[config_key(r['config'])],origin=origin,source_file=str(old) if old.exists() else None)
    save(OUT/'labels.json',labels);rows=[]
    for ctx,d in zip(contexts,decisions):
        if not d['resolved']:rows.append(dict(context=ctx['id'],bench=ctx['bench'],resolved=False));continue
        results={a:outcome(ctx['root_cpi'],ctx['root_area'],[labels[label_id(ctx['bench'],d['candidates'][i]['config'])]]) for a,i in d['choices'].items()}
        gap=(results['scalar']['best_cpi']-results['mscs']['best_cpi'])/ctx['root_cpi']
        rows.append(dict(context=ctx['id'],bench=ctx['bench'],resolved=True,outcomes=results,normalized_gap=gap,status='win' if gap>1e-6 else 'loss' if gap< -1e-6 else 'tie'))
        print(json.dumps(dict(phase='outcome',context=ctx['id'],gains={a:r['gain_pct'] for a,r in results.items()})),flush=True)
    counts={s:sum(r.get('status')==s for r in rows) for s in ('win','tie','loss')};counts['unresolved']=sum(not r['resolved'] for r in rows)
    gate=counts['loss']==counts['unresolved']==0 and any(r.get('status')=='win' and r['bench']!='602.gcc_s' for r in rows)
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,counts=counts,development_gate_passed=gate,new_candidate_forwards=new,reused_candidate_labels=reused,
        logical_candidate_queries=2*sum(r['resolved'] for r in rows),area_counts=dict(area.counts),area_failures=area.store.n_fail,
        inherited_basis_area_observations=basis_count,root_forwards=4,backward_outputs=48,
        logical_area_prefix_accesses={a:sum(len(d['prefixes'][a]) for d in decisions) for a in ('scalar','mscs')},
        feature_builds=builds,feature_seconds=feature_seconds,wall_seconds=time.perf_counter()-start))
    from exchange_validate import validate
    save(OUT/'validation.json',validate(OUT))
    print(json.dumps(dict(phase='complete',counts=counts,development_gate_passed=gate)),flush=True)


if __name__=='__main__':main()
