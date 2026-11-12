"""One area-feasible query per model on a common variable-width candidate pool."""
from pathlib import Path
import json,sqlite3,time
import numpy as np
import torch
from secant_support import HERE,ROOT,read,sha,save,array_save,Inputs,FeatureCache,LocalModel,load_model,exact,config_key,compute_w_min
from core import cpis
from matched_support import label_id,outcome
from width_priority_support import width_moves,predict,check_scalar_gradient
import run_adaptive as area_runner

SRC=HERE/'confirmation_s42_01';SUP=HERE/'reclaim_supported_s42_01';OUT=HERE/'width_priority_s42_01'


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    sourceplan=read(SRC/'plan.json');sources=dict(sourceplan['source_sha256'])
    for p in [*HERE.glob('width_priority*.py'),HERE/'width_priority_protocol.txt',SRC/'validation.json',SUP/'validation.json']:
        sources[str(p)]=sha(p)
    assert read(SRC/'validation.json')['passed'] and read(SUP/'validation.json')['passed']
    for ctx in sourceplan['contexts']:
        p=SUP/'roots'/ctx['id']/'decision.json';sources[str(p)]=sha(p)
    plan=dict(contexts=sourceplan['contexts'],source_sha256=sources,seed=42,scope='already inspected roots, width-domain development pilot')
    save(OUT/'plan.json',plan)
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    model=load_model();requests={};area_truth={};decisions=[];feature_builds=0;feature_seconds=0.
    for ctx in plan['contexts']:
        root=ctx['root'];cache=FeatureCache(Inputs(ctx['bench'],root['l1_size']))
        local=LocalModel(model,cache.get(root),compute_w_min(root),'local')
        np.testing.assert_allclose(local.cpi,ctx['root_cpi'],rtol=3e-6)
        checks=check_scalar_gradient(model,local)
        pool=width_moves(root,4,{config_key(root)})
        excluded={config_key(r['action']['config']) for r in read(SUP/'roots'/ctx['id']/'decision.json')['unsupported_restrictions']}
        allowed=[i for i,a in enumerate(pool) if config_key(a['config']) not in excluded]
        xs=torch.stack([cache.get(a['config']) for a in pool]);widths=[compute_w_min(a['config']) for a in pool]
        odds=torch.einsum('nskd,cnd->cnsk',local.gell,xs-local.x[None])
        pred=predict(local.ell,odds,widths)
        choices={};prefixes={};resolved=True
        for arm in ('scalar','mscs'):
            order=sorted(allowed,key=lambda i:(float(pred[arm+'_cpi'][i]),i));prefix=[];choice=None
            for i in order:
                c=pool[i]['config'];key=config_key(c)
                if key not in area_truth:area_truth[key]=area.get(c)
                aa=area_truth[key]
                prefix.append(dict(index=i,config=c,area=aa,error=area.store.errors.get(area.store.key(c)) if aa is None else None))
                if aa is None:resolved=False;break
                if aa<=ctx['area_cap']+1e-9:choice=i;break
            choices[arm]=choice;prefixes[arm]=prefix
        if any(v is None for v in choices.values()):resolved=False
        prefix=OUT/'roots'/ctx['id'];snap=prefix/'predictions.npz'
        array_save(snap,root_ell=local.ell,root_m=local.m,logodds_slope=odds,root_u=pred['root_u'],scalar_u_slope=pred['scalar_u_slope'],
            predicted_m=pred['predicted_m'],scalar_stage=pred['scalar_stage'],mscs_stage=pred['mscs_stage'])
        decision=dict(context=ctx['id'],candidates=pool,widths=widths,choices=choices,prefixes=prefixes,resolved=resolved,
            scalar_predicted_cpis=pred['scalar_cpi'].tolist(),mscs_predicted_cpis=pred['mscs_cpi'].tolist(),
            known_unsupported_excluded=[dict(c) for c in sorted(excluded)],prediction_file=str(snap),prediction_sha256=sha(snap),
            derivative_check=checks,root_cpi=local.cpi,root_width=compute_w_min(root),candidate_labels_observed=0)
        save(prefix/'decision.json',decision);decisions.append(decision)
        if resolved:
            for i in set(choices.values()):
                c=pool[i]['config'];lid=label_id(ctx['bench'],c)
                requests[lid]=dict(config=c,bench=ctx['bench'],x=xs[i],context=ctx['id'])
        feature_builds+=cache.builds;feature_seconds+=cache.seconds
        print(json.dumps(dict(phase='frozen',context=ctx['id'],pool=len(pool),resolved=resolved,
            selected_edits={a:pool[i]['edits'] if i is not None else None for a,i in choices.items()},
            area_prefixes={a:len(p) for a,p in prefixes.items()})),flush=True)
    save(OUT/'frozen_batch.json',dict(candidate_labels_observed=0,decision_sha256={d['context']:sha(OUT/'roots'/d['context']/'decision.json') for d in decisions}))
    labels={};new=0;reused=0
    for lid,r in requests.items():
        old=SRC/'truth'/r['context']/f'{lid}.npz'
        if old.exists():
            with np.load(old) as z:mm=torch.from_numpy(z['m'].copy())
            cp=float(cpis(mm,compute_w_min(r['config'])).mean());origin='reused_prior_paid_label';reused+=1
        else:
            cp,mm=exact(model,r['x'],compute_w_min(r['config']));origin='new_forward';new+=1
        array_save(OUT/'labels'/f'{lid}.npz',m=mm)
        labels[lid]=dict(config=r['config'],bench=r['bench'],cpi=cp,area=area_truth[config_key(r['config'])],origin=origin,
            source_file=str(old) if old.exists() else None)
    save(OUT/'labels.json',labels);rows=[]
    for ctx,d in zip(plan['contexts'],decisions):
        if not d['resolved']:rows.append(dict(context=ctx['id'],bench=ctx['bench'],resolved=False));continue
        results={a:outcome(ctx['root_cpi'],ctx['area_cap'],[labels[label_id(ctx['bench'],d['candidates'][i]['config'])]]) for a,i in d['choices'].items()}
        gap=(results['scalar']['best_cpi']-results['mscs']['best_cpi'])/ctx['root_cpi']
        explanations={}
        with np.load(d['prediction_file']) as z:
            rm=torch.from_numpy(z['root_m']);root_account=(cpis(rm,compute_w_min(ctx['root']))[...,None]*rm).mean((0,1))
            for arm,i in d['choices'].items():
                c=d['candidates'][i]['config'];lid=label_id(ctx['bench'],c);w=compute_w_min(c)
                pm=torch.from_numpy(z['predicted_m'][i])
                with np.load(OUT/'labels'/f'{lid}.npz') as label:am=torch.from_numpy(label['m'])
                pred_delta=(cpis(pm,w)[...,None]*pm).mean((0,1))-root_account
                actual_delta=(cpis(am,w)[...,None]*am).mean((0,1))-root_account
                explanations[arm]=dict(classes=['Base','FP','BP','SP'],mscs_predicted_account_change=pred_delta.tolist(),
                    actual_account_change=actual_delta.tolist(),scope='stage accounts averaged; exact frozen surrogate, not gem5')
        row=dict(context=ctx['id'],bench=ctx['bench'],resolved=True,outcomes=results,normalized_gap=gap,
            status='win' if gap>1e-6 else 'loss' if gap< -1e-6 else 'tie',explanations=explanations)
        rows.append(row);print(json.dumps(dict(phase='outcome',context=ctx['id'],gains={a:r['gain_pct'] for a,r in results.items()})),flush=True)
    counts={s:sum(r.get('status')==s for r in rows) for s in ('win','tie','loss')};counts['unresolved']=sum(not r['resolved'] for r in rows)
    gate=counts['loss']==0 and counts['unresolved']==0 and any(r.get('status')=='win' and r['bench']!='602.gcc_s' for r in rows)
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,counts=counts,development_gate_passed=gate,new_candidate_forwards=new,reused_candidate_labels=reused,
        logical_candidate_queries=2*sum(r['resolved'] for r in rows),area_counts=dict(area.counts),area_failures=area.store.n_fail,
        logical_area_prefix_accesses={a:sum(len(d['prefixes'][a]) for d in decisions) for a in ('scalar','mscs')},
        root_forwards=len(decisions),backward_outputs=12*len(decisions),diagnostic_forwards=len(decisions),diagnostic_backward_outputs=4*len(decisions),
        feature_builds=feature_builds,feature_seconds=feature_seconds,wall_seconds=time.perf_counter()-start))
    from width_priority_validate import validate
    save(OUT/'validation.json',validate(OUT))
    print(json.dumps(dict(phase='complete',counts=counts,development_gate_passed=gate)),flush=True)


if __name__=='__main__':main()
