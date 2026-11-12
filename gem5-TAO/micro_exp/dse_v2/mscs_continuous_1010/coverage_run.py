"""Same default architecture and same fixed rule across every SPEC17 workload."""
from pathlib import Path
import json,sqlite3,time
import numpy as np
import torch
from secant_support import HERE,ROOT,read,sha,save,array_save,Inputs,FeatureCache,LocalModel,load_model,exact,config_key,compute_w_min
from core import cpis
from common import SPEC17,PRECOMPUTE
from config_dse import DEFAULT_CONFIG
from width_priority_support import width_moves,predict,check_scalar_gradient
from matched_support import label_id,outcome
import run_adaptive as area_runner

OUT=HERE/'coverage_s42_01'


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    sources=dict(read(HERE/'width_priority_s42_01/plan.json')['source_sha256'])
    for p in [*HERE.glob('coverage*.py'),HERE/'coverage_protocol.txt']:
        sources[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in sources.items())
    root=dict(DEFAULT_CONFIG);pool=[a for a in width_moves(root,4,{config_key(root)}) if a['config']['fetchBufferSize']>=16]
    widths=[compute_w_min(a['config']) for a in pool]
    coverage=[]
    for b in SPEC17:
        paths=sorted((Path(PRECOMPUTE)/'l1_64'/b).glob('slice_*/dse_precomputed.npy'))
        ids=[int(p.parent.name.split('_')[1]) for p in paths if int(p.parent.name.split('_')[1])%2==0][:50]
        coverage.append(dict(bench=b,slice_ids=ids,available=len(ids)==50))
    save(OUT/'plan.json',dict(seed=42,workloads=SPEC17,root=root,candidates=pool,widths=widths,input_coverage=coverage,
        source_sha256=sources,scope='fixed-rule coverage diagnostic; default configs not claimed novel'))
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    root_area=area.get(root);assert root_area is not None
    cap=1.10*root_area;save(OUT/'area_plan.json',dict(root_area=root_area,area_cap=cap,sources=area.store.sources))
    areas={};requests={};decisions=[];builds=0;feature_seconds=0.;model=load_model()
    for cov in coverage:
        bench=cov['bench']
        if not cov['available']:
            d=dict(bench=bench,resolved=False,reason='fewer than 50 slices',candidate_labels_observed=0)
            save(OUT/'roots'/bench/'decision.json',d);decisions.append(d);continue
        inputs=Inputs(bench,64);assert inputs.ids==cov['slice_ids'];cache=FeatureCache(inputs)
        local=LocalModel(model,cache.get(root),compute_w_min(root),'local');checks=check_scalar_gradient(model,local)
        xs=torch.stack([cache.get(a['config']) for a in pool]);odds=torch.einsum('nskd,cnd->cnsk',local.gell,xs-local.x[None])
        pred=predict(local.ell,odds,widths);choices={};prefixes={};resolved=True
        for arm in ('scalar','mscs'):
            order=sorted(range(len(pool)),key=lambda i:(float(pred[arm+'_cpi'][i]),i));prefix=[];choice=None
            for i in order:
                c=pool[i]['config'];key=config_key(c)
                if key not in areas:areas[key]=area.get(c)
                aa=areas[key];prefix.append(dict(index=i,area=aa,error=area.store.errors.get(area.store.key(c)) if aa is None else None))
                if aa is None:resolved=False;break
                if aa<=cap+1e-9:choice=i;break
            choices[arm]=choice;prefixes[arm]=prefix
        if any(i is None for i in choices.values()):resolved=False
        snap=OUT/'roots'/bench/'predictions.npz'
        array_save(snap,root_m=local.m,root_ell=local.ell,logodds_slope=odds,predicted_m=pred['predicted_m'],
            scalar_stage=pred['scalar_stage'],mscs_stage=pred['mscs_stage'])
        d=dict(bench=bench,root_cpi=local.cpi,choices=choices,prefixes=prefixes,resolved=resolved,
            scalar_predicted_cpis=pred['scalar_cpi'].tolist(),mscs_predicted_cpis=pred['mscs_cpi'].tolist(),derivative_check=checks,
            prediction_file=str(snap),prediction_sha256=sha(snap),candidate_labels_observed=0)
        save(OUT/'roots'/bench/'decision.json',d);decisions.append(d)
        if resolved:
            for i in set(choices.values()):
                c=pool[i]['config'];requests[label_id(bench,c)]=dict(config=c,bench=bench,x=xs[i])
        builds+=cache.builds;feature_seconds+=cache.seconds
        print(json.dumps(dict(phase='frozen',bench=bench,root_cpi=local.cpi,resolved=resolved,
            edits={a:pool[i]['edits'] if i is not None else None for a,i in choices.items()})),flush=True)
    save(OUT/'frozen_batch.json',dict(candidate_labels_observed=0,decision_sha256={d['bench']:sha(OUT/'roots'/d['bench']/'decision.json') for d in decisions}))
    labels={}
    for lid,r in requests.items():
        cp,mm=exact(model,r['x'],compute_w_min(r['config']));array_save(OUT/'labels'/f'{lid}.npz',m=mm)
        labels[lid]=dict(bench=r['bench'],config=r['config'],cpi=cp,area=areas[config_key(r['config'])])
    save(OUT/'labels.json',labels);rows=[]
    for d in decisions:
        if not d['resolved']:rows.append(dict(bench=d['bench'],resolved=False));continue
        results={a:outcome(d['root_cpi'],cap,[labels[label_id(d['bench'],pool[i]['config'])]]) for a,i in d['choices'].items()}
        gap=(results['scalar']['best_cpi']-results['mscs']['best_cpi'])/d['root_cpi'];explanations={}
        with np.load(d['prediction_file']) as z:
            rm=torch.from_numpy(z['root_m']);raccount=(cpis(rm,compute_w_min(root))[...,None]*rm).mean((0,1))
            for arm,i in d['choices'].items():
                c=pool[i]['config'];lid=label_id(d['bench'],c);w=compute_w_min(c);pm=torch.from_numpy(z['predicted_m'][i])
                with np.load(OUT/'labels'/f'{lid}.npz') as zz:am=torch.from_numpy(zz['m'])
                explanations[arm]=dict(classes=['Base','FP','BP','SP'],
                    predicted_change=((cpis(pm,w)[...,None]*pm).mean((0,1))-raccount).tolist(),
                    actual_change=((cpis(am,w)[...,None]*am).mean((0,1))-raccount).tolist())
        row=dict(bench=d['bench'],resolved=True,root_cpi=d['root_cpi'],outcomes=results,normalized_advantage=gap,
            status='win' if gap>1e-6 else 'loss' if gap< -1e-6 else 'tie',choices_differ=d['choices']['scalar']!=d['choices']['mscs'],explanations=explanations)
        rows.append(row);print(json.dumps(dict(phase='outcome',bench=d['bench'],status=row['status'],gains={a:r['gain_pct'] for a,r in results.items()})),flush=True)
    counts={s:sum(r.get('status')==s for r in rows) for s in ('win','tie','loss')};counts['unresolved']=sum(not r['resolved'] for r in rows)
    nroots=sum(c['available'] for c in coverage)
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,counts=counts,mean_normalized_advantage=float(np.mean([r['normalized_advantage'] for r in rows if r['resolved']])),
        choice_disagreements=sum(r.get('choices_differ',False) for r in rows),new_candidate_forwards=len(labels),logical_candidate_queries=2*sum(r['resolved'] for r in rows),
        root_forwards=nroots,backward_outputs=12*nroots,diagnostic_forwards=nroots,diagnostic_backward_outputs=4*nroots,
        area_counts=dict(area.counts),area_failures=area.store.n_fail,
        logical_area_prefix_accesses={a:sum(len(d.get('prefixes',{}).get(a,[])) for d in decisions) for a in ('scalar','mscs')},
        feature_builds=builds,feature_seconds=feature_seconds,wall_seconds=time.perf_counter()-start))
    from coverage_validate import validate
    save(OUT/'validation.json',validate(OUT));print(json.dumps(dict(phase='complete',counts=counts)),flush=True)


if __name__=='__main__':main()
