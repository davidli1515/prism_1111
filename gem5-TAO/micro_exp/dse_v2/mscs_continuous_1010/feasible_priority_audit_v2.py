"""Resolve design-only area before spending the performance-query budget."""
from pathlib import Path
import sqlite3,time
import numpy as np
import torch
from secant_support import HERE,ROOT,read,sha,save,array_save,Inputs,load_model,exact,config_key,compute_w_min
from matched_support import label_id,outcome
from core import cpis
import run_adaptive as area_runner

SRC=HERE/'confirmation_s42_01';SUP=HERE/'reclaim_supported_s42_01';OUT=HERE/'feasible_priority_s42_02'


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    plan0=read(SRC/'plan.json');assert read(SRC/'validation.json')['passed'] and read(SUP/'validation.json')['passed']
    sources=dict(plan0['source_sha256'])
    for directory in (SRC,SUP):
        for p in directory.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.npz'):sources[str(p)]=sha(p)
    for p in (HERE/'feasible_priority_protocol.txt',Path(__file__),HERE/'feasible_priority_validate.py'):sources[str(p)]=sha(p)
    save(OUT/'plan.json',dict(contexts=plan0['contexts'],source_sha256=sources,seed=42,scope='constrained priority on four already inspected roots'))
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    contexts=[];area_truth={}
    for ctx in plan0['contexts']:
        file=SRC/'walks'/ctx['id']/'scalar/decision_00.json';d=read(file)
        failed=read(SUP/'roots'/ctx['id']/'decision.json')['unsupported_restrictions']
        excluded={config_key(a['action']['config']) for a in failed}
        allowed=[i for i,a in enumerate(d['candidates']) if config_key(a['config']) not in excluded]
        selections={};prefixes={};resolved=True
        for arm in ('scalar','mscs'):
            order=sorted(allowed,key=lambda i:(d[arm+'_predicted_cpis'][i],i));prefix=[];selected=None
            for i in order:
                c=d['candidates'][i]['config'];key=config_key(c)
                if key not in area_truth:area_truth[key]=area.get(c)
                aa=area_truth[key]
                rec=dict(index=i,config=c,area=aa,error=area.store.errors.get(area.store.key(c)) if aa is None else None)
                prefix.append(rec)
                if aa is None:resolved=False;break
                if aa<=ctx['area_cap']+1e-9:selected=i;break
            selections[arm]=selected;prefixes[arm]=prefix
        if any(i is None for i in selections.values()):resolved=False
        record=dict(context=ctx['id'],decision_file=str(file),excluded=[dict(c) for c in sorted(excluded)],
            choices=selections,prefixes=prefixes,resolved=resolved)
        contexts.append(record)
        print(__import__('json').dumps(dict(phase='area_frozen',context=ctx['id'],choices=selections,
            area_prefixes={a:len(p) for a,p in prefixes.items()},resolved=resolved)),flush=True)
    save(OUT/'frozen_choices.json',dict(contexts=contexts,candidate_cpi_labels_observed=0))
    labels={};new=0;reused=0;model=None;inputs={}
    for rec in contexts:
        if not rec['resolved']:continue
        ctx=next(c for c in plan0['contexts'] if c['id']==rec['context']);d=read(rec['decision_file'])
        for i in sorted(set(rec['choices'].values())):
            c=d['candidates'][i]['config'];lid=label_id(ctx['bench'],c)
            if lid in labels:continue
            old=SRC/'truth'/ctx['id']/f'{lid}.npz'
            if old.exists():
                with np.load(old) as z:mm=torch.from_numpy(z['m'].copy())
                cp=float(cpis(mm,compute_w_min(c)).mean());reused+=1;origin='reused_prior_paid_label'
            else:
                if model is None:model=load_model()
                key=(ctx['bench'],c['l1_size'])
                if key not in inputs:inputs[key]=Inputs(*key)
                cp,mm=exact(model,inputs[key].build(c),compute_w_min(c));new+=1;origin='new_forward'
            array_save(OUT/'labels'/f'{lid}.npz',m=mm)
            labels[lid]=dict(config=c,bench=ctx['bench'],cpi=cp,area=area_truth[config_key(c)],origin=origin,
                source_file=str(old) if old.exists() else None)
    save(OUT/'labels.json',labels);rows=[]
    for rec in contexts:
        ctx=next(c for c in plan0['contexts'] if c['id']==rec['context']);d=read(rec['decision_file'])
        if not rec['resolved']:rows.append(dict(context=ctx['id'],resolved=False));continue
        results={a:outcome(ctx['root_cpi'],ctx['area_cap'],[labels[label_id(ctx['bench'],d['candidates'][i]['config'])]]) for a,i in rec['choices'].items()}
        gap=(results['scalar']['best_cpi']-results['mscs']['best_cpi'])/ctx['root_cpi']
        rows.append(dict(context=ctx['id'],bench=ctx['bench'],resolved=True,outcomes=results,normalized_gap=gap,
            status='win' if gap>1e-6 else 'loss' if gap< -1e-6 else 'tie'))
    counts={s:sum(r.get('status')==s for r in rows) for s in ('win','tie','loss')}
    counts['unresolved']=sum(not r['resolved'] for r in rows)
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,counts=counts,new_candidate_forwards=new,reused_candidate_labels=reused,
        logical_candidate_queries=2*sum(r['resolved'] for r in rows),area_counts=dict(area.counts),area_failures=area.store.n_fail,
        logical_area_prefix_accesses={a:sum(len(r['prefixes'][a]) for r in contexts) for a in ('scalar','mscs')},
        inherited_root_forwards=4,inherited_backward_outputs=48,new_root_forwards=0,new_backward_outputs=0,
        non_gcc_win=any(r.get('status')=='win' and r['bench']!='602.gcc_s' for r in rows),wall_seconds=time.perf_counter()-start))
    from feasible_priority_validate import validate
    save(OUT/'validation.json',validate(OUT))
    print(__import__('json').dumps(dict(counts=counts,rows=rows,new_candidate_forwards=new),indent=2),flush=True)


if __name__=='__main__':main()
