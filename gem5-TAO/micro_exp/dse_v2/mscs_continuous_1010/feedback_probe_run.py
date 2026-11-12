"""Six frozen-state, equal-history tests; never pretend these are new walks."""
import json,time,sqlite3
import numpy as np
import torch
from quadratic_support import HERE,ROOT,read,sha,save,array_save,Inputs,FeatureCache,LocalModel,load_model,exact,config_key,compute_w_min
from core import cpis
from matched_support import label_id,outcome
from priority_walk_support import scalar_response,scalar_quadratic
from feedback_probe_support import mscs_response,mscs_quadratic
import run_adaptive as area_runner

SRC=HERE/'priority_walk_s42_01';OUT=HERE/'feedback_probe_s42_01';ARMS=('mscs_raw','scalar_quadratic','mscs_quadratic')


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False);assert read(SRC/'validation.json')['passed']
    sourceplan=read(SRC/'plan.json');sources=dict(sourceplan['source_sha256'])
    for p in [*SRC.rglob('*.json'),*SRC.rglob('*.npz'),*HERE.glob('feedback_probe*.py'),HERE/'feedback_probe_protocol.txt']:
        sources[str(p)]=sha(p)
    contexts=[dict(id=f'{b}_before_q{s+1}',bench=b,step=s) for b in sourceplan['benchmarks'] for s in (4,5)]
    cap=read(SRC/'area_plan.json')['area_cap'];save(OUT/'plan.json',dict(contexts=contexts,source_sha256=sources,arms=ARMS,area_cap=cap,
        scope='six conditional raw-M S C S states; equal past feedback; not independent new trajectories'))
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    model=load_model();caches={};decisions=[];requests={};areas={}
    for ctx in contexts:
        b=ctx['bench'];oldfile=SRC/'walks'/b/'mscs_raw'/f"decision_{ctx['step']:02}.json";old=read(oldfile)
        cache=caches.setdefault(b,FeatureCache(Inputs(b,64))) if b not in caches else caches[b]
        root=old['root'];local=LocalModel(model,cache.get(root),compute_w_min(root),'local')
        np.testing.assert_allclose(local.cpi,old['root_cpi'],rtol=3e-6)
        cand=old['candidates'];xs=torch.stack([cache.get(a['config']) for a in cand]);widths=old['widths']
        history=old['history_ids'];hc=[];hm=[]
        root_id=label_id(b,sourceplan['root']);config_lookup={root_id:sourceplan['root']}
        for s in range(ctx['step']):
            past=read(SRC/'walks'/b/'mscs_raw'/f'round_{s:02}.json')['trial'];config_lookup[past['label_id']]=past['config']
        for lid in history:
            hc.append(cache.get(config_lookup[lid]))
            with np.load(SRC/'truth'/b/f'{lid}.npz') as z:hm.append(torch.from_numpy(z['m'].copy()))
        hx=torch.stack(hc);sp,si=scalar_quadratic(local,xs,widths,hx,torch.stack([scalar_response(m) for m in hm]))
        mp,mi=mscs_quadratic(local,xs,widths,hx,torch.stack([mscs_response(m) for m in hm]))
        scores=dict(mscs_raw=torch.tensor(old['scores']),scalar_quadratic=sp,mscs_quadratic=mp)
        arrays=dict(root_m=local.m,root_ell=local.ell,logodds_slope=mi['logodds_slope'])
        arrays.update({'scalar__'+k:v for k,v in si.items()});arrays.update({'mscs__'+k:v for k,v in mi.items()})
        choices={};prefixes={};resolved=True
        for arm in ARMS:
            order=sorted(range(len(cand)),key=lambda i:(float(scores[arm][i]),i));choice=None;prefix=[]
            for i in order:
                c=cand[i]['config'];key=config_key(c)
                if key not in areas:areas[key]=area.get(c)
                aa=areas[key];prefix.append(dict(index=i,area=aa,error=area.store.errors.get(area.store.key(c)) if aa is None else None))
                if aa is None:resolved=False;break
                if aa<=cap+1e-9:choice=i;break
            choices[arm]=choice;prefixes[arm]=prefix
        if any(i is None for i in choices.values()):resolved=False
        assert choices['mscs_raw']==old['choice']
        snap=OUT/'states'/ctx['id']/'predictions.npz';array_save(snap,**arrays)
        d=dict(context=ctx['id'],old_decision=str(oldfile),root=root,root_cpi=local.cpi,history_ids=history,candidates=cand,widths=widths,
            scores={a:p.tolist() for a,p in scores.items()},choices=choices,prefixes=prefixes,resolved=resolved,
            prediction_file=str(snap),prediction_sha256=sha(snap),non_history_candidate_labels_observed=0)
        save(OUT/'states'/ctx['id']/'decision.json',d);decisions.append(d)
        if resolved:
            for i in set(choices.values()):
                c=cand[i]['config'];requests[label_id(b,c)]=dict(config=c,bench=b,x=xs[i])
        print(json.dumps(dict(phase='frozen',context=ctx['id'],resolved=resolved,edits={a:cand[i]['edits'] if i is not None else None for a,i in choices.items()})),flush=True)
    save(OUT/'frozen_batch.json',dict(non_history_candidate_labels_observed=0,decision_sha256={d['context']:sha(OUT/'states'/d['context']/'decision.json') for d in decisions}))
    labels={};new=0;reused=0
    for lid,r in requests.items():
        old=SRC/'truth'/r['bench']/f'{lid}.npz'
        if old.exists():
            with np.load(old) as z:mm=torch.from_numpy(z['m'].copy())
            cp=float(cpis(mm,compute_w_min(r['config'])).mean());reused+=1;origin='reused_after_freeze'
        else:cp,mm=exact(model,r['x'],compute_w_min(r['config']));new+=1;origin='new_forward'
        array_save(OUT/'labels'/f'{lid}.npz',m=mm)
        labels[lid]=dict(config=r['config'],bench=r['bench'],cpi=cp,area=areas[config_key(r['config'])],origin=origin,source_file=str(old) if old.exists() else None)
    save(OUT/'labels.json',labels);rows=[]
    for ctx,d in zip(contexts,decisions):
        if not d['resolved']:rows.append(dict(context=ctx['id'],bench=ctx['bench'],resolved=False));continue
        results={a:outcome(d['root_cpi'],cap,[labels[label_id(ctx['bench'],d['candidates'][i]['config'])]]) for a,i in d['choices'].items()}
        ms=results['mscs_quadratic']['best_cpi'];sc=results['scalar_quadratic']['best_cpi'];raw=results['mscs_raw']['best_cpi']
        row=dict(context=ctx['id'],bench=ctx['bench'],resolved=True,outcomes=results,advantage_vs_scalar=(sc-ms)/d['root_cpi'],advantage_vs_raw=(raw-ms)/d['root_cpi'])
        rows.append(row);print(json.dumps(dict(phase='outcome',context=ctx['id'],gains={a:r['gain_pct'] for a,r in results.items()})),flush=True)
    gate=all(r['resolved'] for r in rows) and any(r['bench']=='654.roms_s' and r.get('advantage_vs_scalar',0)>1e-6 for r in rows)
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,roms_development_gate_passed=gate,new_candidate_forwards=new,reused_candidate_labels=reused,
        logical_candidate_queries=3*sum(r['resolved'] for r in rows),new_root_forwards=6,backward_outputs=72,
        area_counts=dict(area.counts),area_failures=area.store.n_fail,logical_area_prefix_accesses={a:sum(len(d['prefixes'][a]) for d in decisions) for a in ARMS},
        feature_builds=sum(c.builds for c in caches.values()),feature_seconds=sum(c.seconds for c in caches.values()),wall_seconds=time.perf_counter()-start))
    from feedback_probe_validate import validate
    save(OUT/'validation.json',validate(OUT));print(json.dumps(dict(phase='complete',roms_development_gate_passed=gate)),flush=True)


if __name__=='__main__':main()
