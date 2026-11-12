"""Generate partners from gradients, never from a named resource-pair rule."""
import json,time,sqlite3
import numpy as np
import torch
from quadratic_support import HERE,ROOT,read,sha,save,array_save,Inputs,FeatureCache,LocalModel,load_model,exact,config_key,compute_w_min
from core import cpis
from matched_support import label_id,outcome
from width_priority_support import predict
from pair_priority_support import dense_singles,pair_universe
import run_adaptive as area_runner

SRC=HERE/'priority_walk_s42_01';OUT=HERE/'pair_priority_s42_01'
ARMS=('scalar_single','mscs_single','scalar_joint','mscs_joint')


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False);assert read(SRC/'validation.json')['passed']
    sourceplan=read(SRC/'plan.json');sources=dict(sourceplan['source_sha256'])
    for p in [*SRC.rglob('*.json'),*SRC.rglob('*.npz'),*HERE.glob('pair_priority*.py'),HERE/'pair_priority_protocol.txt']:
        sources[str(p)]=sha(p)
    contexts=[dict(id=f'{b}_before_q{s+1}',bench=b,step=s) for b in sourceplan['benchmarks'] for s in (4,5)]
    cap=read(SRC/'area_plan.json')['area_cap'];save(OUT/'plan.json',dict(contexts=contexts,arms=ARMS,area_cap=cap,source_sha256=sources,
        scope='automatic positive-resource pairing on six inspected states; common domain, no designated pair'))
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    model=load_model();caches={};areas={};decisions=[];requests={};n_pairs=0;n_full_pairs=0;endpoint_requests=0
    def getarea(c):
        key=config_key(c)
        if key not in areas:areas[key]=area.get(c)
        return areas[key]
    for ctx in contexts:
        b=ctx['bench'];old=read(SRC/'walks'/b/'mscs_raw'/f"decision_{ctx['step']:02}.json");root=old['root'];width=compute_w_min(root)
        seen={config_key(sourceplan['root'])}
        for s in range(ctx['step']):seen.add(config_key(read(SRC/'walks'/b/'mscs_raw'/f'round_{s:02}.json')['trial']['config']))
        if b not in caches:caches[b]=FeatureCache(Inputs(b,64))
        cache=caches[b];local=LocalModel(model,cache.get(root),width,'local');np.testing.assert_allclose(local.cpi,old['root_cpi'],rtol=3e-6)
        singles=dense_singles(root,seen);xs=torch.stack([cache.get(a['config']) for a in singles]);widths=[compute_w_min(a['config']) for a in singles]
        de=torch.einsum('nskd,cnd->cnsk',local.gell,xs-local.x[None]);single_pred=predict(local.ell,de,widths)
        basis_areas=[];eligible=[];resolved=True
        for i,a in enumerate(singles):
            if a['kind']!='single_resource' or a['sign']!=1:continue
            aa=getarea(a['config']);endpoint_requests+=1
            basis_areas.append(dict(index=i,area=aa,error=area.store.errors.get(area.store.key(a['config'])) if aa is None else None))
            if aa is None:resolved=False
            elif aa<=cap+1e-9:eligible.append(i)
        pairs=pair_universe(root,singles,eligible,seen);n_pairs+=len(pairs)
        if pairs:
            indices=torch.tensor([a['indices'] for a in pairs]);pd=de[indices[:,0]]+de[indices[:,1]]
            pair_pred=predict(local.ell,pd,[width]*len(pairs))
        else:pair_pred={a+'_cpi':torch.zeros(0) for a in ('scalar','mscs')}
        proposed={};pair_prefixes={}
        for kind in ('scalar','mscs'):
            order=sorted(range(len(pairs)),key=lambda i:(float(pair_pred[kind+'_cpi'][i]),i));choice=None;prefix=[]
            if resolved:
                for i in order:
                    c=pairs[i]['config'];aa=getarea(c)
                    prefix.append(dict(index=i,area=aa,error=area.store.errors.get(area.store.key(c)) if aa is None else None))
                    if aa is None:resolved=False;break
                    if aa<=cap+1e-9:choice=i;break
            proposed[kind]=choice;pair_prefixes[kind]=prefix
        pool=list(singles);full_pair_info=[];extra_de=[]
        for kind,i in proposed.items():
            if i is None:continue
            a=pairs[i]
            if any(v['config']==a['config'] for v in pool):continue
            x=cache.get(a['config']);actual=x-local.x;i0,i1=a['indices'];chord=xs[i0]+xs[i1]-2*local.x
            full_pair_info.append(dict(pair_index=i,generator=kind,pool_index=len(pool),input_chord_error=float(torch.linalg.norm(actual-chord)),
                input_delta_norm=float(torch.linalg.norm(actual)),indices=a['indices']))
            extra_de.append(torch.einsum('nskd,nd->nsk',local.gell,actual));pool.append(a);n_full_pairs+=1
        full_de=torch.cat([de,torch.stack(extra_de)]) if extra_de else de
        full_widths=widths+[width]*len(extra_de);pred=predict(local.ell,full_de,full_widths)
        choices={};prefixes={}
        for arm in ARMS:
            kind=arm.split('_')[0];allowed=list(range(len(singles) if arm.endswith('_single') else len(pool)))
            order=sorted(allowed,key=lambda i:(float(pred[kind+'_cpi'][i]),i));choice=None;prefix=[]
            if resolved:
                for i in order:
                    c=pool[i]['config'];aa=getarea(c)
                    prefix.append(dict(index=i,area=aa,error=area.store.errors.get(area.store.key(c)) if aa is None else None))
                    if aa is None:resolved=False;break
                    if aa<=cap+1e-9:choice=i;break
            choices[arm]=choice;prefixes[arm]=prefix
        if any(i is None for i in choices.values()):resolved=False
        snap=OUT/'states'/ctx['id']/'predictions.npz'
        array_save(snap,root_m=local.m,root_ell=local.ell,logodds_slope=full_de,predicted_m=pred['predicted_m'],
            scalar_stage=pred['scalar_stage'],mscs_stage=pred['mscs_stage'])
        d=dict(context=ctx['id'],root=root,root_cpi=local.cpi,seen=[dict(k) for k in sorted(seen)],singles_count=len(singles),
            basis_areas=basis_areas,eligible_pair_endpoints=eligible,pairs=pairs,
            pair_scalar_scores=pair_pred['scalar_cpi'].tolist(),pair_mscs_scores=pair_pred['mscs_cpi'].tolist(),
            proposed_pairs=proposed,pair_prefixes=pair_prefixes,full_pair_info=full_pair_info,candidates=pool,widths=full_widths,
            scalar_scores=pred['scalar_cpi'].tolist(),mscs_scores=pred['mscs_cpi'].tolist(),choices=choices,prefixes=prefixes,resolved=resolved,
            prediction_file=str(snap),prediction_sha256=sha(snap),non_history_candidate_labels_observed=0)
        save(OUT/'states'/ctx['id']/'decision.json',d);decisions.append(d)
        if resolved:
            for i in set(choices.values()):
                c=pool[i]['config'];requests[label_id(b,c)]=dict(config=c,bench=b,x=cache.get(c))
        print(json.dumps(dict(phase='frozen',context=ctx['id'],pairs=len(pairs),resolved=resolved,
            proposed={a:pairs[i]['edits'] if i is not None else None for a,i in proposed.items()},
            selected={a:pool[i]['edits'] if i is not None else None for a,i in choices.items()})),flush=True)
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
        ms=results['mscs_joint']['best_cpi'];sc=results['scalar_joint']['best_cpi'];single=results['mscs_single']['best_cpi']
        row=dict(context=ctx['id'],bench=ctx['bench'],resolved=True,outcomes=results,advantage_vs_scalar_joint=(sc-ms)/d['root_cpi'],
            advantage_vs_mscs_single=(single-ms)/d['root_cpi']);rows.append(row)
        print(json.dumps(dict(phase='outcome',context=ctx['id'],gains={a:r['gain_pct'] for a,r in results.items()})),flush=True)
    gate=all(r['resolved'] for r in rows) and any(r['bench']=='654.roms_s' and min(r.get('advantage_vs_scalar_joint',0),r.get('advantage_vs_mscs_single',0))>1e-6 for r in rows) and np.mean([r.get('advantage_vs_scalar_joint',0) for r in rows])>1e-6 and sum(r.get('advantage_vs_scalar_joint',0)<-1e-6 for r in rows)<=1
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,development_gate_passed=bool(gate),new_candidate_forwards=new,reused_candidate_labels=reused,
        logical_candidate_queries=4*sum(r['resolved'] for r in rows),root_forwards=6,backward_outputs=72,
        area_counts=dict(area.counts),area_failures=area.store.n_fail,common_endpoint_area_accesses=endpoint_requests,
        chord_pairs_scored=n_pairs,full_pair_inputs_built=n_full_pairs,
        logical_pair_area_prefix_accesses={a:sum(len(d['pair_prefixes'][a]) for d in decisions) for a in ('scalar','mscs')},
        logical_final_area_prefix_accesses={a:sum(len(d['prefixes'][a]) for d in decisions) for a in ARMS},
        feature_builds=sum(c.builds for c in caches.values()),feature_seconds=sum(c.seconds for c in caches.values()),wall_seconds=time.perf_counter()-start))
    from pair_priority_validate import validate
    save(OUT/'validation.json',validate(OUT));print(json.dumps(dict(phase='complete',development_gate_passed=bool(gate))),flush=True)


if __name__=='__main__':main()
