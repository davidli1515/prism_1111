"""Three cold-start policies, one paid candidate per round, private feedback."""
import json,time,sqlite3
import numpy as np
import torch
from quadratic_support import HERE,ROOT,read,sha,save,array_save,FeatureCache,Inputs,LocalModel,load_model,exact,config_key,compute_w_min
from config_dse import DEFAULT_CONFIG
from core import legal
from width_priority_support import width_moves,predict,check_scalar_gradient
from priority_walk_support import scalar_response,scalar_quadratic
from matched_support import label_id
from neighborhood_validate import BENCHES
import run_adaptive as area_runner

OUT=HERE/'priority_walk_s42_01';ARMS=('scalar_raw','scalar_quadratic','mscs_raw')


def choose_root():
    catalog=[a for a in width_moves(DEFAULT_CONFIG,1,{config_key(DEFAULT_CONFIG)}) if a['config']['fetchBufferSize']>=16]
    order=np.random.RandomState(42).permutation(len(catalog)).tolist();root=dict(DEFAULT_CONFIG);used=set();accepted=[];audit=[]
    assert order[:4]==read(HERE/'neighborhood_s42_01/root_plan.json')['selected_indices']
    for index in order[4:]:
        a=catalog[index];keys=set(a['edits']);candidate=dict(root)
        for k,v in a['edits'].items():candidate[k]=v['after']
        ok=not bool(keys&used) and legal(candidate)
        audit.append(dict(index=index,accepted=ok,edits=a['edits']))
        if ok:
            root=candidate;used.update(keys);accepted.append(index)
        if len(accepted)==2:break
    assert len(accepted)==2
    previous=[]
    folders=[HERE/'coverage_s42_01',*[HERE/'neighborhood_s42_01'/f'neighbor_{i}' for i in range(4)]]
    for folder in folders:
        p=read(folder/'plan.json');previous.extend([p['root']]+[a['config'] for a in p['candidates']])
    assert root not in previous,'Predeclared root overlaps earlier coverage domain; do not silently replace.'
    return root,dict(catalog=catalog,permutation=order,accepted_indices=accepted,audit=audit,prior_configs_checked=len(previous))


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    assert read(HERE/'neighborhood_s42_01/summary.json')['development_gate_passed'] and read(HERE/'neighborhood_s42_01/validation.json')['passed']
    sources=dict(read(HERE/'coverage_s42_01/plan.json')['source_sha256'])
    for p in [*HERE.glob('priority_walk*.py'),HERE/'priority_walk_protocol.txt',HERE/'quadratic_support.py',
        HERE/'neighborhood_s42_01/root_plan.json',HERE/'neighborhood_s42_01/summary.json',HERE/'neighborhood_s42_01/validation.json']:
        sources[str(p)]=sha(p)
    root,selection=choose_root()
    save(OUT/'plan.json',dict(seed=42,benchmarks=BENCHES,root=root,selection=selection,arms=ARMS,rounds=6,topk=1,history=6,
        source_sha256=sources,scope='three outcome-selected workloads; new composed architecture; short matched-budget trajectories'))
    print(json.dumps(dict(phase='root_frozen',root=root,accepted_indices=selection['accepted_indices'])),flush=True)
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    root_area=area.get(root);assert root_area is not None;cap=1.10*root_area
    save(OUT/'area_plan.json',dict(root_area=root_area,area_cap=cap,sources=area.store.sources))
    model=load_model();all_results=[];nforward=0;nlocal=0;feature_builds=0;feature_seconds=0.;prefix_accesses={a:0 for a in ARMS}
    area_values={config_key(root):root_area};logical_locals={a:set() for a in ARMS}
    for bench in BENCHES:
        cache=FeatureCache(Inputs(bench,64));locals_={};truth={}
        def getlocal(c):
            nonlocal nlocal
            key=config_key(c)
            if key not in locals_:
                loc=LocalModel(model,cache.get(c),compute_w_min(c),'local');locals_[key]=loc;nlocal+=1
                check=check_scalar_gradient(model,loc)
                save(OUT/'root_checks'/bench/f'{label_id(bench,c)}.json',check)
            return locals_[key]
        initial=getlocal(root);initial_cpi=initial.cpi;root_id=label_id(bench,root)
        truth[root_id]=dict(config=root,cpi=initial_cpi,m=initial.m)
        array_save(OUT/'truth'/bench/f'{root_id}.npz',m=initial.m)
        states={a:dict(root=dict(root),cpi=initial_cpi,area=root_area,radius=4,seen={config_key(root)},history=[root_id],queries=0,
            active=True,curve=[]) for a in ARMS}
        for step in range(6):
            frozen={}
            for arm,state in states.items():
                if not state['active']:continue
                loc=getlocal(state['root']);logical_locals[arm].add((bench,config_key(state['root'])))
                np.testing.assert_allclose(loc.cpi,state['cpi'],rtol=3e-6)
                cand=[a for a in width_moves(state['root'],state['radius'],state['seen']) if a['config']['fetchBufferSize']>=16]
                assert cand
                xs=torch.stack([cache.get(a['config']) for a in cand]);widths=[compute_w_min(a['config']) for a in cand]
                de=torch.einsum('nskd,cnd->cnsk',loc.gell,xs-loc.x[None]);pred=predict(loc.ell,de,widths)
                arrays=dict(root_m=loc.m,root_ell=loc.ell,logodds_slope=de,predicted_m=pred['predicted_m'],
                    scalar_stage=pred['scalar_stage'],mscs_stage=pred['mscs_stage'])
                history=state['history'][-6:]
                if arm=='scalar_quadratic':
                    hx=torch.stack([cache.get(truth[k]['config']) for k in history]);hy=torch.stack([scalar_response(truth[k]['m']) for k in history])
                    scores,info=scalar_quadratic(loc,xs,widths,hx,hy);arrays.update(info)
                else:scores=pred['mscs_cpi' if arm=='mscs_raw' else 'scalar_cpi']
                order=sorted(range(len(cand)),key=lambda i:(float(scores[i]),i));prefix=[];choice=None;resolved=True
                for i in order:
                    c=cand[i]['config'];key=config_key(c)
                    if key not in area_values:area_values[key]=area.get(c)
                    aa=area_values[key];prefix.append(dict(index=i,area=aa,error=area.store.errors.get(area.store.key(c)) if aa is None else None))
                    if aa is None:resolved=False;break
                    if aa<=cap+1e-9:choice=i;break
                prefix_accesses[arm]+=len(prefix)
                if choice is None:resolved=False
                folder=OUT/'walks'/bench/arm;snap=folder/f'predictions_{step:02}.npz';array_save(snap,**arrays)
                d=dict(bench=bench,arm=arm,step=step,root=state['root'],root_cpi=state['cpi'],radius=state['radius'],candidates=cand,widths=widths,
                    scores=scores.tolist(),scalar_predicted_cpis=pred['scalar_cpi'].tolist(),mscs_predicted_cpis=pred['mscs_cpi'].tolist(),
                    history_ids=history,choice=choice,area_prefix=prefix,resolved=resolved,prediction_file=str(snap),prediction_sha256=sha(snap),
                    phase='before any current-round candidate CPI label')
                save(folder/f'decision_{step:02}.json',d);frozen[arm]=(d,xs)
            save(OUT/'batches'/bench/f'freeze_{step:02}.json',dict(decision_sha256={a:sha(OUT/'walks'/bench/a/f'decision_{step:02}.json') for a in frozen}))
            requests={}
            for arm,(d,xs) in frozen.items():
                if not d['resolved']:continue
                i=d['choice'];c=d['candidates'][i]['config'];requests[label_id(bench,c)]=(c,xs[i])
            for lid,(c,x) in requests.items():
                if lid not in truth:
                    cp,mm=exact(model,x,compute_w_min(c));nforward+=1;truth[lid]=dict(config=c,cpi=cp,m=mm)
                    array_save(OUT/'truth'/bench/f'{lid}.npz',m=mm)
            for arm,(d,_) in frozen.items():
                state=states[arm];trial=None;accepted=False
                if not d['resolved']:state['active']=False
                else:
                    i=d['choice'];c=d['candidates'][i]['config'];lid=label_id(bench,c);value=truth[lid];aa=area_values[config_key(c)]
                    trial=dict(label_id=lid,config=c,cpi=value['cpi'],area=aa,score=d['scores'][i])
                    state['history'].append(lid);state['seen'].add(config_key(c));state['queries']+=1
                    if value['cpi']<state['cpi']*(1-1e-6):
                        state.update(root=c,cpi=value['cpi'],area=aa,radius=4);accepted=True
                    else:state['radius']=max(1,state['radius']//2)
                    state['curve'].append(state['cpi'])
                save(OUT/'walks'/bench/arm/f'round_{step:02}.json',dict(trial=trial,accepted=accepted,resolved=d['resolved'],
                    next_root=state['root'],next_cpi=state['cpi'],next_area=state['area'],next_radius=state['radius'],queries=state['queries'],
                    decision_sha256=sha(OUT/'walks'/bench/arm/f'decision_{step:02}.json')))
            print(json.dumps(dict(phase='round',bench=bench,step=step+1,gains={a:100*(1-s['cpi']/initial_cpi) for a,s in states.items()})),flush=True)
        results={}
        for arm,state in states.items():
            r=dict(bench=bench,arm=arm,initial_cpi=initial_cpi,final_cpi=state['cpi'],gain_pct=100*(1-state['cpi']/initial_cpi),
                final_config=state['root'],final_area=state['area'],curve=state['curve'],queries=state['queries'],resolved=state['active'])
            save(OUT/'walks'/bench/arm/'summary.json',r);results[arm]=r
        all_results.append(results);feature_builds+=cache.builds;feature_seconds+=cache.seconds
        print(json.dumps(dict(phase='workload_done',bench=bench,gains={a:r['gain_pct'] for a,r in results.items()},
            curves={a:r['curve'] for a,r in results.items()})),flush=True)
    rows=[]
    for rr in all_results:
        r=rr['mscs_raw'];resolved=all(v['resolved'] and v['queries']==6 for v in rr.values())
        if not resolved:rows.append(dict(bench=r['bench'],resolved=False));continue
        env=np.minimum(rr['scalar_raw']['curve'],rr['scalar_quadratic']['curve']);ms=np.array(r['curve'])
        adv=float(np.mean((env-ms)/r['initial_cpi']));end=float((env[-1]-ms[-1])/r['initial_cpi'])
        rows.append(dict(bench=r['bench'],resolved=True,scalar_envelope=env.tolist(),anytime_advantage=adv,endpoint_advantage=end,
            anytime_status='win' if adv>1e-6 else 'loss' if adv< -1e-6 else 'tie',endpoint_status='win' if end>1e-6 else 'loss' if end< -1e-6 else 'tie'))
    gate=all(r['resolved'] for r in rows) and sum(r.get('anytime_status')=='win' for r in rows)>=2 and all(r.get('endpoint_advantage',-1)>=-1e-6 for r in rows) and np.mean([r.get('anytime_advantage',0) for r in rows])>1e-6
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(results=all_results,comparisons=rows,development_gate_passed=bool(gate),new_candidate_forwards=nforward,
        logical_candidate_queries=sum(r['queries'] for rr in all_results for r in rr.values()),root_forwards=nlocal,backward_outputs=12*nlocal,
        diagnostic_forwards=nlocal,diagnostic_backward_outputs=4*nlocal,logical_root_evaluations={a:len(v) for a,v in logical_locals.items()},
        area_counts=dict(area.counts),area_failures=area.store.n_fail,logical_area_prefix_accesses=prefix_accesses,
        feature_builds=feature_builds,feature_seconds=feature_seconds,wall_seconds=time.perf_counter()-start))
    from priority_walk_validate import validate
    save(OUT/'validation.json',validate(OUT));print(json.dumps(dict(phase='complete',development_gate_passed=bool(gate),comparisons=rows)),flush=True)


if __name__=='__main__':main()
