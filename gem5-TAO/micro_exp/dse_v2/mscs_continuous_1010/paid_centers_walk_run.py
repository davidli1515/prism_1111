"""Unchanged pair-greedy policies; generalized prospective-root experiment harness."""
import json,time,sqlite3
import numpy as np
import torch
from quadratic_support import HERE,ROOT,read,sha,save,array_save,FeatureCache,Inputs,LocalModel,load_model,exact,config_key,compute_w_min
from core import cpis
from matched_support import label_id
from paid_centers_walk_support import prepare_memory
from exact_cached_inputs import CachedInputs
import run_adaptive as area_runner

ARMS=('scalar_joint','mscs_joint')


def run(OUT, root, benchmarks, root_plan):
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    original=read(HERE/'pair_walk_s42_01/plan.json');sources=dict(original['source_sha256'])
    for p in [*HERE.glob('paid_centers*.py'),HERE/'paid_centers_walk_protocol.txt',HERE/'exact_cached_inputs.py',HERE/'shepard_support.py',
              HERE/'pair_walk_s42_01/summary.json',HERE/'pair_walk_s42_01/validation.json',root_plan]:
        sources[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in sources.items())
    area_runner.OUT=OUT;area=area_runner.Areas()
    old_dbs=set(ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'))|set(HERE.rglob('area_store/areas.sqlite'))
    for p in sorted(old_dbs):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    areas={}
    def getarea(c):
        key=config_key(c)
        if key not in areas:
            aa=area.get(c);areas[key]=(aa,area.store.errors.get(area.store.key(c)) if aa is None else None)
        return areas[key]
    root_area,error=getarea(root)
    if root_area is None:
        save(OUT/'root_area_failure.json',dict(root=root,error=error));raise RuntimeError(error)
    cap=1.10*root_area
    plan=dict(root=root,benchmarks=benchmarks,arms=ARMS,rounds=6,topk=1,area_cap=cap,root_area=root_area,
        source_sha256=sources,root_plan=str(root_plan),scope='all-paid-center interpolation development; inspected root; raw common pair proposals',root_forwards_include_all_paid_centers=True)
    save(OUT/'plan.json',plan)
    model=load_model();nlocal=0;new=0;reused=0;builds=0;feature_seconds=0.;results=[]
    root_access={a:set() for a in ARMS};pair_endpoints={a:0 for a in ARMS};pair_joint_areas={a:0 for a in ARMS};final_areas={a:0 for a in ARMS}
    pair_scores_count=0;full_pair_builds=0
    for bench in plan['benchmarks']:
        cache=FeatureCache(CachedInputs(bench,64));locals_={};truth={}
        def getlocal(c):
            nonlocal nlocal
            key=config_key(c)
            if key not in locals_:
                locals_[key]=LocalModel(model,cache.get(c),compute_w_min(c),'local');nlocal+=1
            return locals_[key]
        initial=getlocal(root);root_id=label_id(bench,root);truth[root_id]=dict(config=root,cpi=initial.cpi,m=initial.m)
        array_save(OUT/'truth'/bench/f'{root_id}.npz',m=initial.m)
        states={a:dict(root=dict(root),cpi=initial.cpi,area=root_area,radius=4,seen={config_key(root)},history=[root_id],curve=[],queries=0,active=True) for a in ARMS}
        for step in range(6):
            frozen={}
            for arm,state in states.items():
                if not state['active']:continue
                local=getlocal(state['root'])
                center_configs=[truth[lid]['config'] for lid in state['history']]
                centers=[getlocal(c) for c in center_configs]
                root_access[arm].update((bench,config_key(c)) for c in center_configs)
                np.testing.assert_allclose(local.cpi,state['cpi'],rtol=3e-6)
                d,arrays=prepare_memory(local,centers,cache,state['root'],state['radius'],state['seen'],arm.endswith('_joint'),getarea,cap)
                kind=arm.split('_')[0];scores=d[kind+'_scores'];order=sorted(range(len(scores)),key=lambda i:(scores[i],i));choice=None;prefix=[]
                if d['resolved']:
                    for i in order:
                        aa,error=getarea(d['candidates'][i]['config']);prefix.append(dict(index=i,area=aa,error=error))
                        if aa is None:d['resolved']=False;break
                        if aa<=cap+1e-9:choice=i;break
                if choice is None:d['resolved']=False
                pair_endpoints[arm]+=sum(r['endpoint_requests'] for r in d['pair_traces'].values())
                pair_joint_areas[arm]+=sum(r['joint_requests'] for r in d['pair_traces'].values());final_areas[arm]+=len(prefix)
                pair_scores_count+=len(d['pairs']);full_pair_builds+=len(d['full_pair_info'])
                folder=OUT/'walks'/bench/arm;snap=folder/f'predictions_{step:02}.npz';array_save(snap,**arrays)
                d.update(bench=bench,arm=arm,step=step,root=state['root'],root_cpi=state['cpi'],radius=state['radius'],history_ids=list(state['history']),center_configs=center_configs,
                    choice=choice,area_prefix=prefix,prediction_file=str(snap),prediction_sha256=sha(snap),phase='before any arm current-round candidate labels')
                save(folder/f'decision_{step:02}.json',d);frozen[arm]=d
            save(OUT/'batches'/bench/f'freeze_{step:02}.json',dict(decision_sha256={a:sha(OUT/'walks'/bench/a/f'decision_{step:02}.json') for a in frozen}))
            requests={}
            for d in frozen.values():
                if d['resolved']:
                    c=d['candidates'][d['choice']]['config'];requests[label_id(bench,c)]=c
            for lid,c in requests.items():
                if lid in truth:continue
                cp,mm=exact(model,cache.get(c),compute_w_min(c));new+=1
                truth[lid]=dict(config=c,cpi=cp,m=mm);array_save(OUT/'truth'/bench/f'{lid}.npz',m=mm)
                save(OUT/'truth'/bench/f'{lid}.json',dict(config=c,origin='new_forward',source_file=None))
            for arm,d in frozen.items():
                state=states[arm];trial=None;accepted=False
                if not d['resolved']:state['active']=False
                else:
                    i=d['choice'];c=d['candidates'][i]['config'];lid=label_id(bench,c);value=truth[lid];aa,_=getarea(c)
                    trial=dict(config=c,label_id=lid,cpi=value['cpi'],area=aa,kind=d['candidates'][i]['kind'])
                    state['seen'].add(config_key(c));state['history'].append(lid);state['queries']+=1
                    if value['cpi']<state['cpi']*(1-1e-6):state.update(root=c,cpi=value['cpi'],area=aa,radius=4);accepted=True
                    else:state['radius']=max(1,state['radius']//2)
                    state['curve'].append(state['cpi'])
                save(OUT/'walks'/bench/arm/f'round_{step:02}.json',dict(trial=trial,accepted=accepted,resolved=d['resolved'],queries=state['queries'],
                    next_root=state['root'],next_cpi=state['cpi'],next_area=state['area'],next_radius=state['radius'],
                    decision_sha256=sha(OUT/'walks'/bench/arm/f'decision_{step:02}.json')))
            print(json.dumps(dict(phase='round',root_slot=OUT.name,bench=bench,step=step+1,gains={a:100*(1-s['cpi']/initial.cpi) for a,s in states.items()})),flush=True)
        rr={}
        for arm,state in states.items():
            r=dict(bench=bench,arm=arm,initial_cpi=initial.cpi,final_cpi=state['cpi'],gain_pct=100*(1-state['cpi']/initial.cpi),
                final_config=state['root'],final_area=state['area'],curve=state['curve'],queries=state['queries'],resolved=state['active'])
            save(OUT/'walks'/bench/arm/'summary.json',r);rr[arm]=r
        results.append(rr);builds+=cache.builds;feature_seconds+=cache.seconds
    rows=[]
    for rr in results:
        b=rr['mscs_joint']['bench'];resolved=all(v['resolved'] and v['queries']==6 for v in rr.values())
        if not resolved:rows.append(dict(bench=b,resolved=False));continue
        old=read(HERE/'pair_walk_s42_01/walks'/b/'scalar_joint/summary.json')
        oldm=read(HERE/'pair_walk_s42_01/walks'/b/'mscs_joint/summary.json')
        sc=np.minimum(rr['scalar_joint']['curve'],old['curve']);ms=np.array(rr['mscs_joint']['curve']);den=rr['mscs_joint']['initial_cpi']
        versus_raw=(oldm['final_cpi']-ms[-1])/den
        adv=float(((sc-ms)/den).mean());end=float((sc[-1]-ms[-1])/den)
        rows.append(dict(bench=b,resolved=True,anytime_advantage=adv,endpoint_advantage=end,scalar_envelope=sc.tolist(),endpoint_advantage_vs_raw_mscs=float(versus_raw),
            anytime_status='win' if adv>1e-6 else 'loss' if adv< -1e-6 else 'tie',endpoint_status='win' if end>1e-6 else 'loss' if end< -1e-6 else 'tie'))
    gate=all(r['resolved'] for r in rows) and np.mean([r.get('anytime_advantage',0) for r in rows])>1e-6 and np.mean([r.get('endpoint_advantage',0) for r in rows])>1e-6 and sum(r.get('endpoint_status')=='win' for r in rows)>=2 and sum(r.get('endpoint_advantage_vs_raw_mscs',0)>1e-6 for r in rows)>=2
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(results=results,comparisons=rows,development_gate_passed=bool(gate),new_candidate_forwards=new,reused_old_labels=reused,
        logical_candidate_queries=sum(r['queries'] for rr in results for r in rr.values()),root_forwards=nlocal,backward_outputs=12*nlocal,
        logical_root_evaluations={a:len(v) for a,v in root_access.items()},logical_pair_endpoint_accesses=pair_endpoints,logical_pair_area_accesses=pair_joint_areas,
        logical_final_area_accesses=final_areas,chord_pairs_scored=pair_scores_count,full_pair_inputs_built=full_pair_builds,
        area_counts=dict(area.counts),area_failures=area.store.n_fail,feature_builds=builds,feature_seconds=feature_seconds,wall_seconds=time.perf_counter()-start))
    from paid_centers_walk_validate import validate_leaf
    save(OUT/'validation.json',validate_leaf(OUT))
    print(json.dumps(dict(phase='root_complete',root_slot=OUT.name,comparisons=rows)),flush=True)
    return read(OUT/'summary.json')



def main():
    OUT=HERE/'paid_centers_walk_s42_01'
    assert read(HERE/'paid_centers_s42_01/summary.json')['pilot_gate_passed']
    assert read(HERE/'paid_centers_s42_01/validation.json')['passed']
    original=read(HERE/'pair_walk_s42_01/plan.json')
    run(OUT,original['root'],original['benchmarks'],HERE/'pair_walk_s42_01/plan.json')


if __name__=='__main__':main()
