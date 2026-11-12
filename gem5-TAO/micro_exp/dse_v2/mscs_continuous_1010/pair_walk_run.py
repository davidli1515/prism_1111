"""Four matched one-query greedy policies; lazy exact feasibility only."""
import json,time,sqlite3
import numpy as np
import torch
from quadratic_support import HERE,ROOT,read,sha,save,array_save,FeatureCache,Inputs,LocalModel,load_model,exact,config_key,compute_w_min
from core import cpis
from matched_support import label_id
from pair_walk_support import prepare
import run_adaptive as area_runner

SRC=HERE/'priority_walk_s42_01';OUT=HERE/'pair_walk_s42_01'
ARMS=('scalar_single','mscs_single','scalar_joint','mscs_joint')


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    assert read(HERE/'pair_priority_s42_01/summary.json')['development_gate_passed']
    assert read(HERE/'pair_priority_s42_01/validation.json')['passed'] and read(HERE/'lazy_pair_audit_s42_02/validation.json')['passed']
    original=read(SRC/'plan.json');root=original['root'];sources=dict(original['source_sha256'])
    for p in [*HERE.glob('pair_walk*.py'),HERE/'pair_walk_protocol.txt',HERE/'pair_priority_support.py',HERE/'lazy_pair_support_v2.py',
        *SRC.rglob('*.json'),*SRC.rglob('*.npz'),HERE/'pair_priority_s42_01/summary.json',HERE/'pair_priority_s42_01/validation.json',
        HERE/'lazy_pair_audit_s42_02/summary.json',HERE/'lazy_pair_audit_s42_02/validation.json']:
        sources[str(p)]=sha(p)
    area0=read(SRC/'area_plan.json');cap=area0['area_cap'];root_area=area0['root_area']
    plan=dict(root=root,benchmarks=original['benchmarks'],arms=ARMS,rounds=6,topk=1,area_cap=cap,root_area=root_area,
        source_sha256=sources,scope='four-way development ablation on inspected roots; not independent confirmation')
    save(OUT/'plan.json',plan)
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    areas={config_key(root):(root_area,None)}
    def getarea(c):
        key=config_key(c)
        if key not in areas:
            aa=area.get(c);areas[key]=(aa,area.store.errors.get(area.store.key(c)) if aa is None else None)
        return areas[key]
    model=load_model();nlocal=0;new=0;reused=0;builds=0;feature_seconds=0.;results=[]
    root_access={a:set() for a in ARMS};pair_endpoints={a:0 for a in ARMS};pair_joint_areas={a:0 for a in ARMS};final_areas={a:0 for a in ARMS}
    pair_scores_count=0;full_pair_builds=0
    for bench in plan['benchmarks']:
        cache=FeatureCache(Inputs(bench,64));locals_={};truth={}
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
                local=getlocal(state['root']);root_access[arm].add((bench,config_key(state['root'])))
                np.testing.assert_allclose(local.cpi,state['cpi'],rtol=3e-6)
                d,arrays=prepare(local,cache,state['root'],state['radius'],state['seen'],arm.endswith('_joint'),getarea,cap)
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
                d.update(bench=bench,arm=arm,step=step,root=state['root'],root_cpi=state['cpi'],radius=state['radius'],history_ids=list(state['history']),
                    choice=choice,area_prefix=prefix,prediction_file=str(snap),prediction_sha256=sha(snap),phase='before any arm current-round candidate labels')
                save(folder/f'decision_{step:02}.json',d);frozen[arm]=d
            save(OUT/'batches'/bench/f'freeze_{step:02}.json',dict(decision_sha256={a:sha(OUT/'walks'/bench/a/f'decision_{step:02}.json') for a in frozen}))
            requests={}
            for d in frozen.values():
                if d['resolved']:
                    c=d['candidates'][d['choice']]['config'];requests[label_id(bench,c)]=c
            for lid,c in requests.items():
                if lid in truth:continue
                old=SRC/'truth'/bench/f'{lid}.npz'
                if old.exists():
                    with np.load(old) as z:mm=torch.from_numpy(z['m'].copy())
                    cp=float(cpis(mm,compute_w_min(c)).mean());reused+=1;origin='old_paid_label_reused_after_choice'
                else:cp,mm=exact(model,cache.get(c),compute_w_min(c));new+=1;origin='new_forward'
                truth[lid]=dict(config=c,cpi=cp,m=mm);array_save(OUT/'truth'/bench/f'{lid}.npz',m=mm)
                save(OUT/'truth'/bench/f'{lid}.json',dict(config=c,origin=origin,source_file=str(old) if old.exists() else None))
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
            print(json.dumps(dict(phase='round',bench=bench,step=step+1,gains={a:100*(1-s['cpi']/initial.cpi) for a,s in states.items()})),flush=True)
        rr={}
        for arm,state in states.items():
            r=dict(bench=bench,arm=arm,initial_cpi=initial.cpi,final_cpi=state['cpi'],gain_pct=100*(1-state['cpi']/initial.cpi),
                final_config=state['root'],final_area=state['area'],curve=state['curve'],queries=state['queries'],resolved=state['active'])
            save(OUT/'walks'/bench/arm/'summary.json',r);rr[arm]=r
        results.append(rr);builds+=cache.builds;feature_seconds+=cache.seconds
    rows=[]
    for rr in results:
        b=rr['mscs_joint']['bench'];old=read(SRC/'walks'/b/'scalar_quadratic/summary.json');resolved=all(v['resolved'] and v['queries']==6 for v in rr.values())
        if not resolved:rows.append(dict(bench=b,resolved=False));continue
        env=np.minimum(np.minimum(rr['scalar_single']['curve'],rr['scalar_joint']['curve']),old['curve']);ms=np.array(rr['mscs_joint']['curve']);den=rr['mscs_joint']['initial_cpi']
        adv=float(((env-ms)/den).mean());end=float((env[-1]-ms[-1])/den);single=(rr['mscs_single']['final_cpi']-ms[-1])/den
        rows.append(dict(bench=b,resolved=True,scalar_envelope=env.tolist(),anytime_advantage=adv,endpoint_advantage=end,endpoint_advantage_vs_mscs_single=float(single),
            anytime_status='win' if adv>1e-6 else 'loss' if adv< -1e-6 else 'tie',endpoint_status='win' if end>1e-6 else 'loss' if end< -1e-6 else 'tie'))
    gate=all(r['resolved'] for r in rows) and sum(r.get('anytime_status')=='win' for r in rows)>=2 and np.mean([r.get('anytime_advantage',0) for r in rows])>1e-6 and all(r.get('endpoint_advantage',-1)>=-1e-6 for r in rows) and any(r['bench']=='654.roms_s' and r.get('endpoint_advantage_vs_mscs_single',0)>1e-6 for r in rows)
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(results=results,comparisons=rows,development_gate_passed=bool(gate),new_candidate_forwards=new,reused_old_labels=reused,
        logical_candidate_queries=sum(r['queries'] for rr in results for r in rr.values()),root_forwards=nlocal,backward_outputs=12*nlocal,
        logical_root_evaluations={a:len(v) for a,v in root_access.items()},logical_pair_endpoint_accesses=pair_endpoints,logical_pair_area_accesses=pair_joint_areas,
        logical_final_area_accesses=final_areas,chord_pairs_scored=pair_scores_count,full_pair_inputs_built=full_pair_builds,
        area_counts=dict(area.counts),area_failures=area.store.n_fail,feature_builds=builds,feature_seconds=feature_seconds,wall_seconds=time.perf_counter()-start))
    from pair_walk_validate import validate
    save(OUT/'validation.json',validate(OUT));print(json.dumps(dict(phase='complete',development_gate_passed=bool(gate),comparisons=rows)),flush=True)


if __name__=='__main__':main()
