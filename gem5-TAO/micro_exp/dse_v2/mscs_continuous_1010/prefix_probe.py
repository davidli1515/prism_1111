"""Complete the predeclared common-prefix probe, retaining every prior version."""
import json
from pathlib import Path
import sqlite3
import sys
import time
import torch
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
FRESH=ROOT/'mscs_fresh_disagreement_1010/results_s42_01'
sys.path.insert(0,str(ROOT/'mscs_fresh_disagreement_1010'))
from fresh_disagreement_support import (read,sha,save,array_save,choices,outcome,pressure,
    LocalAllocator,LocalModel,Inputs,FeatureCache,load_model,exact,config_key,compute_w_min,moves)
from validate_matched import independent_pairs,advance_feedback
from adaptive_validate import prediction_replay,close
import run_adaptive as area_runner

OUT=HERE/'prefix_s42_01'


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    parent=read(FRESH/'confirmation_plan.json');assert len(parent['contexts'])==1
    ctx=parent['contexts'][0];seed=next(c for c in parent['cells'] if c['primary'])
    initial_d=read(seed['decision_file']);initial_d['prediction_file']=str(FRESH/initial_d['prediction_file'])
    sources={str(p):sha(p) for p in (HERE/'prefix_protocol.txt',Path(__file__),
        FRESH/'confirmation_plan.json',FRESH/'labels.json')}
    sources.update(parent['source_sha256']);assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'plan.json',dict(context=ctx,max_rounds=3,source_sha256=sources,
         scope='continuation of one selected root; first differing batch or six logical queries'))
    # Selection already frozen in the preceding full-pool decision.
    seed_labels={r['id']:r for r in read(FRESH/'labels.json')['records']}
    allowed={seed['label_ids'][str(i)] for i in seed['chosen']['local']}
    seed_labels={k:v for k,v in seed_labels.items() if k in allowed}
    policy=LocalAllocator();seen={config_key(ctx['root'])};cur=ctx['root'];cp=ctx['root_cpi'];aa=ctx['root_area'];radius=4
    model=load_model();cache=FeatureCache(Inputs(ctx['bench'],cur['l1_size']))
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    rounds=[];new_calls=0
    for step in range(3):
        width=compute_w_min(cur)
        if step==0:
            d=dict(initial_d);d['chosen_by_arm']=seed['chosen'];xs=None;local=None
        else:
            local=LocalModel(model,cache.get(cur),width,'local');close(cp,local.cpi)
            candidates=moves(cur,radius,seen);xs=torch.stack([cache.get(a['config']) for a in candidates])
            pred=local.predict(xs);snapshot=Path(f'predictions_{step:02}.npz')
            array_save(OUT/snapshot,root_m=local.m,root_cs=local.cs,root_ell=local.ell,
                stage_slope=pred['stage_slope'],logodds_slope=pred['logodds_slope'],predicted_m=pred['mscs'])
            d=dict(step=step,root=cur,root_cpi=cp,root_area=aa,radius=radius,candidates=candidates,
                scalar_predicted_cpis=pred['scalar_cpi'].tolist(),mscs_predicted_cpis=pred['mscs_cpi'].tolist(),
                predicted_cpis=pred['mscs_cpi'].tolist(),predicted_cpi_slopes=pred['cpi_slope'].tolist(),
                allocator_before=policy.snapshot(),prediction_file=str(snapshot),prediction_sha256=sha(OUT/snapshot))
            d['chosen_by_arm']=choices(d,list(range(len(candidates))))
        d['allocator_before']=policy.snapshot();d['step']=step
        d['phase']='before_this_batch_labels';save(OUT/f'decision_{step:02}.json',d)
        differing=any(set(ids)!=set(d['chosen_by_arm']['scalar']) for ids in d['chosen_by_arm'].values())
        trials={}
        for i in sorted(set(sum(d['chosen_by_arm'].values(),[]))):
            action=d['candidates'][i];c=action['config']
            if step==0:
                old=seed_labels[seed['label_ids'][str(i)]];value,ar=old['cpi'],old['area'];mm=None
                source='paid_primary_seed_batch'
            else:
                value,mm=exact(model,xs[i],width);new_calls+=1
                ar=area.get(c) if value<cp*(1-1e-6) else None;source='new_forward'
                array_save(OUT/f'stacks_{step:02}_{i:03}.npz',m=mm)
            trials[str(i)]=dict(action,cpi=value,area=ar,origin=source,
                scalar_predicted_cpi=d['scalar_predicted_cpis'][i],mscs_predicted_cpi=d['mscs_predicted_cpis'][i],
                area_error=area.store.errors.get(area.store.key(c)) if ar is None else None)
        results={};nexts={}
        for arm,ids in d['chosen_by_arm'].items():
            recs=[trials[str(i)] for i in ids];value=outcome(cp,ctx['area_cap'],recs)
            results[arm]=value
            if value['resolved']:
                best=min((r for r in recs if r['area'] is not None and r['area']<=ctx['area_cap']+1e-9
                    and r['cpi']<cp*(1-1e-6)),key=lambda r:r['cpi'],default=None)
                nexts[arm]=dict(config=cur if best is None else best['config'],cpi=cp if best is None else best['cpi'],
                    area=aa if best is None else best['area'],radius=max(1,radius//2) if best is None else 4,
                    gain_from_original_pct=100*(1-(cp if best is None else best['cpi'])/ctx['root_cpi']))
        picked=[trials[str(i)] for i in d['chosen_by_arm']['local']]
        policy.observe(cp,picked,[r['scalar_predicted_cpi'] for r in picked],
            [r['mscs_predicted_cpi'] for r in picked],[r['cpi'] for r in picked])
        rec=dict(step=step,trials=trials,outcomes=results,next_by_arm=nexts,differing_candidate_sets=differing,
                 allocator_after=policy.snapshot(),logical_queries_per_arm=2*(step+1))
        save(OUT/f'round_{step:02}.json',rec);rounds.append(rec)
        print(json.dumps(dict(step=step+1,queries=2*(step+1),differing=differing,
            gains={a:r['gain_from_original_pct'] for a,r in nexts.items()})),flush=True)
        if differing or len(nexts)!=4:break
        assert len({config_key(n['config']) for n in nexts.values()})==1
        seen.update(config_key(t['config']) for t in picked)
        nxt=nexts['local'];cur,cp,aa,radius=nxt['config'],nxt['cpi'],nxt['area'],nxt['radius']
    save(OUT/'summary.json',dict(rounds=rounds,new_candidate_forwards=new_calls,reused_seed_labels=2,
        stopped_on_disagreement=rounds[-1]['differing_candidate_sets'],area_counts=dict(area.counts),
        area_failures=area.store.n_fail,wall_seconds=time.perf_counter()-start))
    # Independent replay uses earlier labels only, not the LocalAllocator code.
    state=dict(completed_rounds=0,buckets={});seen={config_key(ctx['root'])};cur=ctx['root'];cp=ctx['root_cpi'];radius=4
    for step,r in enumerate(rounds):
        d=read(OUT/f'decision_{step:02}.json');assert d['allocator_before']==state
        assert d['root']==cur and d['radius']==radius;close(d['root_cpi'],cp)
        assert d['candidates']==moves(cur,radius,seen)
        arr=prediction_replay(OUT,d,'local',compute_w_min(cur));arr.close()
        assert independent_pairs(d,list(range(len(d['candidates']))))==d['chosen_by_arm']
        picked=[r['trials'][str(i)] for i in d['chosen_by_arm']['local']]
        advance_feedback(state,dict(root_cpi=cp,trials=picked));assert state==r['allocator_after']
        for arm,ids in d['chosen_by_arm'].items():
            recs=[r['trials'][str(i)] for i in ids]
            feasible=[t for t in recs if t['area'] is not None and t['area']<=ctx['area_cap']+1e-9 and t['cpi']<cp*(1-1e-6)]
            best=min(feasible,key=lambda t:t['cpi'],default=None)
            unknown=[t for t in recs if t['area'] is None and t['cpi']<cp*(1-1e-6)]
            high=cp if best is None else best['cpi'];low=min([high]+[t['cpi'] for t in unknown])
            assert r['outcomes'][arm]['cpi_lower']==low and r['outcomes'][arm]['cpi_upper']==high
            if low==high:
                n=r['next_by_arm'][arm];assert n['config']==(cur if best is None else best['config']);close(n['cpi'],high)
        if r['differing_candidate_sets']:
            assert step==len(rounds)-1
        elif len(r['next_by_arm'])==4:
            seen.update(config_key(t['config']) for t in picked);n=r['next_by_arm']['local']
            cur,cp,radius=n['config'],n['cpi'],n['radius']
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'validation.json',dict(passed=True,rounds_replayed=len(rounds),seed_feedback_primary_only=True,
        first_disagreement_stop=True,max_logical_queries=6,predictors_and_acceptance_replayed=True,sources_unchanged=True))


if __name__=='__main__':main()
