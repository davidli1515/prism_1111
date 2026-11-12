#!/usr/bin/env python3
"""Run unchanged greedy implementations at deterministically selected new roots."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
import time
import traceback
import numpy as np
from newroots_support import (HERE,ROOT,ADAPTIVE,HYBRID,ARMS,BENCHES,STEPS,TOPK,
    DEFAULT_CKPT,config_key,legal,moves,Inputs,load_model,pareto_mask,sha,read,
    exposure_inventory,nearest_unexposed,bind_runners)

OUT=HERE/'results_s42_01'
START=time.perf_counter()


def save(name,obj):
    path=OUT/name; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False,
                  default=lambda v:v.tolist() if hasattr(v,'tolist') else v.item())


def status(phase,**kw):
    value=dict(phase=phase,elapsed_seconds=time.perf_counter()-START,**kw)
    temp=OUT/'status.tmp'; temp.write_text(json.dumps(value,indent=2));temp.replace(OUT/'status.json')
    print(json.dumps(value),flush=True)


def prepare():
    from adaptive_validate import validate
    save('previous_revalidation.json',validate(ADAPTIVE/'results_s42_01'))
    old=read(ADAPTIVE/'results_s42_01/plan.json')
    sources=dict(old['source_sha256'])
    assert all(sha(p)==h for p,h in sources.items())
    exposed,exposure_files,counts=exposure_inventory()
    sources.update(exposure_files)
    contexts,selection=[],[]
    checkpoint=sha(DEFAULT_CKPT)
    for bench in BENCHES:
        p=ROOT/'mscs_guard_v5/results_full_01'/bench/'s42/initial.json'
        archive=read(p);sources[str(p)]=sha(p)
        assert archive['checkpoint_sha256']==checkpoint
        points=np.c_[archive['cpis'],archive['areas']]
        front=np.flatnonzero(pareto_mask(points))
        front=front[np.argsort(points[front,1],kind='stable')]
        configs=[archive['configs'][int(i)] for i in front]
        rank,target,eligible=nearest_unexposed(configs,bench,exposed)
        i=int(front[rank]);c=configs[rank]
        assert legal(c) and (bench,config_key(c)) not in exposed
        contexts.append(dict(id=f'{bench}_newrank{rank}',bench=bench,group='new_greedy_root',
            rank=rank,archive_index=i,root=c,root_cpi=archive['cpis'][i],
            root_area=archive['areas'][i],area_cap=1.10*archive['areas'][i],
            area_sources=archive['area_sources'],initial_candidates=moves(c)))
        selection.append(dict(bench=bench,archive=str(p),front_archive_indices=front.tolist(),
            target_rank=target,selected_rank=rank,eligible_by_rank=eligible,
            previously_exposed_configurations=counts[bench]))
    assert len(contexts)==4
    save('exclusions.json',dict(scope='all configurations in four greedy-development JSON result trees',
        counts=counts,source_sha256=exposure_files,
        exposed=[dict(bench=b,config=dict(k)) for b,k in sorted(exposed)]))
    for p in list(HERE.glob('*.py'))+[HERE/'PROTOCOL.txt',ADAPTIVE/'run_adaptive.py',HYBRID/'run_hybrid.py']:
        sources[str(p)]=sha(p)
    plan=dict(seed=42,contexts=contexts,arms=ARMS,steps=STEPS,topk=TOPK,
        max_candidate_verifications=144,root_selection=selection,
        exclusion_file='exclusions.json',exclusion_sha256=sha(OUT/'exclusions.json'),
        source_sha256=sources,oracle_audit_queries=0,reused_control_candidate_queries=0,
        policy_implementation=str(ADAPTIVE/'adaptive_support.py'),
        scope='new starting configurations for this greedy series; same four benchmarks and source archives; frozen-surrogate only')
    save('plan.json',plan)
    return plan


def main():
    OUT.mkdir(exist_ok=False);status('preparing')
    plan=prepare();harness_h,harness_a=bind_runners(OUT,START)
    assert harness_a.RankAllocator.__module__=='adaptive_support'
    area=harness_a.Areas()
    p=ADAPTIVE/'results_s42_01/area_store/areas.sqlite'
    area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    model=load_model();walks=[]
    for ctx in plan['contexts']:
        assert area.store.sources==ctx['area_sources']
    for ri,ctx in enumerate(plan['contexts']):
        status('loading_inputs',context=ctx['id'])
        inputs=Inputs(ctx['bench'],ctx['root']['l1_size'])
        save(Path('inputs')/f"{ctx['id']}.json",dict(slice_ids=inputs.ids,load_seconds=inputs.load_seconds))
        aa=area.get(ctx['root']);assert aa is not None and abs(aa-ctx['root_area'])<1e-6
        registry={}
        def record(c,cp,m):
            key=config_key(c)
            if key in registry:
                assert np.isclose(cp,registry[key],rtol=3e-6)
            registry[key]=float(cp)
        # This registry only asserts repeated truths; no walk can read its values.
        order=list(ARMS[ri%len(ARMS):])+list(ARMS[:ri%len(ARMS)])
        for arm in order:
            runner=harness_a if arm=='adaptive' else harness_h
            result,_=runner.walk(ctx,arm,model,inputs,area,record)
            walks.append(result)
        del inputs,registry
    paired=[]
    for ctx in plan['contexts']:
        rr={w['arm']:w for w in walks if w['context']==ctx['id']}
        cp=rr['scalar_excess']['final_cpi']
        paired.append(dict(context=ctx['id'],group=ctx['group'],
            final_cpi={a:rr[a]['final_cpi'] for a in ARMS},
            gain_pct={a:100*rr[a]['gain'] for a in ARMS},
            adaptive_vs_scalar_gain_pp=100*(cp-rr['adaptive']['final_cpi'])/ctx['root_cpi'],
            hybrid_vs_scalar_gain_pp=100*(cp-rr['hybrid']['final_cpi'])/ctx['root_cpi']))
    assert all(sha(p)==digest for p,digest in plan['source_sha256'].items())
    save('summary.json',dict(paired=paired,walks=walks,area_counts=dict(area.counts),
        area_failures=area.store.n_fail,area_errors=dict(area.store.errors),
        wall_seconds=time.perf_counter()-START,source_preservation_passed=True,
        limitations=plan['scope']+'; no end-to-end runtime comparison or Protect/Decouple validation'))
    from newroots_validate import validate
    save('validation.json',validate(OUT))
    from newroots_report import report
    report(OUT)
    status('complete',new_candidate_queries=sum(w['counts']['candidate_forward_calls'] for w in walks),
           area_failures=area.store.n_fail)


if __name__=='__main__':
    try:
        main()
    except Exception:
        if OUT.exists():
            (OUT/'ERROR.txt').write_text(traceback.format_exc())
            status('failed',traceback_file=str(OUT/'ERROR.txt'))
        raise
