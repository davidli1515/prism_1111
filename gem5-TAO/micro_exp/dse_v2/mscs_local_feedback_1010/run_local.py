"""Three-case regression gate for local feedback; frozen controls are read-only."""
import json
from pathlib import Path
import sqlite3
import sys
import time
import traceback
import numpy as np
from local_support import (HERE,ROOT,ADAPTIVE,HYBRID,NEWROOTS,ARMS,REGRESSIONS,
    COMPARISON_ARMS,STEPS,TOPK,config_key,Inputs,load_model)
import run_adaptive as base
import local_walk

OUT=HERE/'results_s42_01';START=time.perf_counter()
base.OUT=local_walk.OUT=OUT;base.START=START
save,status,sha,read=base.save,base.status,base.sha,base.load


def prepare():
    sys.path.insert(0,str(NEWROOTS))
    from adaptive_validate import validate as validate_old
    from newroots_validate import validate as validate_new
    save('control_revalidation.json',dict(old=validate_old(ADAPTIVE/'results_s42_01'),
                                        new=validate_new(NEWROOTS/'results_s42_01')))
    old=read(ADAPTIVE/'results_s42_01/plan.json');new=read(NEWROOTS/'results_s42_01/plan.json')
    sources=dict(old['source_sha256']);sources.update(new['source_sha256'])
    assert all(sha(p)==h for p,h in sources.items())
    contexts=[]
    for cid in REGRESSIONS:
        pool=new if 'newrank' in cid else old
        ctx=dict(next(c for c in pool['contexts'] if c['id']==cid))
        ctx['group']='targeted_regression'
        ctx['controls']={a:str(NEWROOTS/'results_s42_01') if 'newrank' in cid else
            str((ADAPTIVE if a=='adaptive' else HYBRID)/'results_s42_01')
            for a in ('scalar_excess','hybrid','adaptive')}
        contexts.append(ctx)
    for directory in (ADAPTIVE/'results_s42_01',HYBRID/'results_s42_01',NEWROOTS/'results_s42_01'):
        for p in directory.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.npz','.txt'):
                sources[str(p)]=sha(p)
    for p in list(HERE.glob('*.py'))+[HERE/'PROTOCOL.txt',HERE/'WALK_DIFF.patch']:
        sources[str(p)]=sha(p)
    plan=dict(seed=42,contexts=contexts,arms=ARMS,steps=STEPS,topk=TOPK,
        max_candidate_verifications=36,source_sha256=sources,
        gate='local no worse than hybrid AND adaptive at all three roots; at least one strict improvement over either; tolerance1e-6*root CPI',
        scope='targeted development regression only; optional fresh starts only after gate passes')
    save('plan.json',plan)
    from local_audit import shadow_audit
    save('shadow_audit.json',shadow_audit(contexts))
    return plan


def main():
    OUT.mkdir(exist_ok=False);status('preparing')
    plan=prepare();area=base.Areas()
    for directory in (ADAPTIVE/'results_s42_01',NEWROOTS/'results_s42_01'):
        area.old.append(sqlite3.connect(f'file:{directory/"area_store/areas.sqlite"}?mode=ro',uri=True))
    model=load_model();walks=[]
    for ctx in plan['contexts']:
        assert area.store.sources==ctx['area_sources']
        status('loading_inputs',context=ctx['id'])
        inputs=Inputs(ctx['bench'],ctx['root']['l1_size'])
        save(Path('inputs')/f"{ctx['id']}.json",dict(slice_ids=inputs.ids,load_seconds=inputs.load_seconds))
        previous=read(Path(ctx['controls']['hybrid'])/'inputs'/f"{ctx['id']}.json")
        assert inputs.ids==previous['slice_ids']
        aa=area.get(ctx['root']);assert aa is not None and abs(aa-ctx['root_area'])<1e-6
        registry={}
        def record(c,cp,m):
            key=config_key(c)
            if key in registry:assert np.isclose(cp,registry[key],rtol=3e-6)
            registry[key]=float(cp)
        result,_=local_walk.walk(ctx,'local',model,inputs,area,record);walks.append(result)
    paired=[];no_worse=True;strict=False
    for ctx,result in zip(plan['contexts'],walks):
        results={a:read(Path(p)/'walks'/ctx['id']/a/'summary.json') for a,p in ctx['controls'].items()}
        results['local']=result;tol=1e-6*ctx['root_cpi']
        for a in ('hybrid','adaptive'):
            no_worse &= result['final_cpi']<=results[a]['final_cpi']+tol
            strict |= result['final_cpi']<results[a]['final_cpi']-tol
        paired.append(dict(context=ctx['id'],gain_pct={a:100*results[a]['gain'] for a in COMPARISON_ARMS},
            final_cpi={a:results[a]['final_cpi'] for a in COMPARISON_ARMS}))
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    save('summary.json',dict(paired=paired,walks=walks,gate_passed=bool(no_worse and strict),
        gate_no_worse=bool(no_worse),gate_strict_improvement=bool(strict),
        area_counts=dict(area.counts),area_failures=area.store.n_fail,area_errors=area.store.errors,
        source_preservation_passed=True,wall_seconds=time.perf_counter()-START))
    from local_validate import validate
    save('validation.json',validate(OUT))
    from local_report import report
    report(OUT)
    status('complete',gate_passed=bool(no_worse and strict),area_failures=area.store.n_fail)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():
            (OUT/'ERROR.txt').write_text(traceback.format_exc());status('failed')
        raise
