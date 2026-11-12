"""Predeclared two-root extension; no change to the successful local policy."""
import json
from pathlib import Path
import sqlite3
import sys
import time
import traceback
import numpy as np
from local_support import (HERE,ROOT,ADAPTIVE,HYBRID,NEWROOTS,COMPARISON_ARMS,
    STEPS,TOPK,DEFAULT_CKPT,config_key,legal,moves,Inputs,load_model)
sys.path.insert(0,str(NEWROOTS))
from newroots_support import DEVELOPMENT,collect_configs,bench_of,pareto_mask
import run_adaptive as adaptive_runner
import run_hybrid as hybrid_runner
import local_walk

OUT=HERE/'fresh_s42_01';START=time.perf_counter()
adaptive_runner.OUT=hybrid_runner.OUT=local_walk.OUT=OUT
adaptive_runner.START=hybrid_runner.START=START
save,status,sha,read=adaptive_runner.save,adaptive_runner.status,adaptive_runner.sha,adaptive_runner.load


def prepare():
    from local_validate import validate
    save('development_revalidation.json',validate(HERE/'results_s42_01'))
    previous=read(HERE/'results_s42_01/summary.json');assert previous['gate_passed']
    oldplan=read(HERE/'results_s42_01/plan.json');sources=dict(oldplan['source_sha256'])
    # In particular, the policy and protocol must retain their BEFORE-stage1 hashes.
    assert all(sha(p)==h for p,h in sources.items())
    used=set();inventory_files={}
    for directory in (*DEVELOPMENT,NEWROOTS/'results_s42_01',HERE/'results_s42_01'):
        for path in sorted(directory.rglob('*.json')):
            bench=next((bench_of(p) for p in path.parts if bench_of(p)),None)
            used.update(collect_configs(read(path),bench));inventory_files[str(path)]=sha(path)
    sources.update(inventory_files)
    contexts=[];selection=[]
    for bench in ('602.gcc_s','605.mcf_s'):
        p=ROOT/'mscs_guard_v5/results_full_01'/bench/'s42/initial.json';archive=read(p)
        sources[str(p)]=sha(p);assert archive['checkpoint_sha256']==sha(DEFAULT_CKPT)
        points=np.c_[archive['cpis'],archive['areas']]
        front=np.flatnonzero(pareto_mask(points));front=front[np.argsort(points[front,1],kind='stable')]
        target=(len(front)-1)//2
        valid=[legal(archive['configs'][int(i)]) and (bench,config_key(archive['configs'][int(i)])) not in used for i in front]
        rank=min((j for j,ok in enumerate(valid) if ok),key=lambda j:(abs(j-target),j))
        i=int(front[rank]);c=archive['configs'][i]
        contexts.append(dict(id=f'{bench}_localfresh{rank}',bench=bench,group='fresh_local_extension',
            rank=rank,archive_index=i,root=c,root_cpi=archive['cpis'][i],root_area=archive['areas'][i],
            area_cap=1.10*archive['areas'][i],area_sources=archive['area_sources'],initial_candidates=moves(c),
            controls={a:str(OUT) for a in ('scalar_excess','hybrid','adaptive')}))
        selection.append(dict(bench=bench,archive=str(p),front_archive_indices=front.tolist(),
            target_rank=target,selected_rank=rank,eligible_by_rank=valid))
    save('exclusions.json',dict(source_sha256=inventory_files,
        exposed=[dict(bench=b,config=dict(k)) for b,k in sorted(used)]))
    for p in list(HERE.glob('*.py'))+[HERE/'PROTOCOL.txt']:
        sources[str(p)]=sha(p)
    plan=dict(seed=42,contexts=contexts,arms=COMPARISON_ARMS,steps=STEPS,topk=TOPK,
        max_candidate_verifications=96,source_sha256=sources,root_selection=selection,
        exclusion_file='exclusions.json',exclusion_sha256=sha(OUT/'exclusions.json'),
        policy_sha256=oldplan['source_sha256'][str(HERE/'local_support.py')],
        phase='predeclared fresh-start extension after development gate passed',
        scope='new starts in same two benchmarks and existing source archives; frozen surrogate only')
    assert sha(HERE/'local_support.py')==plan['policy_sha256']
    save('plan.json',plan);return plan


def main():
    OUT.mkdir(exist_ok=False);status('preparing_fresh')
    plan=prepare();area=adaptive_runner.Areas()
    for directory in (ADAPTIVE/'results_s42_01',NEWROOTS/'results_s42_01',HERE/'results_s42_01'):
        area.old.append(sqlite3.connect(f'file:{directory/"area_store/areas.sqlite"}?mode=ro',uri=True))
    model=load_model();walks=[]
    for index,ctx in enumerate(plan['contexts']):
        assert area.store.sources==ctx['area_sources']
        status('loading_inputs',context=ctx['id']);inputs=Inputs(ctx['bench'],ctx['root']['l1_size'])
        save(Path('inputs')/f"{ctx['id']}.json",dict(slice_ids=inputs.ids,load_seconds=inputs.load_seconds))
        aa=area.get(ctx['root']);assert aa is not None and abs(aa-ctx['root_area'])<1e-6
        registry={}
        def record(c,cp,m):
            key=config_key(c)
            if key in registry:assert np.isclose(cp,registry[key],rtol=3e-6)
            registry[key]=float(cp)
        order=list(COMPARISON_ARMS[index:])+list(COMPARISON_ARMS[:index])
        for arm in order:
            runner=local_walk if arm=='local' else adaptive_runner if arm=='adaptive' else hybrid_runner
            result,_=runner.walk(ctx,arm,model,inputs,area,record);walks.append(result)
    paired=[];no_worse=True;strict=False
    for ctx in plan['contexts']:
        rr={w['arm']:w for w in walks if w['context']==ctx['id']};tol=1e-6*ctx['root_cpi']
        for arm in ('hybrid','adaptive'):
            no_worse &= rr['local']['final_cpi']<=rr[arm]['final_cpi']+tol
            strict |= rr['local']['final_cpi']<rr[arm]['final_cpi']-tol
        paired.append(dict(context=ctx['id'],final_cpi={a:rr[a]['final_cpi'] for a in COMPARISON_ARMS},
            gain_pct={a:100*rr[a]['gain'] for a in COMPARISON_ARMS},
            adaptive_vs_scalar_gain_pp=100*(rr['scalar_excess']['final_cpi']-rr['adaptive']['final_cpi'])/ctx['root_cpi']))
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    save('summary.json',dict(paired=paired,walks=walks,gate_passed=bool(no_worse and strict),
        gate_no_worse=bool(no_worse),gate_strict_improvement=bool(strict),development_gate_passed=True,
        area_counts=dict(area.counts),area_failures=area.store.n_fail,area_errors=area.store.errors,
        source_preservation_passed=True,wall_seconds=time.perf_counter()-START))
    from fresh_validate import validate
    save('validation.json',validate(OUT))
    from fresh_report import report
    report(OUT)
    status('complete_fresh',new_candidate_queries=sum(w['counts']['candidate_forward_calls'] for w in walks),
        area_failures=area.store.n_fail)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():
            (OUT/'ERROR.txt').write_text(traceback.format_exc());status('failed_fresh')
        raise
