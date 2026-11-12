"""All six choices frozen before any non-history labels; fixed raw pair pool."""
import json,time,sqlite3
import numpy as np
import torch
from quadratic_support import HERE,ROOT,read,sha,save,array_save,FeatureCache,LocalModel,load_model,exact,config_key,compute_w_min
from exact_cached_inputs import CachedInputs
from paid_centers_support import predict
from core import cpis
from matched_support import label_id
import run_adaptive as area_runner

SRC=HERE/'pair_walk_s42_01';OUT=HERE/'paid_centers_s42_01'
ARMS=('raw_scalar','raw_mscs','scalar_blend','mscs_blend')


def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=False)
    assert read(SRC/'validation.json')['passed']
    assert read(HERE/'exact_cached_inputs_audit_01/summary.json')['bitwise_equal']
    oldplan=read(SRC/'plan.json');sources=dict(oldplan['source_sha256'])
    for p in [*HERE.glob('paid_centers*.py'),HERE/'paid_centers_protocol.txt',HERE/'exact_cached_inputs.py',
              HERE/'shepard_support.py',HERE/'exact_cached_inputs_audit_01/summary.json',
              *SRC.rglob('*.json'),*SRC.rglob('*.npz')]:sources[str(p)]=sha(p)
    save(OUT/'plan.json',dict(benchmarks=oldplan['benchmarks'],steps=[4,5],arms=ARMS,source_sha256=sources,
        area_cap=oldplan['area_cap'],scope='post-hoc identical-history ranking pilot, frozen raw pair pools'))
    model=load_model();area_runner.OUT=OUT;area=area_runner.Areas()
    for p in sorted(set(ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'))|set(HERE.rglob('area_store/areas.sqlite'))):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    areas={};caches={};locals_={};frozen=[];nlocal=0;builds=0;feature_seconds=0.
    def getarea(c):
        key=config_key(c)
        if key not in areas:
            aa=area.get(c);areas[key]=(aa,area.store.errors.get(area.store.key(c)) if aa is None else None)
        return areas[key]
    for bench in oldplan['benchmarks']:
        cache=FeatureCache(CachedInputs(bench,64));caches[bench]=cache
        folder=SRC/'walks'/bench/'mscs_joint'
        history_configs={label_id(bench,oldplan['root']):oldplan['root']}
        for step in range(6):
            d=read(folder/f'decision_{step:02}.json')
            if step in (4,5):
                center_configs=[history_configs[lid] for lid in d['history_ids']]
                assert len({config_key(c) for c in center_configs})==len(center_configs)
                centers=[]
                for lid,c in zip(d['history_ids'],center_configs):
                    key=(bench,config_key(c))
                    if key not in locals_:
                        local=LocalModel(model,cache.get(c),compute_w_min(c),'local');locals_[key]=local;nlocal+=1
                        with np.load(SRC/'truth'/bench/f'{lid}.npz') as z:np.testing.assert_allclose(local.m.numpy(),z['m'],rtol=3e-6,atol=3e-7)
                    centers.append(locals_[key])
                xs=torch.stack([cache.get(a['config']) for a in d['candidates']])
                pred,arrays=predict(centers,xs,d['widths'])
                scores={a:pred[a].tolist() for a in ('scalar_blend','mscs_blend')}
                scores.update(raw_scalar=d['scalar_scores'],raw_mscs=d['mscs_scores'])
                choices={};prefixes={};resolved=True
                for arm in ARMS:
                    order=sorted(range(len(scores[arm])),key=lambda i:(scores[arm][i],i));prefix=[];choice=None
                    for i in order:
                        aa,error=getarea(d['candidates'][i]['config']);prefix.append(dict(index=i,area=aa,error=error))
                        if aa is None:resolved=False;break
                        if aa<=oldplan['area_cap']+1e-9:choice=i;break
                    choices[arm]=choice;prefixes[arm]=prefix;resolved &= choice is not None
                name=bench+'__'+str(step);snap=OUT/'decisions'/(name+'.npz');array_save(snap,**arrays)
                rec=dict(bench=bench,step=step,source_decision=str(folder/f'decision_{step:02}.json'),history_ids=d['history_ids'],center_configs=center_configs,
                    root=d['root'],root_cpi=d['root_cpi'],candidates=d['candidates'],widths=d['widths'],scores=scores,choices=choices,
                    area_prefixes=prefixes,resolved=bool(resolved),prediction_file=str(snap),prediction_sha256=sha(snap))
                path=OUT/'decisions'/(name+'.json');save(path,rec);frozen.append((path,rec))
            r=read(folder/f'round_{step:02}.json')
            if r['trial']:history_configs[r['trial']['label_id']]=r['trial']['config']
    save(OUT/'freeze.json',dict(decision_sha256={str(p):sha(p) for p,d in frozen},phase='all six states before any non-history labels'))
    labels={};physical=0;reused=0;logical=0;rows=[]
    for path,d in frozen:
        row=dict(bench=d['bench'],step=d['step'],resolved=d['resolved'],arms={})
        if d['resolved']:
            for arm,i in d['choices'].items():
                c=d['candidates'][i]['config'];lid=label_id(d['bench'],c);key=(d['bench'],lid)
                assert lid not in d['history_ids'];logical+=1
                if key not in labels:
                    old=SRC/'truth'/d['bench']/f'{lid}.npz'
                    if old.exists():
                        with np.load(old) as z:mm=torch.from_numpy(z['m'].copy())
                        cp=float(cpis(mm,compute_w_min(c)).mean());origin='old_label_after_freeze';reused+=1
                    else:cp,mm=exact(model,caches[d['bench']].get(c),compute_w_min(c));origin='new_forward';physical+=1
                    labels[key]=cp
                    array_save(OUT/'truth'/d['bench']/f'{lid}.npz',m=mm)
                    save(OUT/'truth'/d['bench']/f'{lid}.json',dict(config=c,origin=origin,source=str(old) if old.exists() else None))
                cp=labels[key];row['arms'][arm]=dict(index=i,label_id=lid,cpi=cp,config=c,gain=max(0.,(d['root_cpi']-cp)/d['root_cpi']))
        rows.append(row)
    resolved=all(r['resolved'] for r in rows);deltas=[];improved=0;winning_workloads=set()
    for r in rows:
        if not r['resolved']:continue
        aa=r['arms'];m=aa['mscs_blend']['gain'];env=max(aa['raw_scalar']['gain'],aa['scalar_blend']['gain'])
        deltas.append(m-env);improved+=m>aa['raw_mscs']['gain']+1e-6
        if m>aa['scalar_blend']['gain']+1e-6:winning_workloads.add(r['bench'])
    gate=resolved and improved>=2 and len(winning_workloads)>=2 and np.mean(deltas)>1e-6
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,pilot_gate_passed=bool(gate),improved_vs_raw_mscs=improved,
        winning_workloads_vs_scalar_blend=sorted(winning_workloads),mean_advantage_vs_scalar_envelope=float(np.mean(deltas)) if deltas else None,
        new_candidate_forwards=physical,reused_old_labels=reused,logical_candidate_queries=logical,
        historical_center_forwards=nlocal,historical_center_backward_outputs=12*nlocal,
        feature_builds=sum(c.builds for c in caches.values()),feature_seconds=sum(c.seconds for c in caches.values()),
        area_counts=dict(area.counts),area_failures=area.store.n_fail,wall_seconds=time.perf_counter()-started))
    from paid_centers_validate import validate
    save(OUT/'validation.json',validate(OUT))
    print(json.dumps(read(OUT/'summary.json'),indent=2))


if __name__=='__main__':main()
