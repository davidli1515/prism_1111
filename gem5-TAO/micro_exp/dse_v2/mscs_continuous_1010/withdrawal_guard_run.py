"""Frozen forecasts -> six matched rankings; no model until all choices freeze."""
import json,time,sqlite3
import numpy as np
import torch
from quadratic_support import HERE,ROOT,read,sha,save,FeatureCache,load_model,exact,array_save,config_key,compute_w_min
from exact_cached_inputs import CachedInputs
from core import cpis
from matched_support import label_id
from withdrawal_guard_support import guard_scores
import run_adaptive as area_runner

SRC=HERE/'active_resources_s42_01';OUT=HERE/'withdrawal_guard_s42_01'
ARMS=('scalar_raw','mscs_raw','scalar_memory','mscs_memory','scalar_guard','mscs_guard')


def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=False)
    assert read(SRC/'validation.json')['passed'];oldplan=read(SRC/'plan.json')
    sources=dict(oldplan['source_sha256'])
    for p in [*HERE.glob('withdrawal_guard*.py'),HERE/'withdrawal_guard_protocol.txt',*SRC.rglob('*.json'),*SRC.rglob('*.npz')]:sources[str(p)]=sha(p)
    save(OUT/'plan.json',dict(benchmarks=oldplan['benchmarks'],steps=list(range(1,6)),arms=ARMS,area_cap=oldplan['area_cap'],source_sha256=sources))
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in sorted(set(ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'))|set(HERE.rglob('area_store/areas.sqlite'))):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    areas={};decisions=[]
    def getarea(c):
        key=config_key(c)
        if key not in areas:
            aa=area.get(c);areas[key]=(aa,area.store.errors.get(area.store.key(c)) if aa is None else None)
        return areas[key]
    for bench in oldplan['benchmarks']:
        for step in range(1,6):
            src=SRC/'walks'/bench/'mscs_memory'/f'decision_{step:02}.json';d=read(src)
            scores={k+'_raw':d['raw_'+k+'_scores'] for k in ('scalar','mscs')}
            scores.update({k+'_memory':d[k+'_scores'] for k in ('scalar','mscs')})
            for kind in ('scalar','mscs'):
                guarded,mask=guard_scores(d['root'],d['candidates'],scores[kind+'_raw'],scores[kind+'_memory'])
                scores[kind+'_guard']=guarded.tolist()
            choices={};prefixes={};resolved=True
            for arm in ARMS:
                order=sorted(range(len(scores[arm])),key=lambda i:(scores[arm][i],i));prefix=[];choice=None
                for i in order:
                    aa,error=getarea(d['candidates'][i]['config']);prefix.append(dict(index=i,area=aa,error=error))
                    if aa is None:resolved=False;break
                    if aa<=oldplan['area_cap']+1e-9:choice=i;break
                choices[arm]=choice;prefixes[arm]=prefix;resolved &= choice is not None
            rec=dict(bench=bench,step=step,source=str(src),source_sha256=sha(src),withdrawal=mask.tolist(),scores=scores,
                     choices=choices,area_prefixes=prefixes,resolved=bool(resolved))
            path=OUT/'decisions'/f'{bench}__{step}.json';save(path,rec);decisions.append((path,rec,d))
    save(OUT/'freeze.json',dict(decision_sha256={str(p):sha(p) for p,r,d in decisions},phase='all15 states before non-history labels'))
    model=load_model();caches={};labels={};new=0;reused=0;logical=0;rows=[]
    for path,r,d in decisions:
        bench=r['bench'];row=dict(bench=bench,step=r['step'],resolved=r['resolved'],arms={})
        if r['resolved']:
            for arm,i in r['choices'].items():
                c=d['candidates'][i]['config'];lid=label_id(bench,c);key=(bench,lid);assert lid not in d['history_ids'];logical+=1
                if key not in labels:
                    src=SRC/'truth'/bench/f'{lid}.npz'
                    if src.exists():
                        with np.load(src) as z:mm=torch.from_numpy(z['m'].copy())
                        cp=float(cpis(mm,compute_w_min(c)).mean());origin='old_label_after_freeze';reused+=1
                    else:
                        if bench not in caches:caches[bench]=FeatureCache(CachedInputs(bench,64))
                        cp,mm=exact(model,caches[bench].get(c),compute_w_min(c));origin='new_forward';new+=1
                    labels[key]=cp;array_save(OUT/'truth'/bench/f'{lid}.npz',m=mm)
                    save(OUT/'truth'/bench/f'{lid}.json',dict(config=c,origin=origin,source=str(src) if src.exists() else None))
                cp=labels[key];row['arms'][arm]=dict(index=i,label_id=lid,config=c,cpi=cp,gain=max(0.,(d['root_cpi']-cp)/d['root_cpi']))
        rows.append(row)
    advantages=[];against_raw=[];improved=0;winning=set()
    for r in rows:
        if not r['resolved']:continue
        a=r['arms'];g=a['mscs_guard']['gain'];env=max(a[k]['gain'] for k in ('scalar_raw','scalar_memory','scalar_guard'))
        advantages.append(g-env);against_raw.append(g-a['mscs_raw']['gain']);improved+=g>a['mscs_raw']['gain']+1e-6
        if g>env+1e-6:winning.add(r['bench'])
    gate=all(r['resolved'] for r in rows) and improved>=2 and len(winning)>=2 and np.mean(advantages)>1e-6 and np.mean(against_raw)>1e-6
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,pilot_gate_passed=bool(gate),improved_vs_raw_mscs=improved,winning_workloads=sorted(winning),
        mean_advantage_vs_scalar_envelope=float(np.mean(advantages)),mean_advantage_vs_raw_mscs=float(np.mean(against_raw)),
        new_candidate_forwards=new,reused_old_labels=reused,logical_candidate_queries=logical,new_root_jacobians=0,
        area_counts=dict(area.counts),area_failures=area.store.n_fail,feature_builds=sum(c.builds for c in caches.values()),
        feature_seconds=sum(c.seconds for c in caches.values()),wall_seconds=time.perf_counter()-started))
    from withdrawal_guard_validate import validate
    save(OUT/'validation.json',validate(OUT))
    print(json.dumps(read(OUT/'summary.json'),indent=2))


if __name__=='__main__':main()
