"""All existing-context decisions freeze before selected candidate labels."""
import json,time,sqlite3
import numpy as np
import torch
from quadratic_support import HERE,ROOT,read,sha,save,FeatureCache,load_model,exact,array_save,config_key,compute_w_min
from exact_cached_inputs import CachedInputs
from core import cpis
from matched_support import label_id
from reclaim_veto_support import ranking,ARMS
import run_adaptive as area_runner

SOURCES=[HERE/'active_resources_s42_01',HERE/'active_confirmation_s42_01/root_0',HERE/'active_confirmation_s42_01/root_1']
OUT=HERE/'reclaim_veto_s42_01'


def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=False)
    for src in SOURCES:assert read(src/'validation.json')['passed']
    paths=[*HERE.glob('reclaim_veto*.py'),HERE/'reclaim_veto_protocol.txt']
    paths += [p for src in SOURCES for p in src.rglob('*.json')]
    sources={str(p):sha(p) for p in paths}
    save(OUT/'plan.json',dict(source_directories=list(map(str,SOURCES)),arms=ARMS,steps=list(range(6)),source_sha256=sources))
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in sorted(set(ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'))|set(HERE.rglob('area_store/areas.sqlite'))):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    areas={};decisions=[]
    def getarea(c):
        key=config_key(c)
        if key not in areas:
            aa=area.get(c);areas[key]=(aa,area.store.errors.get(area.store.key(c)) if aa is None else None)
        return areas[key]
    for slot,src in enumerate(SOURCES):
        oldplan=read(src/'plan.json')
        for bench in oldplan['benchmarks']:
            for step in range(6):
                path=src/'walks'/bench/'scalar_memory'/f'decision_{step:02}.json';d=read(path)
                orders={};blocks={};choices={};prefixes={};resolved=True
                for arm in ARMS:
                    order,blocked,mask=ranking(d['root'],d['root_cpi'],d['candidates'],d['raw_scalar_scores'],d['scalar_scores'],d['raw_mscs_scores'],arm)
                    prefix=[];choice=None
                    for i in order:
                        aa,error=getarea(d['candidates'][i]['config']);prefix.append(dict(index=i,area=aa,error=error))
                        if aa is None:resolved=False;break
                        if aa<=oldplan['area_cap']+1e-9:choice=i;break
                    choices[arm]=choice;prefixes[arm]=prefix;orders[arm]=order;blocks[arm]=blocked;resolved &= choice is not None
                rec=dict(slot=slot,bench=bench,step=step,source=str(path),source_sha256=sha(path),area_cap=oldplan['area_cap'],
                    choices=choices,orders=orders,blocked=blocks,withdrawal=mask,area_prefixes=prefixes,resolved=bool(resolved))
                target=OUT/'decisions'/f'{slot}__{bench}__{step}.json';save(target,rec);decisions.append((target,rec,d))
    assert len(decisions)==54
    save(OUT/'freeze.json',dict(decision_sha256={str(p):sha(p) for p,r,d in decisions},phase='all54 contexts before non-history labels'))
    model=load_model();caches={};labels={};new=0;reused=0;logical=0;rows=[]
    for path,r,d in decisions:
        bench=r['bench'];row=dict(slot=r['slot'],bench=bench,step=r['step'],resolved=r['resolved'],arms={})
        if r['resolved']:
            for arm,i in r['choices'].items():
                c=d['candidates'][i]['config'];lid=label_id(bench,c);key=(bench,lid);assert lid not in d['history_ids'];logical+=1
                if key not in labels:
                    old=next((s/'truth'/bench/f'{lid}.npz' for s in SOURCES if (s/'truth'/bench/f'{lid}.npz').exists()),None)
                    if old is not None:
                        with np.load(old) as z:mm=torch.from_numpy(z['m'].copy())
                        cp=float(cpis(mm,compute_w_min(c)).mean());origin='old_label_after_freeze';reused+=1
                    else:
                        if bench not in caches:caches[bench]=FeatureCache(CachedInputs(bench,64))
                        cp,mm=exact(model,caches[bench].get(c),compute_w_min(c));origin='new_forward';new+=1
                    labels[key]=cp;array_save(OUT/'truth'/bench/f'{lid}.npz',m=mm)
                    save(OUT/'truth'/bench/f'{lid}.json',dict(config=c,origin=origin,source=str(old) if old else None))
                cp=labels[key];row['arms'][arm]=dict(index=i,label_id=lid,config=c,cpi=cp,
                    gain=max(0.,(d['root_cpi']-cp)/d['root_cpi']),blocked_fallback=r['blocked'][arm][i])
        rows.append(row)
    delta=[];wins=losses=0
    for r in rows:
        if not r['resolved']:continue
        a=r['arms'];e=max(a[k]['gain'] for k in ('scalar_raw','scalar_memory','scalar_veto'))
        dd=a['mscs_veto']['gain']-e;delta.append(dd);wins+=dd>1e-6;losses+=dd< -1e-6
    gate=all(r['resolved'] for r in rows) and wins>=2 and losses==0 and np.mean(delta)>1e-6
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,pilot_gate_passed=bool(gate),wins=wins,losses=losses,
        mean_advantage_vs_scalar_envelope=float(np.mean(delta)),new_candidate_forwards=new,reused_old_labels=reused,
        logical_candidate_queries=logical,new_ranking_model_forwards=0,new_root_jacobians=0,
        area_counts=dict(area.counts),area_failures=area.store.n_fail,feature_builds=sum(c.builds for c in caches.values()),
        feature_seconds=sum(c.seconds for c in caches.values()),wall_seconds=time.perf_counter()-started))
    from reclaim_veto_validate import validate
    save(OUT/'validation.json',validate(OUT))
    print(json.dumps({k:v for k,v in read(OUT/'summary.json').items() if k!='rows'},indent=2))


if __name__=='__main__':main()
