#!/usr/bin/env python3
"""Corrected bounded background experiment; immutable decisions and live status."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/'mscs_bounds_1009'))
from audit import region, config_key, compute_w_min, pareto_mask, SliceOracle, DEFAULT_CKPT, NORM_CACHE
from config_dse import ALL_PARAMS, get_param_values
from calibration import controls, fit_arms, infer

OUT = HERE/'results_s42_01'
DATA = ROOT/'mscs_guard_v5/results_full_01'
OLD = ROOT/'mscs_reachable_min_1009/results_s42_01'
BOUNDS = ROOT/'mscs_bounds_1009/results_s42_01'
FE = ('fetchBufferSize','l1i_assoc')
START = time.monotonic()
SOURCES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    SOURCES[str(path)] = sha(path)
    return json.loads(Path(path).read_text())


def save(name,obj):
    with (OUT/name).open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False)


def status(phase,**kw):
    value={'phase':phase,'pid':os.getpid(),'elapsed_seconds':time.monotonic()-START,**kw}
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(OUT/'status.json')
    print(json.dumps(value),flush=True)


def near(a,b):
    return sum(a[k]!=b[k] for k in a if k not in FE)<=1


def group_ids(rows):
    parents=list(range(len(rows)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i in range(len(rows)):
        for j in range(i):
            if near(rows[i]['root'],rows[j]['root']):parents[root(i)]=root(j)
    ids=[root(i) for i in range(len(rows))]
    remap={v:i for i,v in enumerate(sorted(set(ids)))}
    return np.array([remap[v] for v in ids])


def features(c,m):
    m=np.asarray(m,np.float32).reshape(-1,4,4)
    assert m.shape[0]==50 and np.allclose(m.sum(2),1.,atol=2e-6)
    cs=np.minimum(1/(np.maximum(m[:,:,0],.01)*compute_w_min(c)+1e-8),100.).astype(float)
    ci=cs.mean(1);cands,_=region(c)
    assert cands
    maxima={k:max(x[k] for x in cands) for k in FE}
    compact=np.array([np.log1p(ci.mean()),ci.std()/ci.mean(),
        np.log(np.quantile(ci,.9)/np.quantile(ci,.1)),np.mean(np.ptp(cs,axis=1)/ci),
        np.log2(compute_w_min(c)),np.log2(c['l1_size']),
        np.log2(c['fetchBufferSize']),np.log2(c['l1i_assoc']),
        np.log2(maxima[FE[0]]/c[FE[0]]),np.log2(maxima[FE[1]]/c[FE[1]]),np.log1p(len(cands))])
    arch=np.array([get_param_values(k).index(c[k])/(len(get_param_values(k))-1) for k in ALL_PARAMS])
    rich=np.r_[compact,arch,np.quantile(np.log1p(cs),np.linspace(0,1,50),axis=0).ravel()]
    p=m[:,:,1:].astype(float);comp=p/p.sum(2,keepdims=True)
    extra=np.array([np.sum(ci*comp[:,s,k])/ci.sum() for s,k in ((0,0),(3,0),(0,2),(3,1))])
    return compact,rich,extra


def arrays(rows):
    f=[features(r['root'],r['mscs']) for r in rows]
    return tuple(np.array([x[k] for x in f]) for k in range(3))


def development():
    plan=load(OLD/'plan.json');vectors=load(OLD/'roots.json')
    tr=load(OLD/'training_labels.json');te=load(OLD/'test_labels.json')
    rows=[]
    for split, contexts, labels in (('train',plan['train'],tr),('test',plan['test'],te)):
        vv=[v for v in vectors if v['split']==split]
        for c,v,l in zip(contexts,vv,labels):
            cp=l.get('descendant_cpis')
            if cp is None:cp=[x['cpi'] for x in l['records']]
            assert v['config']==c['root']
            rows.append(dict(c,mscs=v['mscs'],slice_ids=v['slice_ids'],descendant_cpis=cp,
                             gain=max(0.,1-min(cp)/c['root_cpi'])))
    old_dec=load(BOUNDS/'decisions.json')['contexts']
    for c in old_dec:
        if not c['candidates']:continue
        labels=load(BOUNDS/f"audit_{c['id']}.json")['labels']
        rows.append({'bench':c['bench'],'root':c['root'],'root_cpi':c['root_point'][0],
                     'candidates':c['candidates'],'mscs':c['mscs'],'slice_ids':c['slice_ids'],
                     'descendant_cpis':[x['cpi'] for x in labels],
                     'gain':max(0.,1-min(x['cpi'] for x in labels)/c['root_point'][0])})
    assert len({(r['bench'],config_key(r['root'])) for r in rows})==len(rows)
    for r in rows:
        assert region(r['root'])[0]==r['candidates']
        assert len(r['descendant_cpis'])==len(r['candidates'])
    return rows,[c['root'] for c in old_dec]


def choose_tests(develop,old_roots):
    tests=[]
    forbidden=[r['root'] for r in develop]+old_roots
    for bench in ('602.gcc_s','605.mcf_s'):
        d=load(DATA/bench/'s42/initial.json')
        assert d['checkpoint_sha256']==sha(DEFAULT_CKPT)
        front=set(np.flatnonzero(pareto_mask(np.c_[d['cpis'],d['areas']])).tolist())
        already_known={config_key(c) for c in d['configs']}
        for pattern in ('buffer','assoc','joint'):
            eligible=[]
            for i,c in enumerate(d['configs']):
                if any(near(c,o) for o in forbidden):continue
                cands,_=region(c)
                if not 2<=len(cands)<=5:continue
                if any(config_key(v) in already_known for v in cands):continue
                nf=len({c['fetchBufferSize']}|{v['fetchBufferSize'] for v in cands})
                na=len({c['l1i_assoc']}|{v['l1i_assoc'] for v in cands})
                kind='joint' if nf>1 and na>1 else 'buffer' if nf>1 else 'assoc'
                if kind!=pattern:continue
                eligible.append((i,c,cands))
            assert eligible, (bench,pattern)
            preferred=[r for r in eligible if r[0] in front] or eligible
            preferred.sort(key=lambda r:(d['areas'][r[0]],r[0]))
            i,c,cands=preferred[(len(preferred)-1)//2]
            tests.append({'bench':bench,'pattern':pattern,'source_index':i,'root':c,
                          'root_cpi':d['cpis'][i],'root_area':d['areas'][i],
                          'was_on_BO_front':i in front,'candidates':cands,
                          'all_descendants_unseen_at_BO':True})
            forbidden.append(c)
    assert len(tests)==6 and sum(len(r['candidates']) for r in tests)<=30
    return tests


def metric(y,p):
    y,p=np.array(y),np.array(p)
    return {'mae':float(np.mean(abs(y-p))),'max_underprediction':float(np.maximum(y-p,0).max()),
            'errors':abs(y-p).tolist(),'predictions':p.tolist()}


def main():
    OUT.mkdir(exist_ok=False)
    status('checking_controls')
    control=controls();save('controls.json',control)
    if not control['positive_pass'] or not control['negative_pass']:
        status('stopped_control_failed',new_candidate_queries=0);return
    status('preparing_fixed_plan',controls_passed=True)
    dev,old_roots=development();groups=group_ids(dev)
    tests=choose_tests(dev,old_roots)
    for p in (HERE/'run.py',HERE/'calibration.py',HERE/'test_calibration.py',HERE/'PROTOCOL.txt',
              HERE/'supervise.py',Path(DEFAULT_CKPT),ROOT/'mscs_bounds_1009/audit.py',
              ROOT/'mscs_response_1008/probe.py',ROOT/'mscs_response_1008/response.py',
              ROOT/'common.py',ROOT.parent/'dse_exp/config_dse.py',ROOT.parent/'dse_exp/dse_engine.py'):
        SOURCES[str(p)]=sha(p)
    for n in ('mean','std'):
        p=Path(NORM_CACHE)/f'18k_s42_{n}.npy';SOURCES[str(p)]=sha(p)
    plan={'seed':42,'development':dev,'development_groups':groups.tolist(),'test':tests,
          'source_sha256':dict(SOURCES),'max_new_candidate_queries':30,'max_root_queries':6,
          'patterns':['buffer','assoc','joint']}
    save('plan.json',plan)
    status('fitting_development',development_rows=len(dev),independent_components=len(set(groups)),
           new_sets=len(tests),candidate_count=sum(len(t['candidates']) for t in tests))
    compact,rich,extra=arrays(dev);y=np.array([r['gain'] for r in dev]);benches=[r['bench'] for r in dev]
    arms=fit_arms(compact,rich,extra,y,groups,benches)
    devreport={k:v for k,v in arms.items() if k!='models'}
    # Additional transfer diagnostics use only previously examined data.
    transfer=[]
    for b in sorted(set(benches)):
        va=np.array([i for i,r in enumerate(dev) if r['bench']==b])
        blocked=set(groups[va]);tr=np.array([i for i in range(len(dev)) if groups[i] not in blocked])
        if len(set(groups[tr]))<3:
            transfer.append({'bench':b,'status':'insufficient_purged_groups'});continue
        fit=fit_arms(compact[tr],rich[tr],extra[tr],y[tr],groups[tr],[benches[i] for i in tr])
        pp,_=infer(fit,compact[va],rich[va],extra[va],[benches[i] for i in va],np.ones(len(va)))
        transfer.append({'bench':b,'train_n':len(tr),'test_n':len(va),
                         'metrics':{k:metric(y[va],v) for k,v in pp.items()}})
    save('development_fit.json',dict(devreport,transfer_diagnostics=transfer,
                                     target_range=[float(y.min()),float(y.max())]))
    status('extracting_fresh_roots',scalar_representation=arms['representation'])
    oracles={};root_records=[]
    for i,t in enumerate(tests):
        b=t['bench']
        if b not in oracles:oracles[b]=SliceOracle(b)
        m,cpi,ids=oracles[b].query(t['root'])
        assert np.isclose(cpi,t['root_cpi'],rtol=3e-6)
        root_records.append(dict(t,mscs=m.tolist(),slice_ids=list(ids)))
        status('extracting_fresh_roots',completed=i+1,total=6)
    save('fresh_roots.json',root_records)
    xc,xr,xm=arrays(root_records)
    cap=np.array([max(0.,1-1/(compute_w_min(r['root'])*r['root_cpi'])) for r in tests])
    preds,perm=infer(arms,xc,xr,xm,[r['bench'] for r in tests],cap)
    save('predictions.json',{'phase':'before_fresh_descendant_labels','predictions':preds,
                             'test_permutation':perm,'scalar_representation':arms['representation']})
    frozen=sha(OUT/'predictions.json')
    status('evaluating_fresh_sets',predictions_frozen=True)
    # Descendant labels from old archives become accessible only now.
    cache={}
    for b in oracles:
        cache[b]={}
        for name in ('initial','scalar','scalar_guard','mscs_guard'):
            d=load(DATA/b/'s42'/f'{name}.json')
            assert d['checkpoint_sha256']==sha(DEFAULT_CKPT)
            for c,v in zip(d['configs'],d['cpis']):
                key=config_key(c)
                if key in cache[b]:assert np.isclose(cache[b][key][0],v,rtol=3e-6)
                cache[b][key]=(v,name)
    counts=Counter();labels=[];new_keys=set()
    for i,t in enumerate(tests):
        values=[];b=t['bench']
        for c in t['candidates']:
            key=config_key(c)
            if key in cache[b]:v,source=cache[b][key];counts['reused_candidate_cpi']+=1
            else:
                assert len(new_keys)<30
                _,v,ids=oracles[b].query(c)
                assert list(ids)==root_records[i]['slice_ids']
                new_keys.add((b,key));counts['new_candidate_cpi']+=1;source='new_frozen_checkpoint'
            values.append({'config':c,'cpi':float(v),'source':source})
        gain=max(0.,1-min(v['cpi'] for v in values)/t['root_cpi'])
        labels.append({'bench':b,'pattern':t['pattern'],'root':t['root'],'root_cpi':t['root_cpi'],
                       'gain':gain,'records':values})
        status('evaluating_fresh_sets',completed=i+1,total=6,counts=dict(counts))
    save('fresh_labels.json',labels)
    truth=[r['gain'] for r in labels];metrics={k:metric(truth,v) for k,v in preds.items()}
    sm,ss,sp=[metrics[k] for k in ('mscs','scalar','shuffled')]
    wins=sum(a<min(b,c)-1e-9 for a,b,c in zip(sm['errors'],ss['errors'],sp['errors']))
    signal=bool(ss['mae']>1e-6 and sm['mae']<=.9*min(ss['mae'],sp['mae']) and wins>=4 and
                sm['max_underprediction']<=ss['max_underprediction']+1e-6)
    assert sha(OUT/'predictions.json')==frozen
    assert all(sha(p)==h for p,h in SOURCES.items())
    assert len(new_keys)<=30 and sum(o.n_queries for o in oracles.values())<=36
    # Independent direct checks of the test set semantics and numeric reductions.
    assert all(not near(t['root'],d['root']) for t in tests for d in dev)
    for t,l in zip(tests,labels):
        assert [r['config'] for r in l['records']]==region(t['root'])[0]
        assert all(compute_w_min(c)==compute_w_min(t['root']) for c in t['candidates'])
        assert all(all(c[k]==t['root'][k] for k in c if k not in FE) for c in t['candidates'])
    result={'metrics':metrics,'truth':truth,'provisional_signal':signal,'consistent_wins':int(wins),
            'counts':dict(counts),'physical_forward_configs':sum(o.n_queries for o in oracles.values()),
            'development_rows':len(dev),'development_groups':len(set(groups)),'fresh_sets':6,
            'scalar_representation':arms['representation'],'predictions_sha256':frozen,
            'source_sha256':SOURCES,'preservation_and_intent_checks_passed':True,
            'wall_seconds':time.monotonic()-START,
            'scope':'small same-workload fresh-configuration confirmation; no safety or DSE saving claim'}
    save('summary.json',result)
    lines=['Corrected small reachable-gain experiment, seed 42',
           f'Positive control passed: {control["positive_pass"]}; negative control passed: {control["negative_pass"]}',
           f'Development: {len(dev)} rows, {len(set(groups))} connected groups, four benchmarks.',
           f'Scalar representation selected using development CV: {arms["representation"]}',
           f'Fresh tests: six sets, patterns buffer / assoc / joint on gcc and mcf.',
           f'Provisional positive signal: {signal}; consistent wins {wins}/6.',
           'MAE in relative CPI-gain percentage points:']
    lines += [f'  {k}: {100*v["mae"]:.6f}' for k,v in metrics.items()]
    lines += ['Per set: benchmark, pattern, true gain%, scalar prediction%, MSCS prediction%']
    lines += [f"  {r['bench']} {r['pattern']} {100*r['gain']:.6f} {100*preds['scalar'][i]:.6f} {100*preds['mscs'][i]:.6f}" for i,r in enumerate(labels)]
    lines += [f'Counts: {dict(counts)}. Total physical forward configurations: {result["physical_forward_configs"]}.',
              'Passing the control only shows the pipeline can detect the specified synthetic signal.',
              'The real result concerns this calibrated predictor and six finite sets. It does not prove',
              'MSCS universally useful/useless, a safe bound, or savings in full DSE. No automatic expansion.']
    (OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n')
    status('complete',provisional_signal=signal,counts=dict(counts),summary=str(OUT/'summary.json'))


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():status('failed',traceback=traceback.format_exc())
        raise
