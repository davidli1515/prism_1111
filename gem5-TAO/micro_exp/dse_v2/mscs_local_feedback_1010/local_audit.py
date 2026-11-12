"""Causal shadow proposals on historical paths, with explicit missing labels."""
import json
from pathlib import Path
from local_support import LocalAllocator,config_key


def read(p):return json.loads(Path(p).read_text())


def shadow_audit(contexts):
    rows=[]
    for ctx in contexts:
        known={}
        for arm,directory in ctx['controls'].items():
            for file in (Path(directory)/'walks'/ctx['id']/arm).glob('round_*.json'):
                for trial in read(file)['trials']:
                    known[config_key(trial['config'])]=trial
        for arm in ('hybrid','adaptive'):
            directory=Path(ctx['controls'][arm]);policy=LocalAllocator()
            for file in sorted((directory/'walks'/ctx['id']/arm).glob('round_*.json')):
                r=read(file);d=read(directory/r['decision_file'])
                before=policy.snapshot()
                ids,_,ranking=policy.choose(d['candidates'],d['scalar_predicted_cpis'],d['mscs_predicted_cpis'])
                found=[known.get(config_key(d['candidates'][i]['config'])) for i in ids]
                missing=sum(t is None or (t['area'] is None and t['cpi']<r['root_cpi']*(1-1e-6)) for t in found)
                gain=None
                if missing==0:
                    best=min([r['root_cpi']]+[t['cpi'] for t in found if t['area'] is not None
                        and t['area']<=ctx['area_cap']+1e-9 and t['cpi']<r['root_cpi']*(1-1e-6)])
                    gain=100*(1-best/r['root_cpi'])
                rows.append(dict(context=ctx['id'],historical_arm=arm,round=r['step']+1,
                    state_before=before,shadow_chosen=ids,historical_chosen=d['chosen'],
                    changed_candidate_set=set(ids)!=set(d['chosen']),
                    missing_required_labels=missing,shadow_same_state_gain_pct=gain,
                    historical_step_gain_pct=100*(1-r['next_cpi']/r['root_cpi']),
                    scope='shadow at logged state; update from logged queries only; not a new trajectory'))
                policy.observe(r['root_cpi'],r['trials'],
                    [t['scalar_predicted_cpi'] for t in r['trials']],
                    [t['mscs_predicted_cpi'] for t in r['trials']],[t['cpi'] for t in r['trials']])
    return dict(rows=rows,extra_model_queries=0)
