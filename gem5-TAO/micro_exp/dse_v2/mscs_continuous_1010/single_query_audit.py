"""Evaluate priority without pretending a shared history is a new full walk."""
from pathlib import Path
import numpy as np
from secant_support import HERE,read,sha,save
from matched_support import outcome
SRC=HERE/'confirmation_s42_01';OUT=HERE/'single_query_audit_s42_01'


def main():
    OUT.mkdir(exist_ok=False);p=read(SRC/'plan.json');assert read(SRC/'validation.json')['passed']
    sources={str(x):sha(x) for x in [Path(__file__),HERE/'single_query_audit_protocol.txt']}
    sources.update({str(x):sha(x) for x in SRC.rglob('*.json')})
    save(OUT/'plan.json',dict(contexts=p['contexts'],source_sha256=sources,new_queries=0))
    choices=[]
    for ctx in p['contexts']:
        for f in sorted((SRC/'walks'/ctx['id']/'hybrid_secant').glob('decision_*.json')):
            d=read(f);ids={a:int(np.argmin(d[a+'_corrected_cpis'])) for a in ('scalar','mscs')}
            assert all(i in d['chosen'] for i in ids.values())
            choices.append(dict(context=ctx['id'],bench=ctx['bench'],step=d['step'],decision=str(f),indices=ids))
    save(OUT/'frozen_choices.json',choices)
    rows=[]
    for r in choices:
        d=read(r['decision']);ctx=next(c for c in p['contexts'] if c['id']==r['context'])
        result=read(Path(r['decision']).with_name(f"round_{r['step']:02}.json"))
        values={a:outcome(d['root_cpi'],ctx['area_cap'],[result['trials'][d['chosen'].index(i)]]) for a,i in r['indices'].items()}
        assert all(x['resolved'] for x in values.values())
        gap=(values['scalar']['best_cpi']-values['mscs']['best_cpi'])/d['root_cpi']
        rows.append(dict(r,outcomes=values,normalized_gap=gap,status='win' if gap>1e-6 else 'loss' if gap< -1e-6 else 'tie'))
    groups={}
    for bench in ['all']+[c['bench'] for c in p['contexts']]:
        rr=[r for r in rows if bench=='all' or r['bench']==bench]
        groups[bench]=dict(wins=sum(r['status']=='win' for r in rr),ties=sum(r['status']=='tie' for r in rr),losses=sum(r['status']=='loss' for r in rr),
                          mean_normalized_gap=float(np.mean([r['normalized_gap'] for r in rr])))
    all_=groups['all'];passed=all_['wins']>all_['losses'] and all_['mean_normalized_gap']>1e-6 and any(r['status']=='win' and r['bench']!='602.gcc_s' for r in rows)
    assert all(sha(f)==h for f,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,groups=groups,gate_passed=passed,new_queries=0,scope='conditional shared-history selected-state audit; not full trajectories'))
    print(__import__('json').dumps(dict(groups=groups,gate_passed=passed),indent=2))


if __name__=='__main__':main()
