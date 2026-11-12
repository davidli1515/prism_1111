"""Separate response information from generic two-predictor diversity."""
from pathlib import Path
import numpy as np
from secant_support import HERE,read,sha,save,compute_w_min,config_key
from matched_support import label_id,outcome
from adaptive_validate import stage_cpi,close

SRC=HERE/'confirmation_s42_01';OUT=HERE/'ensemble_audit_s42_01'


def chosen(anchor,backup):
    a=int(np.argmin(anchor));return [a]+[i for i in np.argsort(backup,kind='stable').tolist() if i!=a][:1]


def main():
    OUT.mkdir(exist_ok=False);plan=read(SRC/'plan.json')
    sources={str(p):sha(p) for p in [Path(__file__),HERE/'ensemble_audit_protocol.txt']}
    sources.update({str(p):sha(p) for p in SRC.rglob('*') if p.is_file() and p.suffix in ('.json','.npz')})
    save(OUT/'plan.json',dict(contexts=plan['contexts'],source_sha256=sources,new_queries=0))
    records=[];proof=[]
    for ctx in plan['contexts']:
        first=None
        for path in sorted((SRC/'walks'/ctx['id']/'hybrid_secant').glob('decision_*.json')):
            d=read(path);anchor=d['scalar_corrected_cpis']
            options={'scalar_ensemble':chosen(anchor,d['scalar_predicted_cpis']),
                     'simple_mscs':chosen(anchor,d['mscs_predicted_cpis']),
                     'full_mscs':chosen(anchor,d['mscs_corrected_cpis'])}
            assert options['full_mscs']==d['chosen']
            if first is None and options['simple_mscs']!=d['chosen']:first=d['step']
            records.append(dict(context=ctx['id'],bench=ctx['bench'],step=d['step'],decision=str(path),choices=options))
        proof.append(dict(context=ctx['id'],first_divergence=first,whole_path_equivalent=first is None))
    save(OUT/'frozen_choices.json',dict(records=records,proof=proof))
    # Only now inspect all previously paid labels. They are audit outcomes,
    # not histories that these hypothetical selectors were allowed to learn from.
    labels={}
    for file in sorted((SRC/'walks').glob('*/*/round_*.json')):
        ctx=next(c for c in plan['contexts'] if c['id']==file.parts[-3])
        for t in read(file)['trials']:
            lid=(ctx['id'],label_id(ctx['bench'],t['config']))
            if lid in labels:
                close(t['cpi'],labels[lid]['cpi'])
                if t['area'] is not None and labels[lid]['area'] is not None:close(t['area'],labels[lid]['area'])
            if lid not in labels or t['area'] is not None:labels[lid]=t
    rows=[]
    for r in records:
        ctx=next(c for c in plan['contexts'] if c['id']==r['context']);d=read(r['decision']);answers={};expl={}
        for arm,indices in r['choices'].items():
            ids=[(r['context'],label_id(ctx['bench'],d['candidates'][i]['config'])) for i in indices]
            if any(l not in labels for l in ids):answers[arm]=dict(resolved=False,missing_labels=[l[1] for l in ids if l not in labels]);continue
            answers[arm]=outcome(d['root_cpi'],ctx['area_cap'],[labels[l] for l in ids])
        with np.load(SRC/d['prediction_file']) as z:
            width=compute_w_min(d['root']);root=z['root_m'];root_pc=(stage_cpi(root,width)[...,None]*root).mean((0,1))
            close(root_pc.sum(),d['root_cpi'])
            for i in sorted(set(sum(r['choices'].values(),[]))):
                a=d['candidates'][i];key=(r['context'],label_id(ctx['bench'],a['config']))
                if key not in labels:continue
                with np.load(SRC/'truth'/ctx['id']/f'{key[1]}.npz') as truth:actual=truth['m']
                pred=z['predicted_m'][i]
                change={name:((stage_cpi(mm,width)[...,None]*mm).mean((0,1))-root_pc).tolist() for name,mm in [('predicted',pred),('actual',actual)]}
                close(sum(change['actual']),labels[key]['cpi']-d['root_cpi'])
                expl[str(i)]=dict(action=a,pressure_order=['Base','FP','BP','SP'],cpi_change=change,
                    raw_scalar_cpi=d['scalar_predicted_cpis'][i],corrected_scalar_cpi=d['scalar_corrected_cpis'][i],
                    raw_mscs_cpi=d['mscs_predicted_cpis'][i],corrected_mscs_cpi=d['mscs_corrected_cpis'][i],
                    actual_cpi=labels[key]['cpi'],area=labels[key]['area'])
        rows.append(dict(r,outcomes=answers,explanations=expl))
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,proof=proof,new_queries=0,
        scope='conditional shared-history audit, full path equivalence only before first divergence'))
    print(__import__('json').dumps(proof,indent=2))
    for r in rows:
        if len({tuple(v) for v in r['choices'].values()})>1:
            print(r['context'],r['step'],r['choices'],{a:x.get('gain_pct','unknown') for a,x in r['outcomes'].items()})


if __name__=='__main__':main()
