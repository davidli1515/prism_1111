"""Post-hoc accuracy audit of already-paid proposals; no candidate model calls."""
import json
import numpy as np
from secant_support import HERE, read, sha, save


def main():
    src=HERE/'pair_walk_s42_01';out=HERE/'pair_walk_error_audit_01'
    out.mkdir(exist_ok=False)
    assert read(src/'validation.json')['passed']
    rows=[];sources={str(src/'validation.json'):sha(src/'validation.json')}
    for bench in read(src/'plan.json')['benchmarks']:
        for arm in ('scalar_joint','mscs_joint'):
            for path in sorted((src/'walks'/bench/arm).glob('decision_*.json')):
                d=read(path);rp=path.with_name(path.name.replace('decision','round'));r=read(rp)
                sources[str(path)]=sha(path);sources[str(rp)]=sha(rp)
                assert d['resolved'] and r['resolved']
                i=d['choice'];c=d['candidates'][i];s=d['scalar_scores'][i];m=d['mscs_scores'][i]
                actual=r['trial']['cpi'];base=d['root_cpi']
                # All quantities are frozen-surrogate CPI, not measured gem5 CPI.
                truth=src/'truth'/bench/(r['trial']['label_id']+'.npz')
                sources[str(truth)]=sha(truth)
                with np.load(d['prediction_file']) as z, np.load(truth) as y:
                    pred=z['predicted_m'][i].astype(float);true=y['m'].astype(float)
                    pell=np.log(np.maximum(pred[...,1:],1e-30))-np.log(np.maximum(pred[...,:1],1e-30))
                    tell=np.log(np.maximum(true[...,1:],1e-30))-np.log(np.maximum(true[...,:1],1e-30))
                    logit_rmse=float(np.sqrt(np.mean((pell-tell)**2)))
                assert m>=s-2e-5  # Shared normalization/clamps preserve Jensen ordering.
                rows.append(dict(bench=bench,arm=arm,step=d['step']+1,kind=c['kind'],
                    edits=c['edits'],root_cpi=base,scalar=s,mscs=m,actual=actual,
                    scalar_error=s-actual,mscs_error=m-actual,
                    scalar_abs_error=abs(s-actual),mscs_abs_error=abs(m-actual),
                    jensen_gap=m-s,logit_rmse=logit_rmse,accepted=r['accepted'],
                    scalar_says_improvement=s<base*(1-1e-6),mscs_says_improvement=m<base*(1-1e-6)))
    groups={}
    for name,selected in [('all',rows)]+[(b,[r for r in rows if r['bench']==b]) for b in read(src/'plan.json')['benchmarks']]:
        groups[name]=dict(n=len(selected),scalar_mae=float(np.mean([r['scalar_abs_error'] for r in selected])),
            mscs_mae=float(np.mean([r['mscs_abs_error'] for r in selected])),
            mscs_more_accurate=sum(r['mscs_abs_error']<r['scalar_abs_error'] for r in selected),
            rejected=sum(not r['accepted'] for r in selected),
            scalar_false_improvement=sum(r['scalar_says_improvement'] and not r['accepted'] for r in selected),
            mscs_false_improvement=sum(r['mscs_says_improvement'] and not r['accepted'] for r in selected))
    save(out/'report.json',dict(rows=rows,groups=groups,source_sha256=sources,
        new_candidate_queries=0,scope='post-hoc, selection-biased audit of paid joint-policy proposals; not an out-of-sample accuracy estimate'))
    assert all(sha(p)==h for p,h in sources.items())
    save(out/'validation.json',dict(passed=True,paid_proposals_only=True,no_new_model_calls=True,source_preservation=True))
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
