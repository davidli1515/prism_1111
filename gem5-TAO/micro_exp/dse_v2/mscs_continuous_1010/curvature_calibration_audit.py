"""Causal one-coefficient shrinkage of the finite component Jensen correction.

Diagnostic only: no ranking changes or new labels. Sequential least squares
uses earlier selected candidates of the same policy, not current outcomes.
"""
import json
import numpy as np
from secant_support import HERE,read,sha,save


def main():
    out=HERE/'curvature_calibration_audit_01';out.mkdir(exist_ok=False)
    src=HERE/'active_resources_s42_01';assert read(src/'validation.json')['passed']
    sources={};rows=[]
    plan=read(src/'plan.json')
    save(out/'protocol.json',dict(source=str(src),arms=plan['arms'],benchmarks=plan['benchmarks'],
        rule='C_scalar + alpha*(C_mscs-C_scalar); alpha=clip(OLS(previous normalized gaps, previous normalized residuals),0,1), default1',
        scope='Selected-label causal forecast audit only; not an unbiased accuracy sample or a search experiment',
        continuation_gate='lower mean absolute CPI error than raw MSCS overall and on at least two of three workloads'))
    for bench in plan['benchmarks']:
        for arm in plan['arms']:
            past=[]
            for path in sorted((src/'walks'/bench/arm).glob('decision_*.json')):
                d=read(path);rpath=path.with_name(path.name.replace('decision','round'));r=read(rpath)
                assert d['resolved'];i=d['choice'];root=d['root_cpi'];sc=d['raw_scalar_scores'][i];mc=d['raw_mscs_scores'][i]
                assert mc>=sc-2e-5
                denominator=sum(g*g for g,e in past)
                alpha=float(np.clip(sum(g*e for g,e in past)/denominator,0.,1.)) if denominator>0 else 1.
                forecast=sc+alpha*(mc-sc)
                # Current value is deliberately consumed only after forecast.
                actual=r['trial']['cpi'];gap=(mc-sc)/root;residual=(actual-sc)/root
                rows.append(dict(bench=bench,arm=arm,query=d['step']+1,history_length=len(past),alpha=alpha,
                    scalar=sc,mscs=mc,calibrated=forecast,actual=actual,root=root,
                    normalized_jensen_gap=gap,normalized_residual=residual))
                past.append((gap,residual))
                sources[str(path)]=sha(path);sources[str(rpath)]=sha(rpath)
    by_bench={}
    for bench in plan['benchmarks']:
        rr=[r for r in rows if r['bench']==bench]
        by_bench[bench]={k:float(np.mean([abs(r[k]-r['actual']) for r in rr])) for k in ('scalar','mscs','calibrated')}
    overall={k:float(np.mean([abs(r[k]-r['actual']) for r in rows])) for k in ('scalar','mscs','calibrated')}
    gate=overall['calibrated']<overall['mscs'] and sum(v['calibrated']<v['mscs'] for v in by_bench.values())>=2
    save(out/'records.json',rows)
    save(out/'summary.json',dict(overall_CPI_MAE=overall,per_workload_CPI_MAE=by_bench,
        continuation_gate_passed=bool(gate),rows=len(rows),new_queries=0,
        alpha_below_one=sum(r['alpha']<1 for r in rows),source_sha256=sources))
    # Re-estimate with independent vector operations, strictly excluding row t.
    for bench in plan['benchmarks']:
        for arm in plan['arms']:
            rr=[r for r in rows if r['bench']==bench and r['arm']==arm]
            gaps=np.array([r['normalized_jensen_gap'] for r in rr]);errors=np.array([r['normalized_residual'] for r in rr])
            for t,r in enumerate(rr):
                assert r['history_length']==t
                den=float(gaps[:t]@gaps[:t]);a=float(np.clip((gaps[:t]@errors[:t])/den,0,1)) if den>0 else 1.
                np.testing.assert_allclose(a,r['alpha'],atol=1e-14)
    assert all(sha(p)==h for p,h in sources.items())
    save(out/'validation.json',dict(passed=True,strictly_private_past_only=True,coefficients_replayed=True,
        all72_original_selected_labels_included=len(rows)==72,zero_new_queries=True,sources_unchanged=True))
    print(json.dumps({k:v for k,v in read(out/'summary.json').items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
