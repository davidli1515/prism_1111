"""Exploratory Pareto-archive audit, zero new queries; original gates unchanged."""
import itertools,json
import numpy as np
from secant_support import HERE,read,sha,save


def hv_strip(points,ref=(1.1,1.)):
    points=[(a,c) for a,c in points if a<ref[0] and c<ref[1]]
    xs=sorted(set([a for a,c in points]+[ref[0]]))
    return sum((b-a)*(ref[1]-min([c for x,c in points if x<=a]+[ref[1]])) for a,b in zip(xs,xs[1:]))


def hv_inclusion(points,ref=(1.1,1.)):
    points=[(a,c) for a,c in points if a<ref[0] and c<ref[1]]
    total=0.
    for k in range(1,len(points)+1):
        for subset in itertools.combinations(points,k):
            total+=(-1)**(k+1)*(ref[0]-max(p[0] for p in subset))*(ref[1]-max(p[1] for p in subset))
    return total


def main():
    out=HERE/'pair_archive_audit_01';out.mkdir(exist_ok=False)
    inputs=[HERE/'pair_walk_s42_01',HERE/'pair_replication_s42_01/root_0',HERE/'pair_replication_s42_01/root_1']
    rows=[];sources={}
    for src in inputs:
        assert read(src/'validation.json')['passed']
        plan=read(src/'plan.json');sources[str(src/'plan.json')]=sha(src/'plan.json')
        for b in plan['benchmarks']:
            curves={}
            for arm in plan['arms']:
                folder=src/'walks'/b/arm;summ=read(folder/'summary.json');assert summ['queries']==6
                sources[str(folder/'summary.json')]=sha(folder/'summary.json')
                points=[(1.,1.)];curve=[]
                for step in range(6):
                    path=folder/f'round_{step:02}.json';r=read(path);sources[str(path)]=sha(path)
                    assert r['resolved'];t=r['trial'];assert t['area']<=plan['area_cap']+1e-9
                    points.append((t['area']/plan['root_area'],t['cpi']/summ['initial_cpi']))
                    strip=hv_strip(points);independent=hv_inclusion(points)
                    np.testing.assert_allclose(strip,independent,rtol=1e-11,atol=1e-13)
                    curve.append(strip)
                curves[arm]=curve
            sc=np.max([v for a,v in curves.items() if a.startswith('scalar')],axis=0)
            ms=np.array(curves['mscs_joint']);end=ms[-1]-sc[-1];avg=float(np.mean(ms-sc))
            rows.append(dict(experiment=str(src),bench=b,curves=curves,scalar_envelope=sc.tolist(),
                endpoint_advantage=float(end),anytime_advantage=avg,
                endpoint_status='win' if end>1e-10 else 'loss' if end< -1e-10 else 'tie'))
    save(out/'summary.json',dict(rows=rows,source_sha256=sources,new_queries=0,
        reference='area=1.10*initial area, CPI=initial CPI, normalize each by its initial value',
        scope='post-hoc secondary objective: all paid feasible candidates form each private Pareto archive; does NOT change failed CPI gates'))
    assert all(sha(p)==h for p,h in sources.items())
    save(out/'validation.json',dict(passed=True,rectangle_union_checked_by_inclusion_exclusion=True,all_paid_points_included=True,sources_unchanged=True))
    print(json.dumps([dict(experiment=r['experiment'],bench=r['bench'],endpoint_status=r['endpoint_status'],
                           endpoint_advantage=r['endpoint_advantage'],anytime_advantage=r['anytime_advantage']) for r in rows],indent=2))


if __name__=='__main__':main()
