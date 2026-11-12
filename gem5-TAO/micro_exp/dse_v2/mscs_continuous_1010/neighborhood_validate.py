"""Independent numerical replay for every workload, including losses."""
import numpy as np
from scipy.special import softmax,expit,logsumexp
from secant_support import HERE,read,sha,config_key,compute_w_min
from common import SPEC17
from core import legal
from config_dse import DEFAULT_CONFIG
from matched_support import label_id
from width_priority_validate import cpi


from pathlib import Path
from width_priority_support import width_moves
BENCHES=['605.mcf_s','619.lbm_s','654.roms_s']


def expected_context(out):
    meta=read(Path(out).parent/'root_plan.json')
    assert meta['benchmarks']==BENCHES and meta['seed']==42
    assert all(sha(p)==h for p,h in meta['source_sha256'].items())
    catalog=[a for a in width_moves(DEFAULT_CONFIG,1,{config_key(DEFAULT_CONFIG)}) if a['config']['fetchBufferSize']>=16]
    assert catalog==meta['catalog']
    indices=np.random.RandomState(42).choice(len(catalog),size=4,replace=False).tolist()
    assert indices==meta['selected_indices']
    slot=int(Path(out).name.split('_')[-1]);expected=catalog[indices[slot]]
    assert expected==meta['selected'][slot]
    return expected


def validate_leaf(out):
    plan=read(out/'plan.json');summary=read(out/'summary.json');labels=read(out/'labels.json');batch=read(out/'frozen_batch.json');area=read(out/'area_plan.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items()) and batch['candidate_labels_observed']==0
    expected=expected_context(out)
    assert plan['workloads']==BENCHES and plan['root']==expected['config']
    assert [r['bench'] for r in summary['rows']]==BENCHES and [r['bench'] for r in plan['input_coverage']]==BENCHES
    pool=plan['candidates'];root=plan['root'];widths=np.array(plan['widths']);counts=dict(win=0,tie=0,loss=0,unresolved=0);required=set();logical=0
    assert len(pool)==len({config_key(a['config']) for a in pool})
    for i,a in enumerate(pool):
        c=a['config'];assert legal(c) and c['fetchBufferSize']>=16 and c['l1_size']==64 and c['bp_type']==0
        assert widths[i]==compute_w_min(c) and a['edits']=={k:dict(before=root[k],after=c[k]) for k in root if c[k]!=root[k]}
    for row in summary['rows']:
        bench=row['bench'];file=out/'roots'/bench/'decision.json';assert sha(file)==batch['decision_sha256'][bench];d=read(file)
        assert d['candidate_labels_observed']==0
        if 'prediction_file' not in d:assert not d['resolved'] and not row['resolved'];counts['unresolved']+=1;continue
        assert sha(d['prediction_file'])==d['prediction_sha256'] and d['derivative_check']['passed']
        with np.load(d['prediction_file']) as z:
            ell=z['root_ell'].astype(float);de=z['logodds_slope'].astype(float)
            u=logsumexp(ell,-1)[None]+(softmax(ell,-1)[None]*de).sum(-1)
            sc=np.minimum(100.,1/(np.maximum(expit(-u),.01)*widths[:,None,None]+1e-8))
            l=ell[None]+de;mm=softmax(np.concatenate([np.zeros_like(l[...,:1]),l],-1),-1);mc=cpi(mm,widths[:,None,None])
            for arm,st in (('scalar',sc),('mscs',mc)):
                np.testing.assert_allclose(st,z[arm+'_stage'],rtol=4e-6,atol=3e-6)
                np.testing.assert_allclose(st.mean((1,2)),d[arm+'_predicted_cpis'],rtol=4e-6,atol=3e-6)
            rc=cpi(z['root_m'],compute_w_min(root));np.testing.assert_allclose(rc.mean(),d['root_cpi'],rtol=3e-6)
            raccount=(rc[...,None]*z['root_m']).mean((0,1));resolved=True
            for arm in ('scalar','mscs'):
                order=sorted(range(len(pool)),key=lambda i:(d[arm+'_predicted_cpis'][i],i));prefix=d['prefixes'][arm];choice=None
                assert [p['index'] for p in prefix]==order[:len(prefix)]
                for j,p in enumerate(prefix):
                    if p['area'] is None:assert j==len(prefix)-1 and p['error'];resolved=False
                    elif p['area']<=area['area_cap']+1e-9:assert j==len(prefix)-1;choice=p['index']
                assert choice==d['choices'][arm]
                if choice is None:resolved=False
            assert resolved==d['resolved']==row['resolved']
            if not resolved:counts['unresolved']+=1;continue
            best={}
            for arm,i in d['choices'].items():
                c=pool[i]['config'];lid=label_id(bench,c);required.add(lid);logical+=1;r=labels[lid]
                assert r['area']==d['prefixes'][arm][-1]['area']<=area['area_cap']+1e-9
                with np.load(out/'labels'/f'{lid}.npz') as lab:
                    actual=cpi(lab['m'],widths[i]);np.testing.assert_allclose(actual.mean(),r['cpi'],rtol=3e-6)
                    ad=(actual[...,None]*lab['m']).mean((0,1))-raccount
                pd=(mc[i,...,None]*mm[i]).mean((0,1))-raccount
                np.testing.assert_allclose(ad,row['explanations'][arm]['actual_change'],rtol=3e-4,atol=3e-6)
                np.testing.assert_allclose(pd,row['explanations'][arm]['predicted_change'],rtol=3e-4,atol=3e-6)
                np.testing.assert_allclose(ad.sum(),r['cpi']-d['root_cpi'],atol=4e-6)
                best[arm]=r['cpi'] if r['cpi']<d['root_cpi']*(1-1e-6) else d['root_cpi'];assert best[arm]==row['outcomes'][arm]['best_cpi']
                np.testing.assert_allclose(row['outcomes'][arm]['gain_pct'],100*(1-best[arm]/d['root_cpi']),atol=1e-10)
        gap=(best['scalar']-best['mscs'])/d['root_cpi'];status='win' if gap>1e-6 else 'loss' if gap< -1e-6 else 'tie'
        np.testing.assert_allclose(gap,row['normalized_advantage'],atol=1e-12);assert status==row['status'];counts[status]+=1
        assert row['choices_differ']==(d['choices']['scalar']!=d['choices']['mscs'])
    assert counts==summary['counts'] and required==set(labels) and logical==summary['logical_candidate_queries']
    assert len(labels)==summary['new_candidate_forwards']
    np.testing.assert_allclose(summary['mean_normalized_advantage'],np.mean([r['normalized_advantage'] for r in summary['rows'] if r['resolved']]),atol=1e-12)
    return dict(passed=True,all_workloads_in_fixed_order=True,independent_neighbor_selection_checked=True,frozen_choices_before_labels=True,physical_width_checked=True,
        scalar_gradient_checked=True,independent_predictions_and_outcomes_replayed=True,component_accounts_replayed=True,
        equal_candidate_query_budget_checked=True,sources_unchanged=True)


