"""Independent common-domain pair enumeration, chord algebra and outcome replay."""
import numpy as np
from scipy.special import softmax,logsumexp,expit
from secant_support import HERE,read,sha,config_key,compute_w_min
from core import legal
from pair_priority_support import dense_singles
from matched_support import label_id
from width_priority_validate import cpi


def forecasts(ell,de,widths):
    u=logsumexp(ell,-1)[None]+(softmax(ell,-1)[None]*de).sum(-1)
    w=np.asarray(widths)[:,None,None]
    sc=np.minimum(100.,1/(np.maximum(expit(-u),.01)*w+1e-8))
    ll=ell[None]+de;mm=softmax(np.concatenate([np.zeros_like(ll[...,:1]),ll],-1),-1)
    return sc,cpi(mm,w),mm


def check_prefix(prefix,order,lookup,cap,expected_choice,allow_empty_failure=False):
    assert [p['index'] for p in prefix]==order[:len(prefix)]
    choice=None;unknown=False
    for j,p in enumerate(prefix):
        if p['area'] is None:assert j==len(prefix)-1 and p['error'];unknown=True
        elif p['area']<=cap+1e-9:assert j==len(prefix)-1;choice=p['index']
    assert choice==expected_choice
    if choice is None and not unknown and prefix:assert len(prefix)==len(order)
    return not unknown


def validate(out):
    src=HERE/'active_resources_s42_01';plan=read(out/'plan.json');summary=read(out/'summary.json');batch=read(out/'frozen_batch.json');labels=read(out/'labels.json')
    oldplan=read(src/'plan.json');expected=[dict(id=f'{b}_before_q{s+1}',bench=b,step=s) for b in oldplan['benchmarks'] for s in (4,5)]
    assert plan['contexts']==expected and all(sha(p)==h for p,h in plan['source_sha256'].items()) and batch['non_history_candidate_labels_observed']==0
    required=set();logical=0;pair_count=0;full_count=0;endpoint_count=0
    for ctx,row in zip(expected,summary['rows']):
        file=out/'states'/ctx['id']/'decision.json';d=read(file);assert sha(file)==batch['decision_sha256'][ctx['id']] and d['non_history_candidate_labels_observed']==0
        original=read(src/'walks'/ctx['bench']/'scalar_raw'/f"decision_{ctx['step']:02}.json")
        assert d['root']==original['root'];np.testing.assert_allclose(d['root_cpi'],original['root_cpi'],rtol=3e-6)
        seen={config_key(oldplan['root'])}
        for s in range(ctx['step']):seen.add(config_key(read(src/'walks'/ctx['bench']/'scalar_raw'/f'round_{s:02}.json')['trial']['config']))
        assert seen=={config_key(c) for c in d['seen']}
        audit=read(HERE/'unused_fu_audit_01/summary.json')['by_workload'];frozen=['l2_size','l2_assoc']
        if ctx['bench'] in audit:
            aa=audit[ctx['bench']]
            frozen += [k for k,v in aa['groups'].items() if v['all_counts_present'] and v['all_units_mapped'] and v['zero_issued_runs']==aa['runs']]
        assert frozen==d['frozen_parameters']
        indices=[i for i,a in enumerate(original['candidates']) if all(a['config'][k]==d['root'][k] for k in frozen)]
        singles=[original['candidates'][i] for i in indices];ns=len(singles)
        assert indices==d['source_pool_indices'] and ns==d['singles_count'] and singles==d['candidates'][:ns]
        previous=read(src/'walks'/ctx['bench']/'scalar_raw'/f"round_{ctx['step']-1:02}.json")
        assert previous['next_root']==d['root'];np.testing.assert_allclose(d['area_cap'],previous['next_area'],atol=1e-9)
        assert d['basis_areas']==d['eligible_pair_endpoints']==[]
        basis=[i for i,a in enumerate(singles) if a['kind']=='single_resource']
        used=set(seen)|{config_key(d['root'])};expected_pairs=[]
        for offset,i in enumerate(basis):
            for j in basis[offset+1:]:
                a=singles[i];b=singles[j]
                if a['parameter']==b['parameter'] or a['sign']==b['sign']:continue
                c=dict(d['root']);c[a['parameter']]=a['config'][a['parameter']];c[b['parameter']]=b['config'][b['parameter']]
                if config_key(c) in used or not legal(c):continue
                assert sum(c[k]>d['root'][k] for k in c)==sum(c[k]<d['root'][k] for k in c)==1
                used.add(config_key(c));expected_pairs.append(dict(indices=[i,j],config=c,kind='opposite_resource_exchange',
                    edits={k:dict(before=d['root'][k],after=c[k]) for k in c if c[k]!=d['root'][k]}))
        assert expected_pairs==d['pairs'];pair_count+=len(expected_pairs)
        assert sha(d['prediction_file'])==d['prediction_sha256']
        with np.load(d['prediction_file']) as z:
            with np.load(original['prediction_file']) as old:
                np.testing.assert_array_equal(z['logodds_slope'][:ns],old['logodds_slope'][indices])
                np.testing.assert_allclose(z['root_ell'],old['root_ell'],rtol=3e-6,atol=3e-6)
            ell=z['root_ell'].astype(float);de=z['logodds_slope'].astype(float)
            sc,mc,mm=forecasts(ell,de,d['widths'])
            for kind,st in (('scalar',sc),('mscs',mc)):
                np.testing.assert_allclose(st,z[kind+'_stage'],rtol=4e-6,atol=3e-6)
                np.testing.assert_allclose(st.mean((1,2)),d[kind+'_scores'],rtol=4e-6,atol=3e-6)
            if expected_pairs:
                # Chord operands were originally summed in float32.
                pde=np.stack([(z['logodds_slope'][p['indices'][0]]+z['logodds_slope'][p['indices'][1]]) for p in expected_pairs]).astype(float)
                ps,pm,_=forecasts(ell,pde,[compute_w_min(d['root'])]*len(expected_pairs))
                np.testing.assert_allclose(ps.mean((1,2)),d['pair_scalar_scores'],rtol=4e-6,atol=3e-6)
                np.testing.assert_allclose(pm.mean((1,2)),d['pair_mscs_scores'],rtol=4e-6,atol=3e-6)
        resolved=all(r['area'] is not None for r in d['basis_areas'])
        for kind in ('scalar','mscs'):
            order=sorted(range(len(expected_pairs)),key=lambda i:(d['pair_'+kind+'_scores'][i],i))
            valid=check_prefix(d['pair_prefixes'][kind],order,None,d['area_cap'],d['proposed_pairs'][kind])
            resolved &= valid
        pool=list(singles);info_expected=[]
        for kind,idx in d['proposed_pairs'].items():
            if idx is None:continue
            a=expected_pairs[idx]
            if any(p['config']==a['config'] for p in pool):continue
            info_expected.append((idx,kind,len(pool),a['indices']));pool.append(a)
        assert pool==d['candidates'] and len(info_expected)==len(d['full_pair_info']);full_count+=len(info_expected)
        for rec,exp in zip(d['full_pair_info'],info_expected):
            assert (rec['pair_index'],rec['generator'],rec['pool_index'],rec['indices'])==exp
            assert rec['input_chord_error']>=0 and rec['input_delta_norm']>0
        assert d['widths']==[compute_w_min(a['config']) for a in pool]
        for arm in plan['arms']:
            kind=arm.split('_')[0];allowed=range(ns if arm.endswith('_base') else len(pool));order=sorted(allowed,key=lambda i:(d[kind+'_scores'][i],i))
            valid=check_prefix(d['prefixes'][arm],order,None,d['area_cap'],d['choices'][arm]);resolved &= valid and d['choices'][arm] is not None
        assert resolved==d['resolved']==row['resolved']
        if not resolved:continue
        best={}
        for arm,i in d['choices'].items():
            c=pool[i]['config'];lid=label_id(ctx['bench'],c);required.add(lid);logical+=1;r=labels[lid]
            assert r['area']==d['prefixes'][arm][-1]['area']<=d['area_cap']+1e-9
            with np.load(out/'labels'/f'{lid}.npz') as z:
                np.testing.assert_allclose(cpi(z['m'],compute_w_min(c)).mean(),r['cpi'],rtol=3e-6)
                if r['origin']=='reused_after_freeze':
                    with np.load(r['source_file']) as old:np.testing.assert_array_equal(z['m'],old['m'])
            best[arm]=r['cpi'] if r['cpi']<d['root_cpi']*(1-1e-6) else d['root_cpi'];assert best[arm]==row['outcomes'][arm]['best_cpi']
            np.testing.assert_allclose(row['outcomes'][arm]['gain_pct'],100*(1-best[arm]/d['root_cpi']),atol=1e-10)
        np.testing.assert_allclose(row['advantage_vs_scalar_envelope'],(min(best['scalar_exchange'],best['scalar_base'])-best['mscs_exchange'])/d['root_cpi'],atol=1e-12)
        np.testing.assert_allclose(row['advantage_vs_mscs_base'],(best['mscs_base']-best['mscs_exchange'])/d['root_cpi'],atol=1e-12)
    assert required==set(labels) and logical==summary['logical_candidate_queries']
    assert sum(r['origin']=='new_forward' for r in labels.values())==summary['new_candidate_forwards']
    assert sum(r['origin']=='reused_after_freeze' for r in labels.values())==summary['reused_candidate_labels']
    assert pair_count==summary['chord_pairs_scored'] and full_count==summary['full_pair_inputs_built'] and endpoint_count==summary['common_endpoint_area_accesses']
    rows=summary['rows'];wins=[r for r in rows if r.get('advantage_vs_scalar_envelope',0)>1e-6]
    gate=all(r['resolved'] for r in rows) and len(wins)>=2 and len({r['bench'] for r in wins})>=2 and np.mean([r.get('advantage_vs_scalar_envelope',0) for r in rows])>1e-6 and sum(r.get('advantage_vs_scalar_envelope',0)<-1e-6 for r in rows)<=1 and any(r.get('advantage_vs_mscs_base',0)>1e-6 for r in rows)
    assert bool(gate)==summary['development_gate_passed']
    return dict(passed=True,signed_pair_universe_independently_enumerated=True,shared_frozen_resource_domain_checked=True,root_area_cap_checked=True,
        chord_and_full_endpoint_prediction_algebra_checked=True,common_pool_area_prefix_and_query_budget_checked=True,
        all_outcomes_and_gate_replayed=True,sources_unchanged=True)


if __name__=='__main__':
    import json
    print(json.dumps(validate(HERE/'signed_budget_s42_01'),indent=2))
