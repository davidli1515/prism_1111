"""Replay signed occupancy, solver values, proposals, predictions and queries."""
import numpy as np
from scipy.special import softmax,expit,logsumexp
from secant_support import HERE,read,sha,config_key,compute_w_min
from core import legal,get_param_values
from matched_support import label_id
from width_priority_validate import cpi


def validate(out):
    plan=read(out/'plan.json');summary=read(out/'summary.json');labels=read(out/'labels.json');batch=read(out/'frozen_batch.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items()) and batch['candidate_labels_observed']==0
    counts=dict(win=0,tie=0,loss=0,unresolved=0);required=set();logical=0
    for ctx,row in zip(plan['contexts'],summary['rows']):
        file=out/'roots'/ctx['id']/'decision.json';assert sha(file)==batch['decision_sha256'][ctx['id']]
        d=read(file);assert d['candidate_labels_observed']==0 and d['area_cap']==ctx['root_area'];assert sha(d['prediction_file'])==d['prediction_sha256']
        root=ctx['root'];width=compute_w_min(root);pool=d['candidates'];basis=d['basis'];names=d['names'];pos={a['parameter'] for a in basis if a['sign']==1}
        assert len(pool)==len({config_key(a['config']) for a in pool})
        with np.load(d['prediction_file']) as z:
            ell=z['root_ell'].astype(float);de=z['logodds_slope'].astype(float);dirs=z['basis_logodds_slope'].astype(float)
            q=softmax(ell,-1);u=logsumexp(ell,-1)[None]+(q[None]*de).sum(-1)
            sc=np.minimum(100.,1/(np.maximum(expit(-u),.01)*width+1e-8))
            l=ell[None]+de;mm=softmax(np.concatenate([np.zeros_like(l[...,:1]),l],-1),-1);mc=cpi(mm,width)
            for arm,st in (('scalar',sc),('mscs',mc)):
                np.testing.assert_allclose(st,z[arm+'_stage'],rtol=4e-6,atol=3e-6)
                np.testing.assert_allclose(st.mean((1,2)),d[arm+'_predicted_cpis'],rtol=4e-6,atol=3e-6)
            low=dict(root);high=dict(root)
            for a in basis:
                k=a['parameter'];low[k]=min(low[k],a['config'][k]);high[k]=max(high[k],a['config'][k])
            assert low==d['supported_low'] and high==d['supported_high']
            for arm,solutions in d['solutions'].items():
                assert len(solutions)==3
                expected_starts=[np.zeros(len(names)),np.array([.25 if high[k]>root[k] else 0 for k in names]),np.array([-.25 if low[k]<root[k] else 0 for k in names])]
                for start_id,s in enumerate(solutions):
                    np.testing.assert_array_equal(s['start'],expected_starts[start_id]);v=dict(zip(names,s['signed_fractions']))
                    for k,val in v.items():assert (-1 if low[k]<root[k] else 0)-1e-9<=val<=(1 if high[k]>root[k] else 0)+1e-9
                    weights=np.array([max(0.,a['sign']*v[a['parameter']]) for a in basis]);physical=dict(root)
                    for k in names:assert sum(weights[j]>0 for j,a in enumerate(basis) if a['parameter']==k)<=1
                    for a,w in zip(basis,weights):physical[a['parameter']]+=w*(a['config'][a['parameter']]-root[a['parameter']])
                    da=np.array(d['basis_areas'])-ctx['root_area'];aslk=-da@weights/ctx['root_area']
                    cslk=physical['numROBEntries']-2*physical['dispatchWidth']-physical['numIQEntries']-physical['LQEntries']-physical['SQEntries']
                    response_delta=np.einsum('r,rnsk->nsk',weights,dirs)
                    if arm=='scalar':
                        b=expit(-(logsumexp(ell,-1)+(q*response_delta).sum(-1)));cp=np.minimum(100.,1/(np.maximum(b,.01)*width+1e-8))
                    else:
                        lhat=ell+response_delta;m=softmax(np.concatenate([np.zeros_like(lhat[...,:1]),lhat],-1),-1);cp=cpi(m,width)
                    np.testing.assert_allclose(cp.mean()/ctx['root_cpi'],s['normalized_predicted_cpi'],rtol=2e-6)
                    np.testing.assert_allclose([aslk,cslk],[s['area_slack'],s['capacity_slack']],atol=2e-9)
                    assert s['admissible']==(aslk>=-1e-7 and cslk>=-1e-7)
                    matching=[p for p in d['proposal_records'] if p['generator']==arm and p['start_id']==start_id]
                    if not s['admissible']:assert not matching;continue
                    for p in matching:
                        cn=dict(root)
                        for k in names:
                            vals=[t for t in get_param_values(k) if low[k]<=t<=high[k]]
                            cn[k]=max(t for t in vals if t<=physical[k]+1e-9) if p['rounding']=='floor' else min(vals,key=lambda t:(abs(t-physical[k]),t))
                        needed=2*cn['dispatchWidth']+cn['numIQEntries']+cn['LQEntries']+cn['SQEntries']
                        cn['numROBEntries']=max(cn['numROBEntries'],needed);assert cn==p['config']
                        assert any(a['config']==cn for a in pool)
        for a in pool:
            c=a['config'];assert legal(c) and compute_w_min(c)==width and all(low[k]<=c[k]<=high[k] for k in root)
            assert a['edits']=={k:dict(before=root[k],after=c[k]) for k in root if c[k]!=root[k]}
        resolved=True
        for arm in ('scalar','mscs'):
            order=sorted(range(len(pool)),key=lambda i:(d[arm+'_predicted_cpis'][i],i));prefix=d['prefixes'][arm];selected=None
            assert [r['index'] for r in prefix]==order[:len(prefix)]
            for j,r in enumerate(prefix):
                assert r['config']==pool[r['index']]['config']
                if r['area'] is None:assert j==len(prefix)-1 and r['error'];resolved=False
                elif r['area']<=ctx['root_area']+1e-9:assert j==len(prefix)-1;selected=r['index']
            assert selected==d['choices'][arm]
            if selected is None:resolved=False
        assert resolved==d['resolved']==row['resolved']
        if not resolved:counts['unresolved']+=1;continue
        best={}
        for arm,i in d['choices'].items():
            c=pool[i]['config'];lid=label_id(ctx['bench'],c);required.add(lid);logical+=1;r=labels[lid]
            assert r['area']==d['prefixes'][arm][-1]['area']<=ctx['root_area']+1e-9
            with np.load(out/'labels'/f'{lid}.npz') as label:
                np.testing.assert_allclose(cpi(label['m'],width).mean(),r['cpi'],rtol=3e-6)
                if r['origin']=='reused_prior_paid_label':
                    with np.load(r['source_file']) as old:np.testing.assert_array_equal(label['m'],old['m'])
            best[arm]=r['cpi'] if r['cpi']<ctx['root_cpi']*(1-1e-6) else ctx['root_cpi'];assert best[arm]==row['outcomes'][arm]['best_cpi']
            np.testing.assert_allclose(row['outcomes'][arm]['gain_pct'],100*(1-best[arm]/ctx['root_cpi']),atol=1e-10)
        gap=(best['scalar']-best['mscs'])/ctx['root_cpi'];status='win' if gap>1e-6 else 'loss' if gap< -1e-6 else 'tie'
        assert row['status']==status;counts[status]+=1
    assert counts==summary['counts'] and required==set(labels) and logical==summary['logical_candidate_queries']
    assert sum(r['origin']=='new_forward' for r in labels.values())==summary['new_candidate_forwards']
    assert sum(r['origin']=='reused_prior_paid_label' for r in labels.values())==summary['reused_candidate_labels']
    gate=counts['loss']==counts['unresolved']==0 and any(r.get('status')=='win' and r['bench']!='602.gcc_s' for r in summary['rows'])
    assert gate==summary['development_gate_passed']
    return dict(passed=True,signed_occupancy_and_constraints_checked=True,all_solver_values_and_roundings_replayed=True,
        complete_endpoint_predictions_replayed=True,common_pool_and_area_prefixes_checked=True,query_budget_checked=True,sources_unchanged=True)


if __name__=='__main__':
    import json
    print(json.dumps(validate(HERE/'exchange_s42_01'),indent=2))
