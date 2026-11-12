"""Independent algebra, width, area-prefix, explanation and budget replay."""
import numpy as np
from scipy.special import softmax,logsumexp,expit
from secant_support import HERE,read,sha,config_key,compute_w_min
from core import legal,moves
from config_dse import get_param_values
from matched_support import label_id


def cpi(m,w):return np.minimum(100.,1/(np.maximum(m[...,0],.01)*w+1e-8))


def validate(out):
    plan=read(out/'plan.json');summary=read(out/'summary.json');labels=read(out/'labels.json');batch=read(out/'frozen_batch.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items()) and batch['candidate_labels_observed']==0
    counts=dict(win=0,tie=0,loss=0,unresolved=0);required=set();logical=0
    for ctx,row in zip(plan['contexts'],summary['rows']):
        file=out/'roots'/ctx['id']/'decision.json';assert sha(file)==batch['decision_sha256'][ctx['id']]
        d=read(file);assert d['candidate_labels_observed']==0 and d['derivative_check']['passed'];assert sha(d['prediction_file'])==d['prediction_sha256']
        pool=d['candidates'];root=ctx['root'];assert len(pool)==len({config_key(a['config']) for a in pool})
        original=moves(root,4,{config_key(root)})
        assert [a['config'] for a in pool[:len(original)]]==[a['config'] for a in original]
        for i,a in enumerate(pool):
            c=a['config'];assert legal(c) and c['l1_size']==root['l1_size'] and c['bp_type']==root['bp_type']
            assert a['edits']=={k:dict(before=root[k],after=c[k]) for k in root if c[k]!=root[k]}
            assert d['widths'][i]==compute_w_min(c)==c['dispatchWidth']
            if a['kind']=='single_resource':assert len(a['edits'])==1;continue
            k=a['parameter'];assert k in ('fetchWidth','decodeWidth','dispatchWidth','commitWidth')
            target=min(12,max(4,root[k]+a['sign']*a['stride']));assert c[k]==target
            # Independently reconstruct the minimal directed closure.
            v=dict(root);v[k]=target
            if k=='fetchWidth':v['decodeWidth']=min(v['decodeWidth'],target);v['dispatchWidth']=min(v['dispatchWidth'],v['decodeWidth'])
            if k=='decodeWidth':v['fetchWidth']=max(v['fetchWidth'],target);v['dispatchWidth']=min(v['dispatchWidth'],target)
            if k=='commitWidth':v['dispatchWidth']=min(v['dispatchWidth'],target)
            if k=='dispatchWidth':
                for p in ('decodeWidth','commitWidth','wbWidth'):v[p]=max(v[p],target)
                v['fetchWidth']=max(v['fetchWidth'],v['decodeWidth'])
            required_rob=2*v['dispatchWidth']+v['numIQEntries']+v['LQEntries']+v['SQEntries']
            v['numROBEntries']=max(root['numROBEntries'],required_rob);assert v==c
        with np.load(d['prediction_file']) as z:
            ell=z['root_ell'].astype(float);de=z['logodds_slope'].astype(float);width=np.array(d['widths'])[:,None,None]
            du=(softmax(ell,-1)[None]*de).sum(-1);uhat=logsumexp(ell,-1)[None]+du
            sc=np.minimum(100,1/(np.maximum(expit(-uhat),.01)*width+1e-8))
            ll=ell[None]+de;mm=softmax(np.concatenate([np.zeros_like(ll[...,:1]),ll],-1),-1);mc=cpi(mm,width)
            np.testing.assert_allclose(du,z['scalar_u_slope'],rtol=2e-5,atol=3e-6)
            for arm,st in (('scalar',sc),('mscs',mc)):
                np.testing.assert_allclose(st,z[arm+'_stage'],rtol=4e-6,atol=3e-6)
                np.testing.assert_allclose(st.mean((1,2)),d[arm+'_predicted_cpis'],rtol=4e-6,atol=3e-6)
            np.testing.assert_allclose(mm,z['predicted_m'],rtol=4e-6,atol=2e-7)
            root_account=(cpi(z['root_m'],compute_w_min(root))[...,None]*z['root_m']).mean((0,1))
            failed=read(HERE/'reclaim_supported_s42_01/roots'/ctx['id']/'decision.json')['unsupported_restrictions']
            excluded={config_key(r['action']['config']) for r in failed}
            assert excluded=={config_key(c) for c in d['known_unsupported_excluded']}
            allowed=[i for i,a in enumerate(pool) if config_key(a['config']) not in excluded]
            resolved=True;best={}
            for arm in ('scalar','mscs'):
                order=sorted(allowed,key=lambda i:(d[arm+'_predicted_cpis'][i],i));prefix=d['prefixes'][arm];selected=None
                assert [p['index'] for p in prefix]==order[:len(prefix)]
                for j,p in enumerate(prefix):
                    assert p['config']==pool[p['index']]['config']
                    if p['area'] is None:assert j==len(prefix)-1 and p['error'];resolved=False
                    elif p['area']<=ctx['area_cap']+1e-9:assert j==len(prefix)-1;selected=p['index']
                assert selected==d['choices'][arm]
                if selected is None:resolved=False
            assert resolved==d['resolved']==row['resolved']
            if not resolved:counts['unresolved']+=1;continue
            for arm,i in d['choices'].items():
                c=pool[i]['config'];lid=label_id(ctx['bench'],c);required.add(lid);logical+=1;r=labels[lid]
                assert r['area']==d['prefixes'][arm][-1]['area']<=ctx['area_cap']+1e-9
                with np.load(out/'labels'/f'{lid}.npz') as truth:
                    am=truth['m'];actual=float(cpi(am,compute_w_min(c)).mean())
                    np.testing.assert_allclose(actual,r['cpi'],rtol=3e-6)
                    if r['origin']=='reused_prior_paid_label':
                        with np.load(r['source_file']) as old:np.testing.assert_array_equal(am,old['m'])
                    delta=(cpi(am,compute_w_min(c))[...,None]*am).mean((0,1))-root_account
                pd=(mc[i,...,None]*mm[i]).mean((0,1))-root_account
                np.testing.assert_allclose(delta,row['explanations'][arm]['actual_account_change'],rtol=3e-4,atol=3e-6)
                np.testing.assert_allclose(pd,row['explanations'][arm]['mscs_predicted_account_change'],rtol=3e-4,atol=3e-6)
                np.testing.assert_allclose(delta.sum(),r['cpi']-ctx['root_cpi'],atol=4e-6)
                best[arm]=r['cpi'] if r['cpi']<ctx['root_cpi']*(1-1e-6) else ctx['root_cpi']
                assert best[arm]==row['outcomes'][arm]['best_cpi']
                np.testing.assert_allclose(row['outcomes'][arm]['gain_pct'],100*(1-best[arm]/ctx['root_cpi']),atol=1e-10)
        gap=(best['scalar']-best['mscs'])/ctx['root_cpi'];status='win' if gap>1e-6 else 'loss' if gap< -1e-6 else 'tie'
        assert row['status']==status;counts[status]+=1
    assert counts==summary['counts'] and required==set(labels) and logical==summary['logical_candidate_queries']
    assert sum(r['origin']=='new_forward' for r in labels.values())==summary['new_candidate_forwards']
    assert sum(r['origin']=='reused_prior_paid_label' for r in labels.values())==summary['reused_candidate_labels']
    gate=counts['loss']==0 and counts['unresolved']==0 and any(r.get('status')=='win' and r['bench']!='602.gcc_s' for r in summary['rows'])
    assert gate==summary['development_gate_passed']
    return dict(passed=True,width_and_closure_checked=True,scalar_gradient_checked=True,independent_prediction_replay=True,
        area_prefixes_checked=True,query_budget_checked=True,component_accounts_checked=True,sources_unchanged=True,development_gate_passed=gate)


if __name__=='__main__':
    import json
    print(json.dumps(validate(HERE/'width_priority_s42_01'),indent=2))
