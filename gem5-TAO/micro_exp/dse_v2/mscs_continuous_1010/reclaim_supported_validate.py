"""Independent finite-model, rounding, legal-domain and outcome replay."""
import numpy as np
from secant_support import HERE,read,sha,save,compute_w_min,config_key
from core import legal,get_param_values,moves,PARAMS
from adaptive_validate import prediction_replay,close,stage_cpi,softmax
from matched_support import label_id


def validate(out):
    plan=read(out/'plan.json');summary=read(out/'summary.json');labels=read(out/'labels.json');freeze=read(out/'frozen_batch.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items());assert freeze['candidate_labels_observed']==0
    count=0;new=0;reused=0;rows=[]
    for ctx,row in zip(plan['contexts'],summary['rows']):
        path=out/'roots'/ctx['id']/'decision.json';d=read(path);c=ctx['root'];width=compute_w_min(c)
        assert d['root']==c and sha(path)==freeze['decision_sha256'][ctx['id']]
        original=[a for a in moves(c,4,{config_key(c)}) if a['sign']==-1];basis=[];areas=[];cursor=0
        for name in PARAMS:
            candidates=sorted([a for a in original if a['parameter']==name],key=lambda a:a['config'][name])
            for a in candidates:
                r=d['area_basis_attempts'][cursor];cursor+=1;assert r['action']==a
                if r['area'] is None:
                    assert r['error'] and r['status']=='unmodeled_backend_failure'
                else:
                    assert r['area']>0 and r['status']=='supported' and r['error'] is None
                    basis.append(a);areas.append(r['area']);break
        assert cursor==len(d['area_basis_attempts'])
        assert d['unsupported_restrictions']==[r for r in d['area_basis_attempts'] if r['area'] is None]
        bounds={a['parameter']:a['config'][a['parameter']] for a in basis}
        singles=[a for a in original if a['parameter'] in bounds and a['config'][a['parameter']]>=bounds[a['parameter']]]
        assert singles==d['singles'] and basis==d['basis'] and areas==d['basis_areas']
        close(np.array(areas)-ctx['root_area'],d['delta_area'])
        z=prediction_replay(out,d,'local',width);n=len(basis);bounds={a['parameter']:a['config'][a['parameter']] for a in basis}
        assert d['checks']['root_jvp']['passed']
        lo=1/width;hi=min(100.,1/(.01*width+1e-8));excess=np.maximum(z['root_cs']-lo,1e-10)
        single_scalar=np.clip(lo+excess[None]*np.exp(np.clip(z['single_stage_slope']/excess[None],-80,80)),lo,hi).mean((1,2))
        ell=z['root_ell'][None]+z['single_logodds_slope']
        mm=softmax(np.concatenate([np.zeros_like(ell[...,:1]),ell],-1));close(mm,z['single_predicted_m'])
        close(single_scalar,d['single_scalar_predicted_cpis']);close(stage_cpi(mm,width).mean((1,2)),d['single_mscs_predicted_cpis'])
        proxy=[]
        for a in singles:
            name=a['parameter'];idx=next(j for j,b in enumerate(basis) if b['parameter']==name)
            ratio=(a['config'][name]-c[name])/(basis[idx]['config'][name]-c[name])
            proxy.append(ctx['root_area']+ratio*d['delta_area'][idx])
        close(proxy,d['single_predicted_areas'])
        def safe_indices(kind):
            return [i for i,v in enumerate(d['single_'+kind+'_predicted_cpis']) if v<=ctx['cpi_cap'] and proxy[i]<ctx['root_area']]
        safe=safe_indices('scalar');anchor=min(safe,key=lambda i:(proxy[i],i)) if safe else int(np.argmin(d['single_scalar_predicted_cpis']))
        assert d['anchor_index']==anchor and d['anchor_unsafe']==(not bool(safe))
        assert d['candidates'][0]['config']==singles[anchor]['config']

        for arm in ('scalar','mscs'):
            info=d['details'][arm];solution=info['solution'];w=np.array(solution['fractions'])
            assert ((w>=0)&(w<=1)).all();check=d['checks'][arm]
            assert check['passed'];close(check['analytic'],check['numeric'],rtol=2e-4,atol=2e-7)
            lo=1/width;hi=min(100.,1/(.01*width+1e-8))
            if arm=='scalar':
                excess=np.maximum(z['root_cs'].astype(float)-lo,1e-10)
                slope=np.einsum('r,rns->ns',w,z['basis_stage_slope'])
                stages=np.clip(lo+excess*np.exp(np.clip(slope/excess,-80,80)),lo,hi)
            else:
                ell=z['root_ell'].astype(float)+np.einsum('r,rnsk->nsk',w,z['basis_logodds_slope'])
                mm=softmax(np.concatenate([np.zeros_like(ell[...,:1]),ell],-1));stages=stage_cpi(mm,width)
            close(stages.mean()/z['root_cs'].mean(),solution['normalized_predicted_cpi'],rtol=1e-5,atol=1e-6)
            if solution['success']:
                assert solution['cpi_constraint_slack']>=-1e-6 and solution['capacity_constraint_slack']>=-1e-6
            for r in info['rounded']:
                expected=dict(c)
                for a,t in zip(basis,w):
                    k=a['parameter'];target=c[k]+t*(a['config'][k]-c[k]);values=[v for v in get_param_values(k) if bounds[k]<=v<=c[k]]
                    expected[k]=max(v for v in values if v<=target+1e-10) if r['rounding']=='floor' else min(values,key=lambda v:(abs(v-target),v))
                required=2*expected['dispatchWidth']+expected['numIQEntries']+expected['LQEntries']+expected['SQEntries']
                if expected['numROBEntries']<required:
                    expected['numROBEntries']=min(v for v in get_param_values('numROBEntries') if required<=v<=c['numROBEntries'])
                assert expected==r['config'] and legal(expected)
                fractions=[(expected[a['parameter']]-c[a['parameter']])/(a['config'][a['parameter']]-c[a['parameter']]) for a in basis]
                close(fractions,r['fractions']);aa=ctx['root_area']+np.dot(fractions,d['delta_area'])
                close(aa,r['predicted_area']);assert aa<ctx['root_area']
                assert r['predicted_safe']==(r[arm+'_predicted_cpi']<=ctx['cpi_cap'])
            rounded=[r for r in info['rounded'] if r['config']!=d['candidates'][0]['config'] and r['predicted_safe']]
            selected=d['candidates'][d['chosen'][arm][1]]
            if rounded:
                assert selected['config']==min(rounded,key=lambda r:r['predicted_area'])['config']
                assert selected['kind']=='joint_rounded'
            else:
                assert selected['kind']=='single_fallback'
                eligible=[i for i in safe_indices(arm) if i!=anchor]
                index=min(eligible,key=lambda i:(proxy[i],i)) if eligible else next(i for i in np.argsort(d['single_'+arm+'_predicted_cpis'],kind='stable') if i!=anchor)
                assert selected['config']==singles[index]['config'] and selected['unsafe_fallback']==(not bool(eligible))
            assert d['cross_ranking'][arm]==np.argsort(d[arm+'_predicted_cpis'],kind='stable').tolist()
            assert len(set(d['chosen'][arm]))==2;count+=2
            best=ctx['root_area'];unknown=False;accepted=None
            for i in d['chosen'][arm]:
                cfg=d['candidates'][i]['config'];assert legal(cfg)
                assert all(cfg[k]<=c[k] for k in c)
                assert compute_w_min(cfg)==width and cfg['l1_size']==c['l1_size'] and cfg['bp_type']==c['bp_type']
                label=labels[label_id(ctx['bench'],cfg)];assert label['config']==cfg
                if label['cpi']<=ctx['cpi_cap']:
                    if label['area'] is None:unknown=True
                    elif label['area']<best:best=label['area'];accepted=label
            lower=0. if unknown else best;r=row['outcomes'][arm]
            assert r['area_lower']==lower and r['area_upper']==best and r['resolved']==(not unknown)
            if r['resolved']:
                close(r['saving_pct'],100*(1-best/ctx['root_area']))
                assert r['accepted_config']==(accepted['config'] if accepted else None)
                assert r['accepted_cpi']==(accepted['cpi'] if accepted else None)
        z.close();rows.append(dict(context=ctx['id'],passed=True))
    for lid,r in labels.items():
        assert lid==label_id(r['bench'],r['config'])
        with np.load(out/'labels'/f'{lid}.npz') as z:
            close(stage_cpi(z['m'],compute_w_min(r['config'])).mean(),r['cpi'])
            if r['origin']=='reused_prior_paid_label':
                reused+=1
                with np.load(r['source_file']) as old:np.testing.assert_array_equal(z['m'],old['m'])
            else:assert r['origin']=='new_forward';new+=1
    assert count==summary['logical_queries'] and new==summary['new_candidate_forwards'] and reused==summary['reused_candidate_labels']
    assert summary['basis_area_requests']==sum(1+len(read(out/'roots'/c['id']/'decision.json')['area_basis_attempts']) for c in plan['contexts'])
    return dict(passed=True,sources_unchanged=True,analytic_gradients_checked=True,root_tangents_match=True,
        finite_models_independently_replayed=True,full_input_endpoint_predictions_replayed=True,
        common_resource_boxes_and_area_information=True,area_only_domain_restrictions_replayed=True,rounding_legal_and_explicit=True,
        all_outcomes_and_query_counts_replayed=True,new_candidate_forwards=new,reused_candidate_labels=reused,
        logical_queries=count,contexts=rows)


if __name__=='__main__':
    import json
    print(json.dumps(validate(HERE/'reclaim_supported_s42_01'),indent=2))
