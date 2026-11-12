"""Replay private histories, scalar-only corrections and matched query curves."""
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax,logsumexp,expit
from secant_support import HERE,read,sha,config_key,compute_w_min
from core import legal
from config_dse import DEFAULT_CONFIG
from width_priority_support import width_moves
from matched_support import label_id
from width_priority_validate import cpi


def root_replay(plan):
    sel=plan['selection'];catalog=[a for a in width_moves(DEFAULT_CONFIG,1,{config_key(DEFAULT_CONFIG)}) if a['config']['fetchBufferSize']>=16]
    assert catalog==sel['catalog'];order=np.random.RandomState(42).choice(len(catalog),len(catalog),replace=False).tolist();assert order==sel['permutation']
    assert order[:4]==read(HERE/'neighborhood_s42_01/root_plan.json')['selected_indices']
    root=dict(DEFAULT_CONFIG);fields=set();accepted=[];audit=[]
    for i in order[4:]:
        a=catalog[i];candidate=dict(root)
        for k,v in a['edits'].items():candidate[k]=v['after']
        ok=not bool(fields&set(a['edits'])) and legal(candidate);audit.append(dict(index=i,accepted=ok,edits=a['edits']))
        if ok:root=candidate;fields.update(a['edits']);accepted.append(i)
        if len(accepted)==2:break
    assert root==plan['root'] and accepted==sel['accepted_indices'] and audit==sel['audit']
    n=0
    for folder in [HERE/'coverage_s42_01',*[HERE/'neighborhood_s42_01'/f'neighbor_{i}' for i in range(4)]]:
        p=read(folder/'plan.json');configs=[p['root']]+[a['config'] for a in p['candidates']];assert root not in configs;n+=len(configs)
    assert n==sel['prior_configs_checked']


def quadratic_replay(z,history,bench,out,widths):
    expected=[]
    for lid in history:
        with np.load(out/'truth'/bench/f'{lid}.npz') as a:
            b=a['m'][...,0].astype(float).clip(1e-30,1-1e-7);expected.append(np.log1p(-b)-np.log(b))
    np.testing.assert_allclose(expected,z['history_response'],rtol=3e-5,atol=4e-6)
    rootu=logsumexp(z['root_ell'].astype(float),-1)
    np.testing.assert_allclose(rootu,z['root_u'],rtol=3e-6,atol=1e-6)
    residual=z['history_response']-z['root_u'][None]-z['history_slope']
    np.testing.assert_array_equal(residual,z['residual'])
    np.testing.assert_allclose(z['gram'],z['gram_linear']**2,rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(z['cross'],z['cross_linear']**2,rtol=1e-12,atol=1e-10)
    gram=z['gram'];ev,vec=np.linalg.eigh(gram);cut=np.max(np.abs(ev),axis=-1,keepdims=True)*1e-6
    inverse=np.divide(1.,ev,out=np.zeros_like(ev),where=np.abs(ev)>cut)
    pinv=np.einsum('nhi,ni,nki->nhk',vec,inverse,vec);weights=np.einsum('nch,nhk->nck',z['cross'],pinv)
    np.testing.assert_allclose(weights,z['weights'],rtol=2e-4,atol=3e-6)
    rhs=z['residual'].transpose(1,0,2).astype(float)
    correction=np.einsum('nch,nhs->ncs',weights,rhs).transpose(1,0,2).astype(np.float32)
    np.testing.assert_allclose(correction,z['correction'],rtol=3e-4,atol=3e-6)
    q=softmax(z['root_ell'].astype(float),-1)
    du=(q[None]*z['logodds_slope']).sum(-1)
    np.testing.assert_allclose(du,z['scalar_u_slope'],rtol=3e-5,atol=3e-6)
    # Recorded RHS was independently checked; replay original float32 response arithmetic.
    u=torch.from_numpy(z['root_u'])[None]+torch.from_numpy(z['scalar_u_slope'])+torch.from_numpy(correction)
    b=torch.sigmoid(-u);w=torch.tensor(widths,dtype=b.dtype)[:,None,None]
    stages=(1/(b.clamp(min=.01)*w+1e-8)).clamp(max=100.)
    np.testing.assert_allclose(stages,z['corrected_stages'],rtol=4e-6,atol=3e-6)
    return stages.mean((1,2)).numpy()


def validate(out):
    plan=read(out/'plan.json');summary=read(out/'summary.json');area=read(out/'area_plan.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items());root_replay(plan)
    assert plan['arms']==['scalar_raw','scalar_quadratic','mscs_raw'] and plan['rounds']==6 and plan['topk']==1 and plan['history']==6
    assert plan['benchmarks']==['605.mcf_s','619.lbm_s','654.roms_s'];assert area['area_cap']==1.10*area['root_area']
    results=[];physical=set();logical=0;prefix_accesses={a:0 for a in plan['arms']};root_keys={a:set() for a in plan['arms']}
    for bench,rr in zip(plan['benchmarks'],summary['results']):
        initial_id=label_id(bench,plan['root'])
        with np.load(out/'truth'/bench/f'{initial_id}.npz') as z:initial_cpi=float(cpi(z['m'],compute_w_min(plan['root'])).mean())
        replayed={}
        for arm in plan['arms']:
            folder=out/'walks'/bench/arm;root=dict(plan['root']);cp=rr[arm]['initial_cpi'];np.testing.assert_allclose(cp,initial_cpi,rtol=3e-6)
            aa=area['root_area'];radius=4;seen={config_key(root)};history=[initial_id];queries=0;curve=[];active=True
            files=sorted(folder.glob('decision_*.json'))
            for file in files:
                d=read(file);step=d['step'];r=read(folder/f'round_{step:02}.json')
                assert root==d['root'] and cp==d['root_cpi'] and radius==d['radius'];root_keys[arm].add((bench,config_key(root)))
                assert read(out/'root_checks'/bench/f'{label_id(bench,root)}.json')['passed']
                assert sha(file)==r['decision_sha256']==read(out/'batches'/bench/f'freeze_{step:02}.json')['decision_sha256'][arm]
                assert d['history_ids']==history[-6:] and d['phase']=='before any current-round candidate CPI label'
                cand=[a for a in width_moves(root,radius,seen) if a['config']['fetchBufferSize']>=16]
                assert cand==d['candidates'];widths=[compute_w_min(a['config']) for a in cand];assert widths==d['widths']
                assert sha(d['prediction_file'])==d['prediction_sha256']
                with np.load(d['prediction_file']) as z:
                    ell=z['root_ell'].astype(float);de=z['logodds_slope'].astype(float);q=softmax(ell,-1)
                    u=logsumexp(ell,-1)[None]+(q[None]*de).sum(-1);w=np.array(widths)[:,None,None]
                    sc=np.minimum(100.,1/(np.maximum(expit(-u),.01)*w+1e-8))
                    ll=ell[None]+de;mm=softmax(np.concatenate([np.zeros_like(ll[...,:1]),ll],-1),-1);mc=cpi(mm,w)
                    for name,st in (('scalar',sc),('mscs',mc)):
                        np.testing.assert_allclose(st,z[name+'_stage'],rtol=4e-6,atol=3e-6)
                        np.testing.assert_allclose(st.mean((1,2)),d[name+'_predicted_cpis'],rtol=4e-6,atol=3e-6)
                    if arm=='scalar_quadratic':independent=quadratic_replay(z,history[-6:],bench,out,widths)
                    else:
                        name='mscs' if arm=='mscs_raw' else 'scalar'
                        independent=torch.from_numpy(z[name+'_stage']).mean((1,2)).numpy()
                    np.testing.assert_allclose(independent,d['scores'],rtol=4e-6,atol=3e-6)
                order=sorted(range(len(cand)),key=lambda i:(d['scores'][i],i));prefix=d['area_prefix'];choice=None;resolved=True
                assert [p['index'] for p in prefix]==order[:len(prefix)];prefix_accesses[arm]+=len(prefix)
                for j,p in enumerate(prefix):
                    if p['area'] is None:assert j==len(prefix)-1 and p['error'];resolved=False
                    elif p['area']<=area['area_cap']+1e-9:assert j==len(prefix)-1;choice=p['index']
                if choice is None:resolved=False
                assert choice==d['choice'] and resolved==d['resolved']==r['resolved']
                # Independent response reconstruction must choose the same first feasible point.
                if resolved:
                    known={p['index']:p['area'] for p in prefix}
                    eligible=[i for i,a in known.items() if a is not None and a<=area['area_cap']+1e-9]
                    assert choice==min(eligible,key=lambda i:(float(independent[i]),i))
                if not resolved:
                    assert file==files[-1] and r['trial'] is None;active=False;continue
                t=r['trial'];c=cand[choice]['config'];lid=label_id(bench,c)
                assert t['config']==c and t['label_id']==lid and t['area']==prefix[-1]['area'] and config_key(c) not in seen
                with np.load(out/'truth'/bench/f'{lid}.npz') as z:np.testing.assert_allclose(cpi(z['m'],compute_w_min(c)).mean(),t['cpi'],rtol=3e-6)
                physical.add((bench,lid));queries+=1;seen.add(config_key(c));history.append(lid)
                accepted=t['cpi']<cp*(1-1e-6);assert accepted==r['accepted']
                if accepted:root=c;cp=t['cpi'];aa=t['area'];radius=4
                else:radius=max(1,radius//2)
                curve.append(cp)
                assert r['next_root']==root and r['next_cpi']==cp and r['next_area']==aa and r['next_radius']==radius and r['queries']==queries
            result=read(folder/'summary.json');assert result==rr[arm]
            assert root==result['final_config'] and cp==result['final_cpi'] and curve==result['curve'] and queries==result['queries']<=6 and active==result['resolved']
            np.testing.assert_allclose(result['gain_pct'],100*(1-cp/result['initial_cpi']),atol=1e-10)
            logical+=queries;replayed[arm]=result
        results.append(replayed)
    assert logical==summary['logical_candidate_queries'] and len(physical)==summary['new_candidate_forwards']
    assert prefix_accesses==summary['logical_area_prefix_accesses'] and {a:len(v) for a,v in root_keys.items()}==summary['logical_root_evaluations']
    assert len(set().union(*root_keys.values()))==summary['root_forwards']
    for rr,row in zip(results,summary['comparisons']):
        resolved=all(v['resolved'] and v['queries']==6 for v in rr.values());assert resolved==row['resolved']
        if not resolved:continue
        env=np.minimum(rr['scalar_raw']['curve'],rr['scalar_quadratic']['curve']);ms=np.array(rr['mscs_raw']['curve']);initial=rr['mscs_raw']['initial_cpi']
        adv=float(((env-ms)/initial).mean());end=float((env[-1]-ms[-1])/initial)
        np.testing.assert_allclose([adv,end],[row['anytime_advantage'],row['endpoint_advantage']],atol=1e-12)
        assert row['anytime_status']==('win' if adv>1e-6 else 'loss' if adv< -1e-6 else 'tie')
        assert row['endpoint_status']==('win' if end>1e-6 else 'loss' if end< -1e-6 else 'tie')
    rows=summary['comparisons'];gate=all(r['resolved'] for r in rows) and sum(r.get('anytime_status')=='win' for r in rows)>=2 and all(r.get('endpoint_advantage',-1)>=-1e-6 for r in rows) and np.mean([r.get('anytime_advantage',0) for r in rows])>1e-6
    assert bool(gate)==summary['development_gate_passed']
    return dict(passed=True,new_root_selection_replayed=True,private_histories_and_query_counts_replayed=True,
        scalar_base_only_transform_and_independent_quadratic_solve_checked=True,variable_width_predictions_replayed=True,
        area_prefixes_and_acceptance_checked=True,all_curves_and_scalar_envelope_replayed=True,sources_unchanged=True)


if __name__=='__main__':
    import json
    print(json.dumps(validate(HERE/'priority_walk_s42_01'),indent=2))
