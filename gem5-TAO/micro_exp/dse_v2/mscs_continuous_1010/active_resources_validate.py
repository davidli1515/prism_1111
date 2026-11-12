"""Complete private-trajectory replay plus independently checked pair feasibility."""
import numpy as np
from secant_support import HERE,read,sha,config_key,compute_w_min
from active_pair_support import dense_singles
from pair_priority_support import pair_universe
from pair_priority_validate import forecasts
from width_priority_validate import cpi
from matched_support import label_id
from paid_centers_walk_numeric import validate_memory


def lazy_replay(pairs,scores,singles,rec,cap):
    known={};order=sorted(range(len(pairs)),key=lambda i:(scores[i],i));selected=None;resolved=True;nend=0;njoint=0
    assert [r['pair_index'] for r in rec['trace']]==order[:len(rec['trace'])]
    for n,t in enumerate(rec['trace']):
        pair=pairs[t['pair_index']]
        def priority(sid):
            key=config_key(singles[sid]['config'])
            if key not in known:return 2
            a,_=known[key];return 0 if a is None or a>cap+1e-9 else 1
        ids=sorted(pair['indices'],key=priority);blocked=False
        for j,e in enumerate(t['endpoint_checks']):
            assert e['index']==ids[j];key=config_key(singles[e['index']]['config']);fresh=key not in known
            assert fresh==e['new_request'];nend+=fresh
            if not fresh:assert known[key]==(e['area'],e['error'])
            known[key]=(e['area'],e['error'])
            if e['area'] is None:
                assert e['error'] and t['status']=='unknown_endpoint_area' and j==len(t['endpoint_checks'])-1;resolved=False;blocked=True;break
            if e['area']>cap+1e-9:
                assert t['status']=='endpoint_outside_declared_domain' and j==len(t['endpoint_checks'])-1;blocked=True;break
        if blocked:
            assert t['pair_area'] is None
            if not resolved:assert n==len(rec['trace'])-1;break
            continue
        assert len(t['endpoint_checks'])==2;njoint+=1
        if t['pair_area'] is None:assert t['status']=='unknown_joint_area' and t['pair_error'];resolved=False;assert n==len(rec['trace'])-1;break
        if t['pair_area']<=cap+1e-9:
            assert t['status']=='selected' and n==len(rec['trace'])-1;selected=t['pair_index'];break
        assert t['status']=='joint_over_cap'
    if selected is None and resolved:assert len(rec['trace'])==len(order)
    assert selected==rec['choice'] and resolved==rec['resolved'] and nend==rec['endpoint_requests'] and njoint==rec['joint_requests']


def validate_leaf(out):
    plan=read(out/'plan.json');summary=read(out/'summary.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    assert plan['arms']==['scalar_raw','mscs_raw','scalar_memory','mscs_memory'] and plan['rounds']==6 and plan['topk']==1
    assert plan['benchmarks']==['605.mcf_s','619.lbm_s','654.roms_s'];assert plan['area_cap']==1.10*plan['root_area']
    queried=set();logical=0;rootkeys={a:set() for a in plan['arms']};pend={a:0 for a in plan['arms']};pjoint={a:0 for a in plan['arms']};finalareas={a:0 for a in plan['arms']}
    pair_count=0;full_count=0
    for bench,rr in zip(plan['benchmarks'],summary['results']):
        initial_id=label_id(bench,plan['root'])
        for arm in plan['arms']:
            folder=out/'walks'/bench/arm;root=dict(plan['root']);cp=rr[arm]['initial_cpi'];aa=plan['root_area'];radius=4
            with np.load(out/'truth'/bench/f'{initial_id}.npz') as z:np.testing.assert_allclose(cpi(z['m'],compute_w_min(root)).mean(),cp,rtol=3e-6)
            seen={config_key(root)};history=[initial_id];history_configs={initial_id:dict(root)};queries=0;curve=[];active=True;files=sorted(folder.glob('decision_*.json'))
            for file in files:
                d=read(file);step=d['step'];r=read(folder/f'round_{step:02}.json')
                assert sha(file)==r['decision_sha256']==read(out/'batches'/bench/f'freeze_{step:02}.json')['decision_sha256'][arm]
                assert d['root']==root and d['root_cpi']==cp and d['radius']==radius and d['history_ids']==history and d['phase']=='before any arm current-round candidate labels'
                expected_ids=history if arm.endswith('_memory') else [label_id(bench,root)]
                assert d['center_ids']==expected_ids and d['center_configs']==[history_configs[lid] for lid in expected_ids]
                rootkeys[arm].update((bench,config_key(c)) for c in d['center_configs'])
                singles=[a for a in dense_singles(root,seen) if a['stride']<=radius];ns=len(singles);assert ns==d['singles_count'] and singles==d['candidates'][:ns]
                positive=[i for i,a in enumerate(singles) if a['kind']=='single_resource' and a['sign']==1]
                pairs=pair_universe(root,singles,positive,seen)
                assert pairs==d['pairs'];pair_count+=len(pairs)
                assert sha(d['prediction_file'])==d['prediction_sha256']
                with np.load(d['prediction_file']) as z:
                    ell=z['root_ell'].astype(float);de=z['logodds_slope'].astype(float)
                    sc,mc,_=forecasts(ell,de,d['widths'])
                    for kind,st in (('scalar',sc),('mscs',mc)):
                        np.testing.assert_allclose(st,z[kind+'_stage'],rtol=4e-6,atol=3e-6)
                        np.testing.assert_allclose(st.mean((1,2)),d['raw_'+kind+'_scores'],rtol=4e-6,atol=3e-6)
                    if pairs:
                        pde=np.stack([z['logodds_slope'][p['indices'][0]]+z['logodds_slope'][p['indices'][1]] for p in pairs]).astype(float)
                        ps,pm,_=forecasts(ell,pde,[compute_w_min(root)]*len(pairs))
                        np.testing.assert_allclose(ps.mean((1,2)),d['pair_scores']['scalar'],rtol=4e-6,atol=3e-6)
                        np.testing.assert_allclose(pm.mean((1,2)),d['pair_scores']['mscs'],rtol=4e-6,atol=3e-6)
                validate_memory(out,bench,dict(d,history_ids=d['center_ids']))
                if arm.endswith('_raw'):
                    assert d['scalar_scores']==d['raw_scalar_scores'] and d['mscs_scores']==d['raw_mscs_scores']
                pool=list(singles);expected_info=[];resolved=True
                for kind,rec in d['pair_traces'].items():
                    lazy_replay(pairs,d['pair_scores'][kind],singles,rec,plan['area_cap']);assert rec['choice']==d['proposed_pairs'][kind]
                    pend[arm]+=rec['endpoint_requests'];pjoint[arm]+=rec['joint_requests'];resolved &= rec['resolved']
                for kind,i in d['proposed_pairs'].items():
                    if i is None:continue
                    a=pairs[i]
                    if any(v['config']==a['config'] for v in pool):continue
                    expected_info.append((kind,i,len(pool)));pool.append(a)
                assert all(all(a['config'][k]==plan['root'][k] for k in plan['frozen_parameters']) for a in pool)
                assert pool==d['candidates'];assert [(r['generator'],r['pair_index'],r['pool_index']) for r in d['full_pair_info']]==expected_info
                full_count+=len(expected_info);assert d['widths']==[compute_w_min(a['config']) for a in pool]
                kind=arm.split('_')[0];order=sorted(range(len(pool)),key=lambda i:(d[kind+'_scores'][i],i));prefix=d['area_prefix'];choice=None
                assert [p['index'] for p in prefix]==order[:len(prefix)];finalareas[arm]+=len(prefix)
                for j,p in enumerate(prefix):
                    if p['area'] is None:assert j==len(prefix)-1 and p['error'];resolved=False
                    elif p['area']<=plan['area_cap']+1e-9:assert j==len(prefix)-1;choice=p['index']
                if choice is None:resolved=False
                assert choice==d['choice'] and resolved==d['resolved']==r['resolved']
                if not resolved:assert file==files[-1] and r['trial'] is None;active=False;continue
                t=r['trial'];c=pool[choice]['config'];lid=label_id(bench,c)
                assert t['config']==c and t['label_id']==lid and t['area']==prefix[-1]['area'] and config_key(c) not in seen
                with np.load(out/'truth'/bench/f'{lid}.npz') as z:np.testing.assert_allclose(cpi(z['m'],compute_w_min(c)).mean(),t['cpi'],rtol=3e-6)
                queried.add((bench,lid));queries+=1;history.append(lid);history_configs[lid]=c;seen.add(config_key(c))
                accepted=t['cpi']<cp*(1-1e-6);assert accepted==r['accepted']
                if accepted:root=c;cp=t['cpi'];aa=t['area'];radius=4
                else:radius=max(1,radius//2)
                curve.append(cp);assert r['next_root']==root and r['next_cpi']==cp and r['next_area']==aa and r['next_radius']==radius and r['queries']==queries
            result=read(folder/'summary.json');assert result==rr[arm] and result['curve']==curve and result['queries']==queries<=6 and result['resolved']==active
            assert result['final_config']==root and result['final_cpi']==cp and result['final_area']==aa
            np.testing.assert_allclose(result['gain_pct'],100*(1-cp/result['initial_cpi']),atol=1e-10);logical+=queries
    new=0;reused=0
    for bench,lid in queried:
        rec=read(out/'truth'/bench/f'{lid}.json')
        if rec['origin']=='old_paid_label_reused_after_choice':
            reused+=1
            with np.load(rec['source_file']) as a,np.load(out/'truth'/bench/f'{lid}.npz') as b:np.testing.assert_array_equal(a['m'],b['m'])
        else:assert rec['origin']=='new_forward';new+=1
    assert new==summary['new_candidate_forwards'] and reused==summary['reused_old_labels'] and logical==summary['logical_candidate_queries']
    assert {a:len(v) for a,v in rootkeys.items()}==summary['logical_root_evaluations'] and len(set().union(*rootkeys.values()))==summary['root_forwards']
    assert pend==summary['logical_pair_endpoint_accesses'] and pjoint==summary['logical_pair_area_accesses'] and finalareas==summary['logical_final_area_accesses']
    assert pair_count==summary['chord_pairs_scored'] and full_count==summary['full_pair_inputs_built']
    for rr,row in zip(summary['results'],summary['comparisons']):
        resolved=all(v['resolved'] and v['queries']==6 for v in rr.values());assert resolved==row['resolved']
        if not resolved:continue
        sc=np.minimum(rr['scalar_raw']['curve'],rr['scalar_memory']['curve']);den=rr['mscs_memory']['initial_cpi'];assert sc.tolist()==row['scalar_envelope']
        for arm in ('mscs_raw','mscs_memory'):
            ms=np.array(rr[arm]['curve']);adv=float(((sc-ms)/den).mean());end=float((sc[-1]-ms[-1])/den);c=row['comparisons'][arm]
            np.testing.assert_allclose([adv,end],[c['anytime_advantage'],c['endpoint_advantage']],atol=1e-12)
            assert c['anytime_status']==('win' if adv>1e-6 else 'loss' if adv< -1e-6 else 'tie') and c['endpoint_status']==('win' if end>1e-6 else 'loss' if end< -1e-6 else 'tie')
    rows=summary['comparisons'];eligible=[];means={}
    for arm in ('mscs_raw','mscs_memory'):
        means[arm]={k:float(np.mean([r.get('comparisons',{}).get(arm,{}).get(k,0) for r in rows])) for k in ('anytime_advantage','endpoint_advantage')}
        if all(r['resolved'] for r in rows) and all(v>1e-6 for v in means[arm].values()) and sum(r['comparisons'][arm]['endpoint_status']=='win' for r in rows)>=2:eligible.append(arm)
    assert eligible==summary['eligible_methods'] and means==summary['mean_advantages']
    assert plan['frozen_parameters']==['l2_size','l2_assoc']
    assert plan['root_forwards_include_all_paid_centers']
    assert reused==0
    return dict(passed=True,all_private_trajectories_replayed=True,dense_grid_radius_and_common_pair_domains_checked=True,
        lazy_feasibility_traces_and_numeric_predictions_replayed=True,exact_query_budgets_checked=True,
        no_prior_candidate_labels_consumed=True,sources_unchanged=True)




if __name__=='__main__':
    import json
    print(json.dumps(validate_leaf(HERE/'active_resources_s42_01'),indent=2))
