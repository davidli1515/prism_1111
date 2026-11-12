"""Zero new queries: prove lazy and eager feasibility choose identical proposals."""
import time,json
import numpy as np
import torch
from secant_support import HERE,read,sha,save,array_save,config_key,compute_w_min
from pair_priority_support import pair_universe
from width_priority_support import predict
from lazy_pair_support_v2 import choose_pair

SRC=HERE/'pair_priority_s42_01';OUT=HERE/'lazy_pair_audit_s42_02'


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False);assert read(SRC/'validation.json')['passed']
    plan0=read(SRC/'plan.json');sources=dict(plan0['source_sha256'])
    for p in [*SRC.rglob('*.json'),*SRC.rglob('*.npz'),HERE/'lazy_pair_support_v2.py',HERE/'lazy_pair_audit_v2.py']:
        sources[str(p)]=sha(p)
    save(OUT/'plan.json',dict(contexts=plan0['contexts'],source_sha256=sources,
        scope='exact equivalence replay on existing six states; zero new model or McPAT queries'))
    rows=[]
    for ctx in plan0['contexts']:
        d=read(SRC/'states'/ctx['id']/'decision.json');assert d['resolved'];ns=d['singles_count'];singles=d['candidates'][:ns]
        positive=[r['index'] for r in d['basis_areas']];seen={config_key(c) for c in d['seen']}
        pairs=pair_universe(d['root'],singles,positive,seen)
        with np.load(d['prediction_file']) as z:
            ell=torch.from_numpy(z['root_ell'].copy());de=torch.from_numpy(z['logodds_slope'][:ns].copy())
        ids=torch.tensor([a['indices'] for a in pairs]);pde=de[ids[:,0]]+de[ids[:,1]]
        pred=predict(ell,pde,[compute_w_min(d['root'])]*len(pairs))
        lookup={config_key(singles[r['index']]['config']):(r['area'],r.get('error')) for r in d['basis_areas']}
        for kind in ('scalar','mscs'):
            for r in d['pair_prefixes'][kind]:lookup[config_key(d['pairs'][r['index']]['config'])]=(r['area'],r.get('error'))
        decisions={};calls={};all_called=set()
        for kind in ('scalar','mscs'):
            accessed=[]
            def old_area(c):
                key=config_key(c);accessed.append(key);all_called.add(key)
                assert key in lookup,'Replay would require an unknown area, so no equivalence claim is made.'
                return lookup[key]
            result=choose_pair(pairs,pred[kind+'_cpi'],singles,old_area,plan0['area_cap'])
            assert result['resolved'] and result['choice'] is not None
            selected=pairs[result['choice']]['config'];original=d['pairs'][d['proposed_pairs'][kind]]['config'];assert selected==original
            # Independently replay the short-circuit decisions in their exact score order.
            order=sorted(range(len(pairs)),key=lambda i:(float(pred[kind+'_cpi'][i]),i));observed=set();n_end=0;n_joint=0
            assert [r['pair_index'] for r in result['trace']]==order[:len(result['trace'])]
            for n,r in enumerate(result['trace']):
                pair=pairs[r['pair_index']];blocked=False
                def expected_order(sid):
                    key=config_key(singles[sid]['config'])
                    if key not in observed:return 2
                    aa,_=lookup[key]
                    return 0 if aa is None or aa>plan0['area_cap']+1e-9 else 1
                ordered=sorted(pair['indices'],key=expected_order)
                for j,e in enumerate(r['endpoint_checks']):
                    assert e['index']==ordered[j];key=config_key(singles[e['index']]['config'])
                    assert e['new_request']==(key not in observed);n_end+=key not in observed;observed.add(key)
                    assert (e['area'],e['error'])==lookup[key]
                    if e['area']>plan0['area_cap']+1e-9:
                        assert j==len(r['endpoint_checks'])-1 and r['status']=='endpoint_outside_declared_domain';blocked=True;break
                if blocked:assert r['pair_area'] is None;continue
                assert len(r['endpoint_checks'])==2;n_joint+=1
                assert r['pair_area']==lookup[config_key(pair['config'])][0]
                if r['pair_area']<=plan0['area_cap']+1e-9:assert n==len(result['trace'])-1 and r['status']=='selected'
                else:assert r['status']=='joint_over_cap'
            assert n_end==result['endpoint_requests'] and n_joint==result['joint_requests'] and len(accessed)==n_end+n_joint
            decisions[kind]=result;calls[kind]=len(accessed)
        save(OUT/'states'/ctx['id']/'replay.json',dict(pairs=pairs,decisions=decisions,selected_configs_equal=True,
            scalar_scores=pred['scalar_cpi'].tolist(),mscs_scores=pred['mscs_cpi'].tolist()))
        row=dict(context=ctx['id'],eager_common_endpoint_requests=len(d['basis_areas']),lazy_endpoint_requests={k:v['endpoint_requests'] for k,v in decisions.items()},
            lazy_joint_requests={k:v['joint_requests'] for k,v in decisions.items()},unique_lazy_area_configs=len(all_called),all_pairs=len(pairs),selected_configs_equal=True)
        rows.append(row);print(json.dumps(row),flush=True)
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,new_model_queries=0,new_area_queries=0,all_equivalent=True,
        eager_common_endpoint_requests=sum(r['eager_common_endpoint_requests'] for r in rows),
        lazy_endpoint_requests={k:sum(r['lazy_endpoint_requests'][k] for r in rows) for k in ('scalar','mscs')},
        unique_lazy_area_configs=sum(r['unique_lazy_area_configs'] for r in rows),wall_seconds=time.perf_counter()-start))
    save(OUT/'validation.json',dict(passed=True,source_preservation=True,all_selected_pairs_identical=True,independent_prefix_replay=True,
        no_new_truth_or_area_queries=True,scope='six already observed states; call-count equivalence, not a walltime benchmark'))


if __name__=='__main__':main()
