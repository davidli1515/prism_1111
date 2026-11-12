"""Independent NumPy interpolation, causal-history and cost replay."""
import numpy as np
from secant_support import HERE,read,sha,config_key,compute_w_min
from matched_support import label_id
from width_priority_validate import cpi


def validate(out):
    plan=read(out/'plan.json');summary=read(out/'summary.json');freeze=read(out/'freeze.json');src=HERE/'pair_walk_s42_01'
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    assert plan['arms']==['raw_scalar','raw_mscs','scalar_blend','mscs_blend'] and plan['steps']==[4,5]
    expected=[(b,s) for b in plan['benchmarks'] for s in plan['steps']]
    assert [(r['bench'],r['step']) for r in summary['rows']]==expected
    centers=set();labels=set();logical=0;deltas=[];improved=0;winning=set()
    for row in summary['rows']:
        b=row['bench'];step=row['step'];path=out/'decisions'/f'{b}__{step}.json';d=read(path);old=read(d['source_decision'])
        assert sha(path)==freeze['decision_sha256'][str(path)] and sha(d['prediction_file'])==d['prediction_sha256']
        assert d['root']==old['root'] and d['root_cpi']==old['root_cpi']
        assert d['history_ids']==old['history_ids'] and d['candidates']==old['candidates'] and d['widths']==old['widths']
        assert d['scores']['raw_scalar']==old['scalar_scores'] and d['scores']['raw_mscs']==old['mscs_scores']
        configs=[read(src/'plan.json')['root']]
        for j in range(step):configs.append(read(src/'walks'/b/'mscs_joint'/f'round_{j:02}.json')['trial']['config'])
        assert configs==d['center_configs'] and [label_id(b,c) for c in configs]==d['history_ids']
        assert len({config_key(c) for c in configs})==len(configs)
        centers.update((b,config_key(c)) for c in configs)
        with np.load(d['prediction_file']) as z:
            ds=z['scalar__distance_squared'];assert np.array_equal(ds,z['mscs__distance_squared'])
            assert (ds>=0).all();zeros=ds==0
            clamped=np.maximum(ds,np.finfo(ds.dtype).tiny)
            relative=clamped.min(0,keepdims=True)/clamped
            weights=relative/relative.sum(0,keepdims=True)
            exact=zeros/np.maximum(zeros.sum(0,keepdims=True),1)
            weights=np.where(zeros.any(0,keepdims=True),exact,weights)
            for kind in ('scalar','mscs'):
                np.testing.assert_allclose(z[kind+'__weights'],weights,rtol=1e-13,atol=1e-15)
                yy=z[kind+'__center_response'].astype(float);slope=z[kind+'__center_slope'].astype(float)
                # Float32 center+slope is what the original Hermite model evaluates.
                pp=(z[kind+'__center_response'][:,None]+z[kind+'__center_slope']).astype(float)
                expand=weights.reshape(*weights.shape,*([1]*(pp.ndim-weights.ndim)))
                pred=(expand*pp).sum(0).astype(np.float32)
                np.testing.assert_allclose(pred,z[kind+'__blended_response'],rtol=2e-6,atol=2e-6)
                for n,lid in enumerate(d['history_ids']):
                    with np.load(src/'truth'/b/f'{lid}.npz') as truth:
                        m=truth['m'].astype(float)
                    target=np.log1p(-m[...,0])-np.log(m[...,0]) if kind=='scalar' else np.log(m[...,1:])-np.log(m[...,:1])
                    np.testing.assert_allclose(yy[n],target,rtol=4e-6,atol=3e-6)
                if kind=='scalar':
                    base=1/(1+np.exp(pred.astype(float)))
                else:
                    full=np.concatenate([np.zeros_like(pred[...,:1]),pred],-1).astype(float)
                    ex=np.exp(full-full.max(-1,keepdims=True));mm=ex/ex.sum(-1,keepdims=True);base=mm[...,0]
                    np.testing.assert_allclose(mm,z['predicted_m'],rtol=4e-6,atol=3e-7)
                stages=np.minimum(100.,1/(np.maximum(base,.01)*np.array(d['widths'])[:,None,None]+1e-8))
                np.testing.assert_allclose(stages,z[kind+'__stage'],rtol=4e-6,atol=3e-6)
                np.testing.assert_allclose(stages.mean((1,2)),d['scores'][kind+'_blend'],rtol=4e-6,atol=3e-6)
        resolved=True
        for arm in plan['arms']:
            scores=d['scores'][arm];order=sorted(range(len(scores)),key=lambda i:(scores[i],i));prefix=d['area_prefixes'][arm]
            assert [r['index'] for r in prefix]==order[:len(prefix)]
            choice=None
            for j,r in enumerate(prefix):
                if r['area'] is None:assert r['error'] and j==len(prefix)-1;resolved=False;break
                if r['area']<=plan['area_cap']+1e-9:assert j==len(prefix)-1;choice=r['index'];break
            assert choice==d['choices'][arm];resolved &= choice is not None
        assert resolved==d['resolved']==row['resolved']
        if not resolved:assert not row['arms'];continue
        for arm,r in row['arms'].items():
            i=d['choices'][arm];c=d['candidates'][i]['config'];lid=label_id(b,c)
            assert r['index']==i and r['config']==c and r['label_id']==lid and lid not in d['history_ids']
            with np.load(out/'truth'/b/f'{lid}.npz') as z:cp=float(cpi(z['m'],compute_w_min(c)).mean())
            np.testing.assert_allclose(cp,r['cpi'],rtol=3e-6)
            np.testing.assert_allclose(max(0.,(d['root_cpi']-r['cpi'])/d['root_cpi']),r['gain'],atol=1e-12)
            labels.add((b,lid));logical+=1
        aa=row['arms'];m=aa['mscs_blend']['gain'];deltas.append(m-max(aa['raw_scalar']['gain'],aa['scalar_blend']['gain']))
        improved+=m>aa['raw_mscs']['gain']+1e-6
        if m>aa['scalar_blend']['gain']+1e-6:winning.add(b)
    physical=0;reused=0
    for b,lid in labels:
        rec=read(out/'truth'/b/f'{lid}.json')
        if rec['origin']=='new_forward':physical+=1;assert rec['source'] is None
        else:
            assert rec['origin']=='old_label_after_freeze';reused+=1
            with np.load(rec['source']) as a,np.load(out/'truth'/b/f'{lid}.npz') as z:np.testing.assert_array_equal(a['m'],z['m'])
    assert physical==summary['new_candidate_forwards'] and reused==summary['reused_old_labels'] and logical==summary['logical_candidate_queries']
    assert len(centers)==summary['historical_center_forwards'] and 12*len(centers)==summary['historical_center_backward_outputs']
    gate=all(r['resolved'] for r in summary['rows']) and improved>=2 and len(winning)>=2 and np.mean(deltas)>1e-6
    assert bool(gate)==summary['pilot_gate_passed'] and improved==summary['improved_vs_raw_mscs'] and sorted(winning)==summary['winning_workloads_vs_scalar_blend']
    if deltas:np.testing.assert_allclose(np.mean(deltas),summary['mean_advantage_vs_scalar_envelope'],atol=1e-12)
    return dict(passed=True,private_all_paid_histories_replayed=True,numpy_interpolation_and_cpi_reconstructed=True,
                rankings_feasibility_and_query_costs_checked=True,all_choices_frozen_before_labels=True,sources_preserved=True)


if __name__=='__main__':
    import json
    print(json.dumps(validate(HERE/'paid_centers_s42_01'),indent=2))
