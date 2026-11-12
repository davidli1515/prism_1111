"""Validate equal past feedback and independent solves before scoring outcomes."""
import numpy as np
import torch
from secant_support import HERE,read,sha,compute_w_min
from matched_support import label_id
from priority_walk_validate import quadratic_replay
from width_priority_validate import cpi


class View:
    def __init__(self,z,prefix):self.z=z;self.prefix=prefix
    def __getitem__(self,k):
        return self.z[self.prefix+k] if self.prefix+k in self.z.files else self.z[k]


def validate(out):
    src=HERE/'priority_walk_s42_01';plan=read(out/'plan.json');summary=read(out/'summary.json');batch=read(out/'frozen_batch.json');labels=read(out/'labels.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items()) and batch['non_history_candidate_labels_observed']==0
    expected=[dict(id=f'{b}_before_q{s+1}',bench=b,step=s) for b in read(src/'plan.json')['benchmarks'] for s in (4,5)]
    assert plan['contexts']==expected;required=set();logical=0
    for ctx,row in zip(expected,summary['rows']):
        file=out/'states'/ctx['id']/'decision.json';d=read(file);old=read(d['old_decision'])
        assert sha(file)==batch['decision_sha256'][ctx['id']] and d['non_history_candidate_labels_observed']==0
        assert d['history_ids']==old['history_ids'] and d['candidates']==old['candidates'] and d['root']==old['root']
        assert d['choices']['mscs_raw']==old['choice'] and d['scores']['mscs_raw']==old['scores']
        assert sha(d['prediction_file'])==d['prediction_sha256']
        with np.load(d['prediction_file']) as z:
            sc=quadratic_replay(View(z,'scalar__'),d['history_ids'],ctx['bench'],src,d['widths'])
            np.testing.assert_allclose(sc,d['scores']['scalar_quadratic'],rtol=4e-6,atol=3e-6)
            expected_y=[]
            for lid in d['history_ids']:
                with np.load(src/'truth'/ctx['bench']/f'{lid}.npz') as t:
                    m=t['m'].astype(float);expected_y.append(np.log(np.maximum(m[...,1:],1e-30))-np.log(np.maximum(m[...,:1],1e-30)))
            np.testing.assert_allclose(expected_y,z['mscs__history_response'],rtol=3e-5,atol=3e-6)
            residual=z['mscs__history_response']-z['root_ell'][None]-z['mscs__history_slope'];np.testing.assert_array_equal(residual,z['mscs__residual'])
            # Both response spaces use precisely the same input-geometry kernel.
            for k in ('gram','cross','weights','gram_linear','cross_linear'):np.testing.assert_array_equal(z['scalar__'+k],z['mscs__'+k])
            gram=z['mscs__gram'];ev,v=np.linalg.eigh(gram);cut=np.abs(ev).max(-1,keepdims=True)*1e-6
            inv=np.divide(1.,ev,out=np.zeros_like(ev),where=np.abs(ev)>cut);pinv=np.einsum('nhi,ni,nki->nhk',v,inv,v)
            weights=np.einsum('nch,nhk->nck',z['mscs__cross'],pinv)
            rhs=residual.transpose(1,0,2,3).reshape(residual.shape[1],residual.shape[0],12).astype(float)
            correction=np.einsum('nch,nhv->ncv',weights,rhs).transpose(1,0,2).reshape(z['mscs__correction'].shape).astype(np.float32)
            np.testing.assert_allclose(correction,z['mscs__correction'],rtol=3e-4,atol=3e-6)
            ell=torch.from_numpy(z['root_ell'])[None]+torch.from_numpy(z['mscs__logodds_slope'])+torch.from_numpy(correction)
            mm=torch.softmax(torch.cat([torch.zeros_like(ell[...,:1]),ell],-1),-1)
            w=torch.tensor(d['widths'],dtype=mm.dtype)[:,None,None];st=(1/(mm[...,0].clamp(min=.01)*w+1e-8)).clamp(max=100.)
            np.testing.assert_allclose(mm,z['mscs__corrected_m'],rtol=4e-6,atol=3e-6)
            np.testing.assert_allclose(st.mean((1,2)),d['scores']['mscs_quadratic'],rtol=4e-6,atol=3e-6)
        resolved=True
        for arm in plan['arms']:
            order=sorted(range(len(d['candidates'])),key=lambda i:(d['scores'][arm][i],i));prefix=d['prefixes'][arm];choice=None
            assert [p['index'] for p in prefix]==order[:len(prefix)]
            for j,p in enumerate(prefix):
                if p['area'] is None:assert j==len(prefix)-1 and p['error'];resolved=False
                elif p['area']<=plan['area_cap']+1e-9:assert j==len(prefix)-1;choice=p['index']
            assert choice==d['choices'][arm]
            if choice is None:resolved=False
        assert resolved==d['resolved']==row['resolved']
        if not resolved:continue
        best={}
        for arm,i in d['choices'].items():
            c=d['candidates'][i]['config'];lid=label_id(ctx['bench'],c);required.add(lid);logical+=1;r=labels[lid]
            assert r['area']==d['prefixes'][arm][-1]['area']<=plan['area_cap']+1e-9
            with np.load(out/'labels'/f'{lid}.npz') as z:
                np.testing.assert_allclose(cpi(z['m'],compute_w_min(c)).mean(),r['cpi'],rtol=3e-6)
                if r['origin']=='reused_after_freeze':
                    with np.load(r['source_file']) as oldlabel:np.testing.assert_array_equal(z['m'],oldlabel['m'])
            best[arm]=r['cpi'] if r['cpi']<d['root_cpi']*(1-1e-6) else d['root_cpi'];assert best[arm]==row['outcomes'][arm]['best_cpi']
        np.testing.assert_allclose(row['advantage_vs_scalar'],(best['scalar_quadratic']-best['mscs_quadratic'])/d['root_cpi'],atol=1e-12)
        np.testing.assert_allclose(row['advantage_vs_raw'],(best['mscs_raw']-best['mscs_quadratic'])/d['root_cpi'],atol=1e-12)
    assert required==set(labels) and logical==summary['logical_candidate_queries']
    assert sum(r['origin']=='new_forward' for r in labels.values())==summary['new_candidate_forwards']
    assert sum(r['origin']=='reused_after_freeze' for r in labels.values())==summary['reused_candidate_labels']
    gate=all(r['resolved'] for r in summary['rows']) and any(r['bench']=='654.roms_s' and r.get('advantage_vs_scalar',0)>1e-6 for r in summary['rows'])
    assert gate==summary['roms_development_gate_passed']
    return dict(passed=True,equal_causal_histories_checked=True,base_only_scalar_control_checked=True,independent_component_correction_replayed=True,
        frozen_choices_area_prefixes_and_outcomes_checked=True,logical_query_budget_checked=True,sources_unchanged=True)


if __name__=='__main__':
    import json
    print(json.dumps(validate(HERE/'feedback_probe_s42_01'),indent=2))
