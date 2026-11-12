"""Bitwise equivalence and timings; no model outputs, labels or search changes."""
import json,time
import numpy as np
import torch
from secant_support import HERE,read,sha,save,config_key
from core import Inputs
from pair_priority_support import dense_singles,pair_universe
from config_dse import DEFAULT_CONFIG
from exact_cached_inputs import CachedInputs,DistributionMemo
import dse_infer as module


def main():
    out=HERE/'exact_cached_inputs_audit_01';out.mkdir(exist_ok=False)
    original_encoder=module.encode_distribution
    sources={str(p):sha(p) for p in [HERE/'exact_cached_inputs.py',HERE/'exact_cached_inputs_audit.py',module.__file__]}
    roots=[DEFAULT_CONFIG,read(HERE/'pair_replication_s42_01/root_plan.json')['selected'][0]['config']]
    configs=[];used=set()
    for root in roots:
        singles=dense_singles(root,{config_key(root)})
        pairs=pair_universe(root,singles,[i for i,a in enumerate(singles) if a['kind']=='single_resource' and a['sign']==1],set())
        # One actual change of every available resource, all width closures,
        # plus 12 deterministically sampled legal pairs. No performance fields.
        selected=[];parameters=set()
        for a in singles:
            p=a.get('parameter')
            if a['kind']=='width_with_explicit_closure' or p not in parameters:
                selected.append(a);parameters.add(p)
        selected.extend(pairs[i] for i in np.random.RandomState(42).choice(len(pairs),min(12,len(pairs)),replace=False))
        for c in [root]+[a['config'] for a in selected]:
            key=config_key(c)
            if key not in used:used.add(key);configs.append(c)
    save(out/'plan.json',dict(configurations=configs,source_sha256=sources,scope='feature-only equivalence test; no surrogate evaluation'))
    rows=[]
    for bench in ('605.mcf_s','619.lbm_s','654.roms_s'):
        base=Inputs(bench,64);fast=CachedInputs(bench,64);assert base.ids==fast.ids
        reference=[];start=time.perf_counter()
        for c in configs:reference.append(base.build(c))
        baseline=time.perf_counter()-start
        start=time.perf_counter()
        for c,expected in zip(configs,reference):
            actual=fast.build(c)
            assert torch.equal(actual,expected),(bench,c)
        cold=time.perf_counter()-start
        start=time.perf_counter()
        for c,expected in zip(configs,reference):assert torch.equal(fast.build(c),expected)
        warm=time.perf_counter()-start
        rows.append(dict(bench=bench,configs=len(configs),reference_seconds=baseline,cold_memo_seconds=cold,
                         warm_memo_seconds=warm,cold_speedup=baseline/cold,warm_speedup=baseline/warm,
                         hits=fast.distributions.hits,misses=fast.distributions.misses,entries=len(fast.distributions.cache)))
        assert module.encode_distribution is original_encoder
    # Exact key distinguishes shape/dtype and content, including unsorted order.
    memo=DistributionMemo(capacity=2)
    for a in (np.array([1.,2.]),np.array([2.,1.]),np.array([1,2]),np.array([3.,1.])):
        np.testing.assert_array_equal(memo(a),original_encoder(a))
    assert len(memo.cache)==2 and memo.misses==4
    assert all(sha(p)==h for p,h in sources.items())
    save(out/'summary.json',dict(rows=rows,bitwise_equal=True,model_queries=0,original_module_unchanged=True,
                              scope='timing is for feature construction only, not full search speedup'))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
