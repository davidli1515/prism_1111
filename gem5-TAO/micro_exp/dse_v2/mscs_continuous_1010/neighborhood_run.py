"""Four architecture neighbors chosen before predictions; fixed three-workload follow-up."""
import time,json
import numpy as np
from secant_support import HERE,read,sha,save,config_key
from config_dse import DEFAULT_CONFIG
from width_priority_support import width_moves
from coverage_run_v2 import run
from neighborhood_validate import BENCHES,validate_leaf

OUT=HERE/'neighborhood_s42_01'


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    assert read(HERE/'coverage_s42_01/validation.json')['passed']
    sources={str(p):sha(p) for p in [*HERE.glob('neighborhood*.py'),HERE/'neighborhood_protocol.txt',HERE/'coverage_run_v2.py',
        HERE/'width_priority_support.py',HERE/'coverage_s42_01/summary.json',HERE/'coverage_s42_01/validation.json']}
    catalog=[a for a in width_moves(DEFAULT_CONFIG,1,{config_key(DEFAULT_CONFIG)}) if a['config']['fetchBufferSize']>=16]
    indices=np.random.RandomState(42).choice(len(catalog),size=4,replace=False).tolist()
    selected=[catalog[i] for i in indices]
    save(OUT/'root_plan.json',dict(seed=42,benchmarks=BENCHES,catalog=catalog,selected_indices=indices,selected=selected,source_sha256=sources))
    print(json.dumps(dict(phase='root_plan_frozen',indices=indices,edits=[a['edits'] for a in selected])),flush=True)
    rows=[];counts={s:0 for s in ('win','tie','loss','unresolved')};physical=0;logical=0
    for slot,a in enumerate(selected):
        folder=OUT/f'neighbor_{slot}';run(folder,a['config'],BENCHES,OUT/'root_plan.json')
        summary=read(folder/'summary.json');assert read(folder/'validation.json')['passed']
        rows.extend([dict(r,neighbor=slot,root=a['config']) for r in summary['rows']])
        for k,v in summary['counts'].items():counts[k]+=v
        physical+=summary['new_candidate_forwards'];logical+=summary['logical_candidate_queries']
    mean=float(np.mean([r['normalized_advantage'] for r in rows if r['resolved']]))
    winning_workloads=sorted({r['bench'] for r in rows if r.get('status')=='win'})
    gate=counts['unresolved']==0 and counts['win']>=6 and len(winning_workloads)>=2 and mean>1e-6 and counts['loss']<=1
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(rows=rows,counts=counts,winning_workloads=winning_workloads,mean_normalized_advantage=mean,
        development_gate_passed=gate,new_candidate_forwards=physical,logical_candidate_queries=logical,wall_seconds=time.perf_counter()-start,
        scope='local replication conditional on three earlier disagreements; not an unbiased all-workload or full-search result'))
    checks={str(slot):validate_leaf(OUT/f'neighbor_{slot}') for slot in range(4)}
    save(OUT/'validation.json',dict(passed=all(v['passed'] for v in checks.values()),leaves=checks,source_preservation=True))
    print(json.dumps(dict(phase='replication_complete',counts=counts,development_gate_passed=gate,mean_normalized_advantage=mean)),flush=True)


if __name__=='__main__':main()
