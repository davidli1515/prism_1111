"""Prospective physical-key sampling and aggregate replay; leaf replay unchanged."""
import numpy as np
from secant_support import HERE,read,sha
from active_resources_validate import validate_leaf


def validate_all(out):
    from active_confirmation_roots import select
    sampled=select();plan=read(out/'root_plan.json');s=read(out/'summary.json')
    assert all(plan[k]==v for k,v in sampled.items())
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    assert plan['selected_method']==s['selected_method']=='mscs_memory'
    rows=[];physical=0;logical=0
    for slot,entry in enumerate(plan['selected']):
        leaf=out/f'root_{slot}';lp=read(leaf/'plan.json');lr=read(leaf/'summary.json')
        assert lp['root']==entry['config'] and lp['root_plan']==str(out/'root_plan.json') and read(leaf/'validation.json')['passed']
        physical+=lr['new_candidate_forwards'];logical+=lr['logical_candidate_queries']
        for row in lr['comparisons']:
            rec=dict(bench=row['bench'],slot=slot,resolved=row['resolved'])
            if row['resolved']:rec.update(row['comparisons']['mscs_memory'])
            rows.append(rec)
    assert rows==s['rows'] and physical==s['new_candidate_forwards'] and logical==s['logical_candidate_queries']==144
    means={k:float(np.mean([r.get(k,0) for r in rows])) for k in ('anytime_advantage','endpoint_advantage')}
    per_workload={b:{k:float(np.mean([r.get(k,0) for r in rows if r['bench']==b])) for k in means} for b in plan['benchmarks']}
    counts={k:{status:sum(r.get(k+'_status')==status for r in rows) for status in ('win','tie','loss')} for k in ('anytime','endpoint')}
    gate=all(r['resolved'] for r in rows) and all(v>1e-6 for v in means.values()) and counts['anytime']['win']>=4 and sum(v['endpoint_advantage']>1e-6 for v in per_workload.values())>=2
    assert means==s['means'] and per_workload==s['per_workload'] and counts==s['counts'] and bool(gate)==s['continuation_gate_passed']
    return dict(passed=True,physical_equivalence_inventory_and_root_sampling_replayed=True,
                global_method_preselected=True,all_contexts_and_query_counts_preserved=True,aggregates_recomputed=True)


if __name__=='__main__':
    import json
    print(json.dumps(validate_all(HERE/'active_confirmation_s42_01'),indent=2))
