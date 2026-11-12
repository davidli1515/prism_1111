"""Outcome-independent root and workload selection, with frozen file inventory."""
import numpy as np
from secant_support import HERE,read,sha,config_key
from active_confirmation_roots import physical_key,INVENTORY
from fresh_disagreement_support import configs_in


def select():
    assert read(HERE/'guarded_greedy_s42_01/validation.json')['passed']
    assert read(HERE/'guarded_greedy_s42_01/summary.json')['development_gate_passed']
    previous=read(HERE/'active_confirmation_s42_01/root_plan.json')
    files=[p for name in (*INVENTORY,'active_confirmation_s42_01','withdrawal_guard_s42_01','guarded_greedy_s42_01')
           for p in sorted((HERE/name).rglob('*.json'))]
    hashes={};known=set()
    for p in files:
        obj=read(p)
        if p.name in ('plan.json','root_plan.json'):
            obj={k:obj[k] for k in ('root','selected','contexts') if k in obj}
        found,unknown=configs_in(obj)
        known.update(physical_key(k) for _,k in found);known.update(physical_key(k) for k in unknown)
        hashes[str(p)]=sha(p)
    catalog=previous['catalog'];order=np.random.RandomState(42).permutation(len(catalog)).tolist()
    index=next(i for i in order if physical_key(config_key(catalog[i]['config'])) not in known)
    allb=read(HERE/'coverage_s42_01/plan.json')['workloads']
    excluded=['602.gcc_s','605.mcf_s','619.lbm_s','623.xalancbmk_s','641.leela_s','654.roms_s']
    pool=[b for b in allb if b not in excluded];perm=np.random.RandomState(42).permutation(len(pool)).tolist()
    return dict(seed=42,root=catalog[index]['config'],selected_index=index,selected_edits=catalog[index]['edits'],
        catalog_source=str(HERE/'active_confirmation_s42_01/root_plan.json'),inventory_sha256=hashes,
        known_hardware_count=len(known),root_permutation=order,excluded_workloads=excluded,workload_pool=pool,
        workload_permutation=perm,benchmarks=[pool[i] for i in perm[:2]])
