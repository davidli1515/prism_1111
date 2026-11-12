"""Exclude observed hardware configurations, not merely unlabeled catalogs."""
import itertools
import numpy as np
from secant_support import HERE,read,sha,config_key
from config_dse import DEFAULT_CONFIG
from core import legal
from width_priority_support import width_moves
from fresh_disagreement_support import configs_in

INVENTORY=('coverage_s42_01','neighborhood_s42_01','priority_walk_s42_01',
           'feedback_probe_s42_01','interaction_diamond_s42_01','pair_priority_s42_01',
           'pair_walk_s42_01','pair_replication_s42_01','paid_centers_s42_01',
           'paid_centers_walk_s42_01','memory_confirmation_s42_01','active_resources_s42_01')
INACTIVE=frozenset(('l2_size','l2_assoc'))


def physical_key(key):return tuple((k,v) for k,v in key if k not in INACTIVE)


def select():
    assert read(HERE/'memory_confirmation_s42_01/validation.json')['passed']
    assert read(HERE/'active_resources_s42_01/validation.json')['passed']
    assert read(HERE/'active_resources_s42_01/summary.json')['eligible_methods']==['mscs_memory']
    files=[p for name in INVENTORY for p in sorted((HERE/name).rglob('*.json'))]
    known=set();hashes={}
    for p in files:
        obj=read(p)
        if p.name in ('plan.json','root_plan.json'):
            obj={k:obj[k] for k in ('root','selected','contexts') if k in obj}
        found,unknown=configs_in(obj)
        known.update(physical_key(k) for _,k in found);known.update(physical_key(k) for k in unknown)
        hashes[str(p)]=sha(p)
    actions=[a for a in width_moves(DEFAULT_CONFIG,1,{config_key(DEFAULT_CONFIG)})
             if a['config']['fetchBufferSize']>=16 and all(a['config'][k]==DEFAULT_CONFIG[k] for k in INACTIVE)]
    catalog=[];used=set()
    for indices in itertools.combinations(range(len(actions)),3):
        edits=[{k for k in DEFAULT_CONFIG if actions[i]['config'][k]!=DEFAULT_CONFIG[k]} for i in indices]
        if any(a & b for a,b in itertools.combinations(edits,2)):continue
        c=dict(DEFAULT_CONFIG)
        for i,keys in zip(indices,edits):c.update({k:actions[i]['config'][k] for k in keys})
        key=physical_key(config_key(c))
        if not legal(c) or key in used:continue
        used.add(key);catalog.append(dict(indices=list(indices),config=c,edits={k:[DEFAULT_CONFIG[k],c[k]] for k in c if c[k]!=DEFAULT_CONFIG[k]}))
    order=np.random.RandomState(42).permutation(len(catalog)).tolist()
    chosen=[i for i in order if physical_key(config_key(catalog[i]['config'])) not in known][:2]
    assert len(chosen)==2
    return dict(seed=42,inventory_sha256=hashes,inventoried_hardware_config_count=len(known),
                actions=actions,catalog=catalog,permutation=order,selected_indices=chosen,
                selected=[catalog[i] for i in chosen],selected_method='mscs_memory')
