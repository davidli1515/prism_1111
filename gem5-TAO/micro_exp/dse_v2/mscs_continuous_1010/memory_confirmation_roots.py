"""Outcome-free, prospective architecture selection; fixed seed, no retries."""
import itertools
import numpy as np
from secant_support import HERE, read, sha, config_key
from config_dse import DEFAULT_CONFIG
from core import legal
from width_priority_support import width_moves
from fresh_disagreement_support import configs_in

INVENTORY = ('coverage_s42_01', 'neighborhood_s42_01', 'priority_walk_s42_01',
             'feedback_probe_s42_01', 'interaction_diamond_s42_01',
             'pair_priority_s42_01', 'pair_walk_s42_01', 'pair_replication_s42_01',
             'paid_centers_s42_01', 'paid_centers_walk_s42_01')


def select():
    files = [p for name in INVENTORY for p in sorted((HERE/name).rglob('*.json'))]
    known = set()
    hashes = {}
    for p in files:
        found, unknown = configs_in(read(p))
        known.update(k for _, k in found)
        known.update(unknown)
        hashes[str(p)] = sha(p)
    actions = [a for a in width_moves(DEFAULT_CONFIG, 1, {config_key(DEFAULT_CONFIG)})
               if a['config']['fetchBufferSize'] >= 16]
    catalog = []
    used = set()
    for indices in itertools.combinations(range(len(actions)), 3):
        edits=[{k for k in DEFAULT_CONFIG if actions[i]['config'][k]!=DEFAULT_CONFIG[k]} for i in indices]
        if any(a & b for a,b in itertools.combinations(edits,2)):
            continue
        c=dict(DEFAULT_CONFIG)
        for i,keys in zip(indices,edits):c.update({k:actions[i]['config'][k] for k in keys})
        key=config_key(c)
        if not legal(c) or c['fetchBufferSize']<16 or key in used:continue
        used.add(key)
        catalog.append(dict(indices=list(indices),config=c,
                            edits={k:[DEFAULT_CONFIG[k],c[k]] for k in c if c[k]!=DEFAULT_CONFIG[k]}))
    order = np.random.RandomState(42).permutation(len(catalog)).tolist()
    eligible = [i for i in order if config_key(catalog[i]['config']) not in known]
    assert len(eligible) >= 2
    chosen = eligible[:2]
    return dict(seed=42, inventory_sha256=hashes, inventoried_config_count=len(known),
                actions=actions, catalog=catalog, permutation=order,
                selected_indices=chosen, selected=[catalog[i] for i in chosen])
