"""Conditional resource value from finite component response predictions."""
from itertools import combinations, product
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIOUS = ROOT/'mscs_jacobian_greedy_1010'
sys.path.insert(0, str(PREVIOUS))
from core import (LocalModel, Inputs, load_model, exact, contributions, config_key,
                  moves, legal, compute_w_min, DEFAULT_CKPT, NORM_CACHE)

MODES = ('scalar', 'mscs', 'no_decouple', 'no_protect', 'no_context')
TAU = .003  # Paper/refine_v3 reclaim tolerance, relative to the common root CPI.


def candidates(root):
    """One-stride singles and all legal two-resource up/up or up/down edits."""
    singles = moves(root, radius=1)
    out = [dict(s, kind='single', parts=[]) for s in singles]
    used = {config_key(c['config']) for c in out}
    up = [i for i, s in enumerate(singles) if s['sign'] == 1]
    down = [i for i, s in enumerate(singles) if s['sign'] == -1]
    for kind, pairs in (('up_up', combinations(up, 2)), ('up_down', product(up, down))):
        for i, j in pairs:
            a, b = singles[i], singles[j]
            if a['parameter'] == b['parameter']:
                continue
            c = dict(root)
            for s in (a, b):
                c[s['parameter']] = s['config'][s['parameter']]
            key = config_key(c)
            if key in used or not legal(c):
                continue
            assert compute_w_min(c) == compute_w_min(root)
            used.add(key)
            out.append({'kind': kind, 'parts': [i, j], 'config': c})
    return out


def context_rows(pool, cp, c0):
    rows = {}
    for k, candidate in enumerate(pool):
        if candidate['kind'] == 'single':
            continue
        u, v = candidate['parts']
        alone = (cp[v]-c0)/c0
        after = (cp[k]-cp[u])/c0
        interaction = (cp[k]-cp[u]-cp[v]+c0)/c0
        protect = candidate['kind'] == 'up_down' and 0 <= alone < TAU and after >= TAU
        rows[k] = {'interaction': float(interaction), 'alone_cost': float(alone),
                   'after_investment_cost': float(after), 'protect': bool(protect),
                   'investment_gain': float((c0-cp[u])/c0),
                   'joint_gain': float((c0-cp[k])/c0)}
    return rows


def select(pool, scalar, mscs, c0, mode):
    """One shared scalar anchor plus one independently ranked proposal.

    Protection only defers the MSCS proposal slot. It cannot veto a verified
    improvement or remove the shared scalar anchor.
    """
    assert mode in MODES
    scalar, mscs = np.asarray(scalar), np.asarray(mscs)
    scalar_ctx = context_rows(pool, scalar, c0)
    scalar_allowed = np.ones(len(pool), dtype=bool)
    for k, row in scalar_ctx.items():
        if row['protect']:
            scalar_allowed[k] = False
    order = np.argsort(scalar, kind='stable')
    anchor = next(int(i) for i in order if scalar_allowed[i])
    values = scalar.copy() if mode == 'scalar' else mscs.copy()
    allowed = scalar_allowed.copy() if mode == 'scalar' else np.ones(len(pool), dtype=bool)
    effective = context_rows(pool, values, c0)
    for k, c in enumerate(pool):
        if c['kind'] == 'single':
            continue
        remove = (c['kind'] == 'up_up' and mode in ('no_decouple', 'no_context') or
                  c['kind'] == 'up_down' and mode in ('no_protect', 'no_context'))
        if remove:
            u, v = c['parts']
            values[k] = max(1/compute_w_min(c['config']), mscs[u]+mscs[v]-c0)
        if mode in ('mscs', 'no_decouple') and effective[k]['protect']:
            allowed[k] = False
    ranking = np.argsort(values, kind='stable')
    second = next(int(i) for i in ranking if i != anchor and allowed[i])
    return {'mode': mode, 'chosen': [anchor, second], 'scores': values.tolist(),
            'allowed': allowed.tolist(), 'anchor': anchor,
            'protect_deferred': [int(i) for i in np.flatnonzero(~allowed)],
            'context_predictions': {str(k): v for k, v in effective.items()}}


def witnesses(pool, scalar, mscs, c0, full_selection):
    """At most one decoupling and one protection audit, fixed before labels."""
    ctx = context_rows(pool, mscs, c0)
    out = {}
    dec = [k for k, r in ctx.items() if pool[k]['kind'] == 'up_up'
           and r['interaction'] < -1e-8 and r['joint_gain'] > 0]
    selected = [k for k in full_selection['chosen'] if k in dec]
    if dec:
        out['decouple'] = selected[0] if selected else min(dec, key=lambda k: (ctx[k]['interaction'], k))
    protect = [k for k, r in ctx.items() if r['protect'] and r['investment_gain'] > 0]
    if protect:
        out['protect'] = max(protect, key=lambda k: (ctx[k]['interaction'], -k))
    return out
