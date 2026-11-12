"""Frozen predictions, pools and pair choices. No trial-label access here."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCAL = ROOT/'mscs_local_feedback_1010'
sys.path.insert(0, str(LOCAL))
from local_support import LocalAllocator, config_key, compute_w_min, DEFAULT_CKPT
from core import STRIDE, get_param_values

OUT = HERE/'results_s42_01'
SOURCES = (LOCAL/'results_s42_01', LOCAL/'fresh_s42_01')
ARMS = ('scalar', 'mscs', 'hybrid', 'local')
KINDS = ('full', 'stride1', 'stride2', 'stride4')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False,
                  default=lambda x: x.tolist() if hasattr(x, 'tolist') else x.item())


def label_id(bench, config):
    return hashlib.sha256(json.dumps([bench, config], sort_keys=True).encode()).hexdigest()


def pool_indices(d, kind):
    if kind == 'full':
        return list(range(len(d['candidates'])))
    step = int(kind[-1]); ids = []
    for i, a in enumerate(d['candidates']):
        p = a['parameter']; vals = get_param_values(p)
        displacement = vals.index(a['config'][p])-vals.index(d['root'][p])
        if a['stride'] == step and displacement == a['sign']*step*STRIDE.get(p, 1):
            ids.append(i)
    return ids


def choices(d, ids):
    actions = [d['candidates'][i] for i in ids]
    s = np.array(d['scalar_predicted_cpis'])[ids]
    m = np.array(d['mscs_predicted_cpis'])[ids]
    so = np.argsort(s, kind='stable').tolist()
    mo = np.argsort(m, kind='stable').tolist()
    policy = LocalAllocator(); policy.state = copy.deepcopy(d['allocator_before'])
    lo, _, _ = policy.choose(actions, s, m)
    assert policy.snapshot() == d['allocator_before']
    assert len(so) >= 2
    orders = dict(scalar=so[:2], mscs=mo[:2],
                  hybrid=[so[0]]+[i for i in mo if i != so[0]][:1], local=lo)
    return {arm: [ids[i] for i in order] for arm, order in orders.items()}


def pressure(m, width):
    cpi = np.minimum(100., 1/(np.maximum(.01, m[..., 0])*width+1e-8))
    return (cpi[..., None]*m[..., 1:]).mean(axis=-3)


def outcome(root_cpi, cap, records, constrained=True):
    """Return an interval when an improving candidate has unknown area."""
    eligible = [r for r in records if r['cpi'] < root_cpi*(1-1e-6)]
    known = [r for r in eligible if not constrained or
             (r['area'] is not None and r['area'] <= cap+1e-9)]
    unknown = [r for r in eligible if constrained and r['area'] is None]
    high = min([root_cpi]+[r['cpi'] for r in known])
    low = min([high]+[r['cpi'] for r in unknown])
    return dict(best_cpi=high if low == high else None, cpi_lower=low, cpi_upper=high,
                gain_pct=100*(1-high/root_cpi) if low == high else None,
                resolved=low == high, unknown_improving=len(unknown))
