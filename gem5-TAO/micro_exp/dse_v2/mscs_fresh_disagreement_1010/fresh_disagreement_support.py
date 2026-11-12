"""Only the experiment harness is new; prediction and policy code are reused."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIOUS = ROOT/'mscs_matched_steps_1010'
sys.path.insert(0, str(PREVIOUS))
from matched_support import (read, sha, save, label_id, choices, pool_indices,
                             outcome, pressure, ARMS, KINDS)
from local_support import (LocalAllocator, LocalModel, Inputs, FeatureCache,
                           load_model, exact, config_key, compute_w_min, legal,
                           moves, DEFAULT_CKPT, contributions)
from common import ALL_PARAMS, PRECOMPUTE, NORM_CACHE, SPEC17, pareto_mask

OUT = HERE/'results_s42_01'
BENCHES = ('602.gcc_s', '605.mcf_s')
MAX_SCREEN = 4


def bench_of(value):
    return next((b for b in SPEC17 if str(value).startswith(b)), None)


def configs_in(obj, hint=None):
    """Find whole configs, not just dicts with a particular enclosing field name."""
    found, unknown = set(), set()
    if isinstance(obj, dict):
        bench = next((bench_of(obj.get(k)) for k in ('bench','benchmark','context','id','root_id')
                      if bench_of(obj.get(k))), hint)
        if all(k in obj for k in ALL_PARAMS) and all(isinstance(obj[k], (int, float)) for k in ALL_PARAMS):
            config = {k: obj[k] for k in ALL_PARAMS}; key = config_key(config)
            if bench:
                found.add((bench, key))
            else:
                unknown.add(key)
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                f, u = configs_in(value, bench_of(key) or bench)
                found.update(f); unknown.update(u)
    elif isinstance(obj, list):
        for value in obj:
            f, u = configs_in(value, hint); found.update(f); unknown.update(u)
    return found, unknown


def array_save(path, **arrays):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **{k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                                 for k, v in arrays.items() if v is not None})


def verify_sources(plan):
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest, path
