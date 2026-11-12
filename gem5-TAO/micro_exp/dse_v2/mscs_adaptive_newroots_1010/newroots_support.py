"""New root selection and harness constants; all search policies remain old code."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ADAPTIVE = ROOT/'mscs_adaptive_greedy_1010'
HYBRID = ROOT/'mscs_hybrid_greedy_1010'
sys.path.insert(0, str(ADAPTIVE))
from adaptive_support import (PARAMS, STEPS, TOPK, DEFAULT_CKPT, NORM_CACHE,
    config_key, compute_w_min, legal, moves, Inputs, load_model, RankAllocator)
from common import pareto_mask

ARMS = ('scalar_excess', 'hybrid', 'adaptive')
BENCHES = ('602.gcc_s', '605.mcf_s', '623.xalancbmk_s', '641.leela_s')
DEVELOPMENT = tuple(ROOT/name/'results_s42_01' for name in (
    'mscs_jacobian_greedy_1010', 'mscs_contextual_1010',
    'mscs_hybrid_greedy_1010', 'mscs_adaptive_greedy_1010'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def bench_of(text):
    return next((b for b in BENCHES if str(text).startswith(b)), None)


def collect_configs(obj, bench=None):
    """Extract configurations, not outcomes, including predicted candidate pools."""
    found = set()
    if isinstance(obj, dict):
        bench = bench_of(obj.get('bench')) or bench_of(obj.get('context')) or bench
        for name in ('root', 'config', 'next_root', 'final_config'):
            c = obj.get(name)
            if bench and isinstance(c,dict) and all(k in c for k in ('fetchWidth','LQEntries','bp_type')):
                found.add((bench,config_key(c)))
        for value in obj.values():
            if isinstance(value,(dict,list)):
                found.update(collect_configs(value,bench))
    elif isinstance(obj,list):
        for value in obj:
            if isinstance(value,(dict,list)):
                found.update(collect_configs(value,bench))
    return found


def exposure_inventory():
    used, files = set(), {}
    for directory in DEVELOPMENT:
        for path in sorted(directory.rglob('*.json')):
            bench = next((bench_of(part) for part in path.parts if bench_of(part)),None)
            used.update(collect_configs(read(path),bench))
            files[str(path)] = sha(path)
    counts = {b:sum(bb == b for bb,_ in used) for b in BENCHES}
    return used, files, counts


def nearest_unexposed(front_configs, bench, exposed, legal_fn=legal):
    """Area-sorted configs in; no CPI/gain/pressure arguments or model access."""
    target = 3*(len(front_configs)-1)//4
    order = sorted(range(len(front_configs)),key=lambda rank:(abs(rank-target),rank))
    eligible = [legal_fn(c) and (bench,config_key(c)) not in exposed for c in front_configs]
    rank = next((rank for rank in order if eligible[rank]),None)
    if rank is None:
        raise ValueError(f'No unexposed legal root on {bench} front')
    return rank, target, eligible


def bind_runners(out, start):
    """Only redirect output/timer globals; call original walk objects unchanged."""
    import run_adaptive
    import run_hybrid
    for module in (run_adaptive,run_hybrid):
        module.OUT = Path(out)
        module.START = start
    return run_hybrid, run_adaptive
