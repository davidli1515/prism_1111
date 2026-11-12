"""
17-parameter design space: the 21 DSE parameters minus the four that the trace features do not
model (precompute uses L2 = 1024 KB / 8-way, TAGE, and clamps the fetch-buffer index), fixed at the
values the features were generated with.  Originals in dse_exp/ are imported, never modified.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CA = os.path.join(os.path.dirname(HERE), "coord_ablation")
if CA not in sys.path:
    sys.path.insert(0, CA)
import coord as C                                   # adds dse_exp/ and exp1/ to sys.path
from config_dse import ALL_PARAMS, sample_random_config, is_valid_config
from dse_engine import _project_to_valid

FIXED = {"l2_size": 1024,        # process_traces.sh L2_KB=1024
         "l2_assoc": 8,          # prepare_base_features.py l2_assoc=8
         "bp_type": 1,           # BP_TYPE=tage (validate_joint_dse: 1 -> TAGE)
         "fetchBufferSize": 32}  # feature index clamped for 16/32/64; 8 makes McPAT fail
SEARCH = [p for p in ALL_PARAMS if p not in FIXED]
assert len(SEARCH) == 17


def project17(c):
    c = dict(c); c.update(FIXED)
    c = _project_to_valid(c)             # also caps L1 assoc by L1 size
    assert all(c[k] == v for k, v in FIXED.items())
    return c


def sample17(rng=None):
    for _ in range(1000):
        c = project17(sample_random_config(rng))
        if is_valid_config(c):
            return c
    raise RuntimeError("no valid 17-param config")


class patched_bo_sampler:
    """Context manager: make dse_engine.run_bo_gem draw candidates from the 17-param space
    (runtime attribute swap on the imported module; the source file is untouched)."""
    def __enter__(self):
        self.orig = C.DE.sample_random_config
        C.DE.sample_random_config = sample17
        return self

    def __exit__(self, *a):
        C.DE.sample_random_config = self.orig
