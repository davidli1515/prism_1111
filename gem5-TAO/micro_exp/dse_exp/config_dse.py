"""
Exp 5c: Design Space Definition

Key gem5 constraints:
  - renameWidth = dispatchWidth (always equal)
  - issueWidth  = dispatchWidth (gem5 implementation constraint)
  - l1d_size = l1i_size = l1_size (shared parameter)
"""

import numpy as np
from collections import OrderedDict

# ============================================================
# Parameter Definitions
# ============================================================

GRADIENT_PARAMS = OrderedDict({
    "fetchWidth":       (4, 12, 1),
    "decodeWidth":      (4, 12, 1),
    "dispatchWidth":    (4, 12, 1),   # = renameWidth = issueWidth
    "commitWidth":      (4, 12, 1),
    "numROBEntries":    (72, 1024, 1),
    "numIQEntries":     (32, 128, 1),
    "LQEntries":        (16, 256, 1),
    "SQEntries":        (16, 256, 1),
    "IntALU":           (1, 8, 1),
    "IntMultDiv":       (1, 8, 1),
    "FP_ALU":           (1, 8, 1),
    "FP_MultDiv":       (1, 8, 1),
    "RdWrPort":         (1, 8, 1),
    "l1_size":          [16, 32, 64, 128, 256],
    "l1d_assoc":        [2, 4, 8, 16, 32],
    "l1i_assoc":        [2, 4, 8, 16, 32],
})

NON_GRADIENT_PARAMS = OrderedDict({
    "wbWidth":          (4, 12, 1),
    "fetchBufferSize":  [8, 16, 32, 64],
    "bp_type":          [0, 1],
    "l2_size":          [512, 1024, 2048, 4096],
    "l2_assoc":         [2, 4, 8, 16, 32],
})

ALL_PARAMS = OrderedDict()
ALL_PARAMS.update(GRADIENT_PARAMS)
ALL_PARAMS.update(NON_GRADIENT_PARAMS)

# ============================================================
# gem_data index mapping (3850:3873)
# ============================================================

GEM_DATA_ARCH_START = 3850
GEM_DATA_ARCH_END = 3873

PARAM_TO_GD_INDEX = {
    "fetchWidth":      3850,
    "decodeWidth":     3851,
    # renameWidth at 3852 — set = dispatchWidth
    "dispatchWidth":   3853,
    # issueWidth at 3854 — set = dispatchWidth
    "wbWidth":         3855,
    "commitWidth":     3856,
    "numROBEntries":   3857,
    "numIQEntries":    3858,
    "LQEntries":       3859,
    "SQEntries":       3860,
    "fetchBufferSize": 3861,
    "IntALU":          3862,
    "IntMultDiv":      3863,
    "FP_ALU":          3864,
    "FP_MultDiv":      3865,
    "RdWrPort":        3866,
    "l1_size":         3867,
    "l2_size":         3868,
    "l1d_assoc":       3869,
    "l1i_assoc":       3870,
    "l2_assoc":        3871,
    "bp_type":         3872,
}

WMIN_PARAM_NAMES = ["fetchWidth", "decodeWidth", "dispatchWidth", "commitWidth"]

# ============================================================
# Default Configuration
# ============================================================

DEFAULT_CONFIG = {
    "fetchWidth": 8, "decodeWidth": 8, "dispatchWidth": 8,
    "commitWidth": 8, "wbWidth": 8,
    "numROBEntries": 192, "numIQEntries": 64,
    "LQEntries": 32, "SQEntries": 32,
    "fetchBufferSize": 32,
    "IntALU": 2, "IntMultDiv": 1, "FP_ALU": 2, "FP_MultDiv": 1, "RdWrPort": 2,
    "l1_size": 64, "l2_size": 1024,
    "l1d_assoc": 2, "l1i_assoc": 2, "l2_assoc": 8,
    "bp_type": 0,
}

# ============================================================
# Validity Constraints
# ============================================================

def is_valid_config(c):
    if not (c["fetchWidth"] >= c["decodeWidth"] >= c["dispatchWidth"]):
        return False
    if not (c["wbWidth"] >= c["dispatchWidth"]):
        return False
    if not (c["commitWidth"] >= c["dispatchWidth"]):
        return False
    DIQ = 2 * c["dispatchWidth"]
    if not (c["numROBEntries"] >= DIQ + c["numIQEntries"] + c["LQEntries"] + c["SQEntries"]):
        return False
    return True

# ============================================================
# Sampling
# ============================================================

def get_param_values(name):
    spec = ALL_PARAMS.get(name)
    if spec is None:
        raise ValueError(f"Unknown param: {name}")
    if isinstance(spec, list):
        return spec
    lo, hi, step = spec
    return list(range(lo, hi + 1, step))


def sample_random_config(rng=None):
    if rng is None:
        rng = np.random.RandomState()
    for _ in range(10000):
        c = {}
        for name in ALL_PARAMS:
            vals = get_param_values(name)
            c[name] = vals[rng.randint(len(vals))]
        widths = sorted([c["fetchWidth"], c["decodeWidth"], c["dispatchWidth"]], reverse=True)
        c["fetchWidth"], c["decodeWidth"], c["dispatchWidth"] = widths
        c["wbWidth"] = max(c["wbWidth"], c["dispatchWidth"])
        c["commitWidth"] = max(c["commitWidth"], c["dispatchWidth"])
        DIQ = 2 * c["dispatchWidth"]
        min_rob = DIQ + c["numIQEntries"] + c["LQEntries"] + c["SQEntries"]
        if c["numROBEntries"] < min_rob:
            c["numROBEntries"] = min(min_rob, 1024)
        if is_valid_config(c):
            return c
    raise RuntimeError("Failed to sample valid config")


def sample_n_configs(n, seed=42):
    rng = np.random.RandomState(seed)
    return [sample_random_config(rng) for _ in range(n)]

# ============================================================
# Conversions
# ============================================================

def config_to_arch_vector(config):
    vec = np.zeros(23, dtype=np.float32)
    for name, gd_idx in PARAM_TO_GD_INDEX.items():
        vec[gd_idx - GEM_DATA_ARCH_START] = config[name]
    vec[3852 - GEM_DATA_ARCH_START] = config["dispatchWidth"]  # renameWidth
    vec[3854 - GEM_DATA_ARCH_START] = config["dispatchWidth"]  # issueWidth
    return vec


def compute_w_min(config):
    return min(config[p] for p in WMIN_PARAM_NAMES)


def config_to_gem5_params(config):
    p = {name: str(config[name]) for name in ALL_PARAMS}
    p["renameWidth"] = str(config["dispatchWidth"])
    p["issueWidth"] = str(config["dispatchWidth"])
    p["l1_size"] = str(config["l1_size"])  # shared for l1d and l1i
    return p


def config_to_bp_choice(config):
    return "TAGE" if config["bp_type"] == 1 else "BiModeBP"


def estimate_design_space_size():
    total = 1
    for name in ALL_PARAMS:
        total *= len(get_param_values(name))
    return total


if __name__ == "__main__":
    print(f"Design space size: {estimate_design_space_size():.2e}")
    print(f"Default valid: {is_valid_config(DEFAULT_CONFIG)}")
    configs = sample_n_configs(5)
    for i, c in enumerate(configs):
        print(f"  Config {i}: fetchW={c['fetchWidth']}, decW={c['decodeWidth']}, "
              f"dispW={c['dispatchWidth']}(=renW=issW), ROB={c['numROBEntries']}, "
              f"IQ={c['numIQEntries']}, l1={c['l1_size']}KB, W_min={compute_w_min(c)}")