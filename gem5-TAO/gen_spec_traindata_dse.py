#!/usr/bin/env python3
"""
Full Training Data Generation Pipeline
=======================================
Pipeline:  gem5 (with checkpoint restore + random micro-arch config)
         → concorde_features (trace → 3873-dim vector + 4×4 MSCS)
         → parse stats.txt for ground-truth CPI
         → save (input, output_mscs, output_cpi, config)

Usage examples:
  # Run all benchmarks, 5 random configs per checkpoint, 100K instructions
  python3 gen_spec_traindata.py --insts-default 100000 --num-samples 5 --workers 20

  # Run only gcc, 10 samples
  python3 gen_spec_traindata.py --only 602.gcc_s --insts-default 100000 --num-samples 10 --workers 20

  # Specify custom output directory
  python3 gen_spec_traindata.py --insts-default 100000 --num-samples 5 --workers 20 --output-dir /scratch/train_data
"""

import argparse, os, re, json, shlex, subprocess, hashlib, glob, sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
import random
import time
import traceback

# ============================================================
# Global Configuration
# ============================================================
GEM5_DIR        = "/home/guest/wl730/gem5-TAO"
GEM5_BIN        = "build/ARM/gem5.opt"                    # relative to GEM5_DIR
GEM5_CONFIG     = "configs/example/se_new.py"             # relative to GEM5_DIR
CONCORDE_BIN    = os.path.join(GEM5_DIR, "concorde_features")
SPEC_BASE       = "/home/guest/wl730/spec17_20_benchmarks_minimal"
CKPT_BASE       = os.path.join(SPEC_BASE, "m5out_cpts")
DEFAULT_OUT_DIR = os.path.join(GEM5_DIR, "train_data")
MAX_CORES       = 20
TIMEOUT_SECS    = 7200   # 2 hours per gem5 run

# Data generation constants (53 ckpts × 800 slices × 20 configs ≈ 848K datapoints)
SLICES_PER_CKPT   = 800       # number of 100K-instruction slices per checkpoint
CONFIGS_PER_SLICE = 20        # random micro-arch configs per slice
INST_PER_SLICE    = 100_000   # 10万 instructions per slice (-I)
SLICE_STRIDE      = 100_000   # 10万 instructions between slice starts
WARMUP_INSTS      = 20_000_000  # 2000万 warmup instructions (-W)

# ============================================================
# Benchmark Command Templates  (paths updated to new location)
# ============================================================
# All paths: /home/wl730/spec17/benchspec/CPU/XXX  →  /home/guest/wl730/spec17_20_benchmarks_minimal/XXX
# Config:    /scratch/gem5-TAO/configs/...          →  configs/... (relative, run from GEM5_DIR)
# NOTE: -I and -P flags are injected dynamically; the -I here is a placeholder.
#       --fast-forward / --standard-switch / -W are REMOVED; checkpoint restore is used instead.

_B = SPEC_BASE  # shorthand

COMMANDS = {
    "602.gcc_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/602.gcc_s/exe/sgcc_base.mytest-64"'
        f' -o "{_B}/602.gcc_s/run/run_base_refspeed_mytest-64.0000/gcc-pp.c'
        f' -O5 -fipa-pta -o gcc-pp.opts-O5_-fipa-pta.s"'
        f' -I 100000'
    ),
    "600.perlbench_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/600.perlbench_s/exe/perlbench_s_base.mytest-64"'
        f' --options="-I{_B}/600.perlbench_s/run/run_base_refspeed_mytest-64.0000/lib'
        f' {_B}/600.perlbench_s/run/run_base_refspeed_mytest-64.0000/checkspam.pl'
        f' 2500 5 25 11 150 1 1 1 1"'
        f' -I 100000'
    ),
    "619.lbm_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/619.lbm_s/exe/lbm_s_base.mytest-64"'
        f' --options="2000 reference.out 0 0 {_B}/619.lbm_s/data/refspeed/input/200_200_260_ldc.of"'
        f' -I 100000'
    ),
    "605.mcf_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/605.mcf_s/exe/mcf_s_base.mytest-64"'
        f' -o "{_B}/605.mcf_s/run/run_base_refspeed_mytest-64.0000/inp.in"'
        f' -I 100000'
    ),
    "620.omnetpp_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/620.omnetpp_s/run/run_base_refspeed_mytest-64.0000/omnetpp_s_base.mytest-64"'
        f' --options="-c General -r 0"'
        f' -I 100000'
    ),
    "623.xalancbmk_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/623.xalancbmk_s/exe/xalancbmk_s_base.mytest-64"'
        f' -o "-v {_B}/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/t5.xml'
        f' {_B}/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/xalanc.xsl"'
        f' -I 100000'
    ),
    "631.deepsjeng_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=16GB'
        f' -c "{_B}/631.deepsjeng_s/run/run_base_refspeed_mytest-64.0000/deepsjeng_s_base.mytest-64"'
        f' -o "{_B}/631.deepsjeng_s/run/run_base_refspeed_mytest-64.0000/ref.txt"'
        f' -I 100000'
    ),
    "641.leela_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/641.leela_s/exe/leela_s_base.mytest-64"'
        f' -o "{_B}/641.leela_s/run/run_base_refspeed_mytest-64.0000/ref.sgf"'
        f' -I 100000'
    ),
    "644.nab_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/644.nab_s/run/run_base_refspeed_mytest-64.0000/nab_s_base.mytest-64"'
        f' -o "3j1n 20140317 220"'
        f' -I 100000'
    ),
    "638.imagick_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/638.imagick_s/exe/imagick_s_base.mytest-64"'
        f' -o "-limit disk 0 {_B}/638.imagick_s/run/run_base_refspeed_mytest-64.0000/refspeed_input.tga'
        f' -resize 817% -rotate -2.76 -shave 540x375 -alpha remove -auto-level'
        f' -contrast-stretch 1x1% -colorspace Lab -channel R -equalize +channel -colorspace sRGB'
        f' -define histogram:unique-colors=false -adaptiveblur 0x5 -despeckle -auto-gamma'
        f' -adaptive-sharpen 55 -enhance -brightness-contrast 10x10 -resize 30%'
        f' {_B}/638.imagick_s/run/run_base_refspeed_mytest-64.0000/refspeed_output.tga"'
        f' -I 100000'
    ),
    "657.xz_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=16GB'
        f' -c "{_B}/657.xz_s/run/run_base_refspeed_mytest-64.0000/xz_s_base.mytest-64"'
        f' -o "{_B}/657.xz_s/run/run_base_refspeed_mytest-64.0000/cpu2006docs.tar.xz'
        f' 6643 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae'
        f' 1036078272 1111795472 4"'
        f' -I 100000'
    ),
    "648.exchange2_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/648.exchange2_s/exe/exchange2_s_base.mytest-64"'
        f' -o "6"'
        f' -I 100000'
    ),
    "607.cactuBSSN_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/607.cactuBSSN_s/run/run_base_test_mytest-64.0000/cactuBSSN_s_base.mytest-64"'
        f' -o "{_B}/607.cactuBSSN_s/run/run_base_refspeed_mytest-64.0000/spec_ref.par"'
        f' -I 100000'
    ),
    "621.wrf_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/621.wrf_s/run/run_base_refspeed_mytest-64.0000/wrf_s_base.mytest-64"'
        f' -I 100000'
    ),
    "508.namd_r": (
        f'{GEM5_BIN} --debug-flags=O3PipeView --debug-file=arm_timetrace.out {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/508.namd_r/run/run_base_refrate_mytest-64.0000/namd_r_base.mytest-64"'
        f' -o "--input {_B}/508.namd_r/run/run_base_refrate_mytest-64.0000/apoa1.input'
        f' --output apoa1.ref.output --iterations 65"'
        f' -I 100000'
    ),
    "526.blender_r": (
        f'{GEM5_BIN} --debug-flags=O3PipeView --debug-file=arm_timetrace.out {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/526.blender_r/run/run_base_refrate_mytest-64.0000/blender_r_base.mytest-64"'
        f' -o "{_B}/526.blender_r/data/refrate/input/sh3_no_char.blend'
        f' --render-output sh3_no_char_ --threads 1 -b -F RAWTGA -s 849 -e 849 -a"'
        f' -I 100000'
    ),
    "511.povray_r": (
        f'{GEM5_BIN} --debug-flags=O3PipeView --debug-file=arm_timetrace.out {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/511.povray_r/run/run_base_refrate_mytest-64.0000/povray_r_base.mytest-64"'
        f' -o "SPEC-benchmark-ref.ini +L{_B}/511.povray_r/run/run_base_refrate_mytest-64.0000"'
        f' -I 100000'
    ),
    "603.bwaves_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView --debug-file=arm_timetrace.out {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/603.bwaves_s/run/run_base_refspeed_mytest-64.0000/speed_bwaves_base.mytest-64"'
        f' -o "bwaves_1"'
        f' --input "{_B}/603.bwaves_s/run/run_base_refspeed_mytest-64.0000/bwaves_1.in"'
        f' -I 100000'
    ),
    "654.roms_s": (
        f'{GEM5_BIN} --debug-flags=O3PipeView --debug-file=arm_timetrace.out {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=4GB'
        f' -c "{_B}/654.roms_s/run/run_base_refspeed_mytest-64.0000/sroms_base.mytest-64"'
        f' --input "{_B}/654.roms_s/run/run_base_refspeed_mytest-64.0000/ocean_benchmark3.in.x"'
        f' -I 100000'
    ),
}


# ============================================================
# Checkpoint Discovery
# ============================================================
def discover_checkpoints(bench_name):
    """Find valid SimPoint checkpoints for a benchmark.
    gem5 -r is 1-indexed over ALL sorted tick subdirs, so we must
    preserve the original index even if some subdirs lack a cpt file.
    Returns list of (ckpt_r_1based, ckpt_tick_name) for valid dirs only.
    """
    ckpt_dir = os.path.join(CKPT_BASE, bench_name)
    if not os.path.isdir(ckpt_dir):
        return []
    subdirs = sorted([
        d for d in os.listdir(ckpt_dir)
        if os.path.isdir(os.path.join(ckpt_dir, d)) and d.isdigit()
    ], key=lambda x: int(x))
    valid = []
    for i, d in enumerate(subdirs):
        if glob.glob(os.path.join(ckpt_dir, d, "cpt.*")):
            valid.append((i + 1, d))  # gem5 -r is 1-indexed
    return valid


# ============================================================
# Random Micro-architecture Sampling (from your paper's Table 3)
# ============================================================
def sample_random_config():
    """Sample a random micro-architecture configuration.
    Returns (params_dict, bp_choice_str).
    - l1_size: single value used for BOTH l1d and l1i
    - bp_type: 0=Simple(BiMode), 1=TAGE  (included in the 23-dim vector)
    - prefetcher: fixed (not a tunable param)
    """
    l1_options = ["16", "32", "64", "128", "256"]
    l2_options = ["512", "1024", "2048", "4096"]
    assoc_options = ["2", "4", "8", "16", "32"]
    bp_choice = random.choice(["BiModeBP", "TAGE"])

    params = {
        # Pipeline Widths (1-12)
        "fetchWidth":    str(random.randint(1, 12)),
        "decodeWidth":   str(random.randint(1, 12)),
        "renameWidth":   str(random.randint(1, 12)),
        "dispatchWidth": str(random.randint(1, 12)),
        "issueWidth":    str(random.randint(1, 12)),
        "wbWidth":       str(random.randint(1, 12)),
        "commitWidth":   str(random.randint(1, 12)),
        # Buffer/Queue Sizes
        "numROBEntries":   str(random.randint(1, 1024)),
        "numIQEntries":    str(random.randint(16, 128)),
        "LQEntries":       str(random.randint(1, 256)),
        "SQEntries":       str(random.randint(1, 256)),
        "fetchBufferSize": str(random.randint(1, 8)),
        # Functional Units (1-8)
        "IntALU":     str(random.randint(1, 8)),
        "IntMultDiv": str(random.randint(1, 8)),
        "FP_ALU":     str(random.randint(1, 8)),
        "FP_MultDiv": str(random.randint(1, 8)),
        "RdWrPort":   str(random.randint(1, 8)),
        # Caches (l1_size is shared for l1d and l1i)
        "l1_size":     random.choice(l1_options),
        "l2_size":     random.choice(l2_options),
        "l1d_assoc":   random.choice(assoc_options),
        "l1i_assoc":   random.choice(assoc_options),
        "l2_assoc":    random.choice(assoc_options),
        # Branch predictor (0=Simple/BiMode, 1=TAGE)
        "bp_type":     0 if bp_choice == "BiModeBP" else 1,
    }
    return params, bp_choice


# ============================================================
# 23-dim Architecture Parameter Vector
# (matches the last 23 dimensions of the 3873-dim Concorde vector)
# l1_size is shared for l1d and l1i; bp_type: 0=Simple, 1=TAGE
# ============================================================
ARCH_PARAM_KEYS = [
    "fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth",
    "issueWidth", "wbWidth", "commitWidth",
    "numROBEntries", "numIQEntries", "LQEntries", "SQEntries",
    "fetchBufferSize",
    "IntALU", "IntMultDiv", "FP_ALU", "FP_MultDiv", "RdWrPort",
    "l1_size", "l2_size",
    "l1d_assoc", "l1i_assoc", "l2_assoc",
    "bp_type",
]
ARCH_PARAM_DIM = len(ARCH_PARAM_KEYS)  # 23


def params_to_vector(params):
    """Convert params dict to a float32 numpy vector of ARCH_PARAM_DIM values,
    in the fixed order defined by ARCH_PARAM_KEYS."""
    return np.array([float(params[k]) for k in ARCH_PARAM_KEYS], dtype=np.float32)


def config_to_hash(params):
    """Short 8-char hash of the config for directory naming."""
    s = json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def config_to_dirname(params):
    """Human-readable directory name from ALL 23 parameters + hash for uniqueness."""
    p = params
    return (
        f"fw{p['fetchWidth']}_dw{p['decodeWidth']}_rw{p['renameWidth']}"
        f"_dpw{p['dispatchWidth']}_iw{p['issueWidth']}_wbw{p['wbWidth']}"
        f"_cw{p['commitWidth']}"
        f"_rob{p['numROBEntries']}_iq{p['numIQEntries']}"
        f"_lq{p['LQEntries']}_sq{p['SQEntries']}_fb{p['fetchBufferSize']}"
        f"_ialu{p['IntALU']}_imd{p['IntMultDiv']}"
        f"_falu{p['FP_ALU']}_fmd{p['FP_MultDiv']}_rdwr{p['RdWrPort']}"
        f"_l1{p['l1_size']}_l2{p['l2_size']}"
        f"_da{p['l1d_assoc']}_ia{p['l1i_assoc']}_la{p['l2_assoc']}"
        f"_bp{p['bp_type']}"
        f"_{config_to_hash(params)}"
    )


# ============================================================
# Build the gem5 Command
# ============================================================
def build_gem5_command(bench, params, bp_choice,
                       ckpt_dir, ckpt_r, out_dir, trace_path,
                       fast_forward, warmup, inst_count):
    """Construct the full gem5 argv list.

    After restoring from checkpoint:
      1. fast-forward `fast_forward` instructions (AtomicSimpleCPU)
      2. switch to DerivO3CPU (standard-switch = fast_forward)
      3. warmup `warmup` instructions
      4. simulate `inst_count` instructions (measured)
    """
    argv = shlex.split(COMMANDS[bench], posix=True)

    # Remove template -I
    i = 0
    while i < len(argv):
        if argv[i] == "-I" and i + 1 < len(argv):
            del argv[i:i + 2]
        else:
            i += 1

    # -d <out_dir> right after gem5 binary
    argv.insert(1, "-d")
    argv.insert(2, out_dir)

    # Checkpoint restore + fast-forward + switch + warmup + inst count
    argv += [
        f"--checkpoint-dir={ckpt_dir}",
        "-r", str(ckpt_r),
        f"--fast-forward={fast_forward}",
        f"--standard-switch={fast_forward}",
        "-W", str(warmup),
        "-I", str(inst_count),
    ]

    # Trace dump path
    argv += ["-P", f"system.switch_cpus_1[0].dump_inst_trace_path='{trace_path}'"]

    # Random micro-arch parameters
    cpu = "system.switch_cpus_1[0]"
    argv += [
        "-P", f"{cpu}.fetchWidth={params['fetchWidth']}",
        "-P", f"{cpu}.decodeWidth={params['decodeWidth']}",
        "-P", f"{cpu}.renameWidth={params['renameWidth']}",
        "-P", f"{cpu}.dispatchWidth={params['dispatchWidth']}",
        "-P", f"{cpu}.issueWidth={params['issueWidth']}",
        "-P", f"{cpu}.wbWidth={params['wbWidth']}",
        "-P", f"{cpu}.commitWidth={params['commitWidth']}",
        "-P", f"{cpu}.LQEntries={params['LQEntries']}",
        "-P", f"{cpu}.SQEntries={params['SQEntries']}",
        "-P", f"{cpu}.numROBEntries={params['numROBEntries']}",
        "-P", f"{cpu}.numIQEntries={params['numIQEntries']}",
        "-P", f"{cpu}.fetchBufferSize={params['fetchBufferSize']}",
        "-P", f"{cpu}.fupool.FUList[0].count={params['IntALU']}",
        "-P", f"{cpu}.fupool.FUList[1].count={params['IntMultDiv']}",
        "-P", f"{cpu}.fupool.FUList[2].count={params['FP_ALU']}",
        "-P", f"{cpu}.fupool.FUList[3].count={params['FP_MultDiv']}",
        "-P", f"{cpu}.fupool.FUList[8].count={params['RdWrPort']}",
        f"--bp-type={bp_choice}",
        "--l1d_size", f"{params['l1_size']}kB",
        "--l1i_size", f"{params['l1_size']}kB",
        "--l2_size",  f"{params['l2_size']}kB",
        "--l3_size",  "4096kB",
        "--l1d_assoc", params['l1d_assoc'],
        "--l1i_assoc", params['l1i_assoc'],
        "--l2_assoc",  params['l2_assoc'],
    ]
    return argv


# ============================================================
# Parse CPI from gem5 stats.txt
# ============================================================
def parse_cpi_from_stats(stats_path):
    """Extract CPI from gem5 stats.txt.
    Tries switch_cpus_1 first (standard-switch mode), then system.cpu.
    Returns float CPI or None on failure.
    """
    if not os.path.isfile(stats_path):
        return None

    # Patterns to try, in order of preference
    patterns = [
        r"system\.switch_cpus_1\S*\.cpi\s+([\d.]+)",   # standard-switch detailed CPU
        r"system\.switch_cpus\S*\.cpi\s+([\d.]+)",      # alternative switch name
        r"system\.cpu\S*\.cpi\s+([\d.]+)",              # direct CPU
    ]

    with open(stats_path, "r") as f:
        content = f.read()

    for pat in patterns:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            return float(m.group(1))

    # Fallback: compute from numCycles / committedInsts
    cycle_match = re.search(r"system\.switch_cpus_1\S*\.numCycles\s+(\d+)", content)
    inst_match  = re.search(r"system\.switch_cpus_1\S*\.committedInsts\s+(\d+)", content)
    if cycle_match and inst_match:
        cycles = int(cycle_match.group(1))
        insts  = int(inst_match.group(1))
        if insts > 0:
            return cycles / insts

    return None


# ============================================================
# Run concorde_features
# ============================================================
# Path to the external simulator binary that concorde_features calls internally.
# The user said "我会设置成external simulator的绝对路径，你不用管",
# so set this to whatever your external sim path is:
CONCORDE_SIM_PATH = "/home/guest/wl730/gem5-TAO/build/ARM/gem5.opt"


def run_concorde_features(m5out_dir, params, bp_choice):
    """Run concorde_features on a completed gem5 datapoint.

    Interface:
      ./concorde_features /path/to/dp --sim-path /my/sim --l1 <l1d_size> --l2 <l2_size> --bp <tage|bimode>

    Where /path/to/dp = m5out_dir (gem5 output directory for this datapoint).

    Outputs written INTO m5out_dir:
      - 3873_vector.npy   (3873-dim feature vector)
      - cpi.npy           (scalar CPI from concorde analytical model)
      - mscs.npy          (4×4 MSCS matrix)

    After loading 3873_vector.npy, we overwrite the last 23 dims with the
    actual sampled architecture parameters.

    Returns (features_3873, mscs_4x4, concorde_cpi) as numpy arrays,
    or (None, None, None) on failure.
    """
    # Map bp_choice to the flag concorde_features expects
    bp_flag = "tage" if bp_choice == "TAGE" else "bimode"

    cmd = [
        CONCORDE_BIN,
        m5out_dir,
        "--sim-path", CONCORDE_SIM_PATH,
        "--l1", str(params["l1_size"]),
        "--l2", str(params["l2_size"]),
        "--bp", bp_flag,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=GEM5_DIR,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes
        )

        if result.returncode != 0:
            print(f"    [WARN] concorde_features failed (rc={result.returncode}): "
                  f"{result.stderr[:500]}")
            return None, None, None

        # ---- Load the three output .npy files from m5out_dir ----
        feat_path = os.path.join(m5out_dir, "3873_vector.npy")
        mscs_path = os.path.join(m5out_dir, "mscs.npy")
        cpi_path  = os.path.join(m5out_dir, "cpi.npy")

        features = None
        mscs = None
        concorde_cpi = None

        if os.path.isfile(feat_path):
            features = np.load(feat_path).astype(np.float32)
        else:
            print(f"    [WARN] 3873_vector.npy not found in {m5out_dir}")

        if os.path.isfile(mscs_path):
            mscs = np.load(mscs_path).astype(np.float32)
            if mscs.size == 16:
                mscs = mscs.reshape(4, 4)
        else:
            print(f"    [WARN] mscs.npy not found in {m5out_dir}")

        if os.path.isfile(cpi_path):
            concorde_cpi = float(np.load(cpi_path))
        else:
            print(f"    [WARN] cpi.npy not found in {m5out_dir}")

        # ---- Overwrite last 23 dims of 3873_vector with actual arch params ----
        if features is not None and features.shape[0] >= ARCH_PARAM_DIM:
            param_vec = params_to_vector(params)
            features[-ARCH_PARAM_DIM:] = param_vec
            # Save the updated vector back
            np.save(feat_path, features)

        return features, mscs, concorde_cpi

    except subprocess.TimeoutExpired:
        print(f"    [WARN] concorde_features timed out for {m5out_dir}")
        return None, None, None
    except Exception as e:
        print(f"    [WARN] concorde_features exception: {e}")
        traceback.print_exc()
        return None, None, None


# ============================================================
# Single Sample Runner
# ============================================================
def run_one_sample(bench, ckpt_r, ckpt_tick, slice_k, sample_id,
                   output_base, sem):
    """Run one datapoint: gem5 → concorde_features → overwrite cpi.npy.

    Directory: <output_base>/<bench>/ckpt_<tick>/slice_<k>/<config_dirname>/
    Files produced:
      3873_vector.npy  (input, last 23 dims = arch params)
      mscs.npy         (output)
      cpi.npy          (output, ground-truth from gem5)
      stats.txt + other gem5 files
    """
    tag = f"{bench}/ckpt{ckpt_tick}/s{slice_k}/cfg{sample_id}"
    fast_forward = slice_k * SLICE_STRIDE

    with sem:
        try:
            params, bp_choice = sample_random_config()
            dp_dir = os.path.join(
                output_base, bench, f"ckpt_{ckpt_tick}",
                f"slice_{slice_k}", config_to_dirname(params)
            )
            os.makedirs(dp_dir, exist_ok=True)
            trace_path = os.path.join(dp_dir, f"trace_{bench}.txt")

            argv = build_gem5_command(
                bench, params, bp_choice,
                os.path.join(CKPT_BASE, bench), ckpt_r,
                dp_dir, trace_path,
                fast_forward, WARMUP_INSTS, INST_PER_SLICE
            )

            p = subprocess.Popen(
                argv, cwd=GEM5_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            try:
                stdout, _ = p.communicate(timeout=TIMEOUT_SECS)
                gem5_ok = (p.returncode == 0)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate(timeout=10)
                gem5_ok = False

            if not gem5_ok:
                print(f"  [{tag}] FAIL")
                return False

            # concorde_features → 3873_vector.npy, mscs.npy, cpi.npy
            run_concorde_features(dp_dir, params, bp_choice)

            # Overwrite cpi.npy with ground-truth from gem5
            gt_cpi = parse_cpi_from_stats(os.path.join(dp_dir, "stats.txt"))
            if gt_cpi is not None:
                np.save(os.path.join(dp_dir, "cpi.npy"), np.float32(gt_cpi))

            print(f"  [{tag}] OK CPI={gt_cpi}")
            return True

        except Exception as e:
            print(f"  [{tag}] ERR: {e}")
            return False


# ============================================================
# Aggregate all results into single numpy arrays
# ============================================================
def aggregate_results(output_base):
    """Walk the output tree and aggregate all successful datapoints.

    Per datapoint directory (= gem5 out_dir):
      <dp_dir>/
        3873_vector.npy   ← INPUT  (last 23 dims = arch params)
        mscs.npy          ← OUTPUT (4×4 MSCS)
        cpi.npy           ← OUTPUT (ground-truth CPI from gem5)
        stats.txt         ← gem5 raw stats
    """
    print("\n=== Aggregating results ===")
    all_features = []   # input:  3873-dim vector
    all_mscs = []       # output: 16-dim (flattened 4×4)
    all_cpi = []        # output: ground-truth CPI

    for root, dirs, files in os.walk(output_base):
        # A valid datapoint has all three .npy files
        if "3873_vector.npy" not in files or "cpi.npy" not in files:
            continue

        feat_path = os.path.join(root, "3873_vector.npy")
        mscs_path = os.path.join(root, "mscs.npy")
        cpi_path  = os.path.join(root, "cpi.npy")

        cpi = float(np.load(cpi_path))
        all_cpi.append(cpi)

        all_features.append(np.load(feat_path))

        if os.path.isfile(mscs_path):
            all_mscs.append(np.load(mscs_path).flatten())

    # Save aggregated arrays
    agg_dir = os.path.join(output_base, "_aggregated")
    os.makedirs(agg_dir, exist_ok=True)

    if all_cpi:
        np.save(os.path.join(agg_dir, "all_cpi.npy"),
                np.array(all_cpi, dtype=np.float32))
        print(f"  CPI (ground-truth) samples: {len(all_cpi)}")

    if all_features:
        np.save(os.path.join(agg_dir, "all_features_3873.npy"),
                np.stack(all_features))
        print(f"  Feature (input) samples: {len(all_features)}")

    if all_mscs:
        np.save(os.path.join(agg_dir, "all_mscs.npy"),
                np.stack(all_mscs))
        print(f"  MSCS (output) samples: {len(all_mscs)}")

    if all_meta:
        with open(os.path.join(agg_dir, "all_meta.json"), "w") as f:
            json.dump(all_meta, f, indent=2)

    print(f"  Aggregated files saved to: {agg_dir}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Training data pipeline: gem5 → concorde_features → .npy files\n"
                    "53 ckpts × 800 slices × 20 configs ≈ 848K datapoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--only", nargs="*",
                    help="Subset of benchmarks (e.g., 602.gcc_s 605.mcf_s)")
    ap.add_argument("--workers", type=int, default=MAX_CORES,
                    help=f"Max parallel gem5 processes (default={MAX_CORES})")
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR,
                    help=f"Base output directory (default={DEFAULT_OUT_DIR})")
    ap.add_argument("--slices", type=int, default=SLICES_PER_CKPT,
                    help=f"Slices per checkpoint (default={SLICES_PER_CKPT})")
    ap.add_argument("--configs", type=int, default=CONFIGS_PER_SLICE,
                    help=f"Random configs per slice (default={CONFIGS_PER_SLICE})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only print job counts and one example command")
    ap.add_argument("--skip-aggregate", action="store_true")
    args = ap.parse_args()

    workers = min(args.workers, MAX_CORES)
    output_base = args.output_dir
    os.makedirs(output_base, exist_ok=True)

    benches = args.only if args.only else list(COMMANDS.keys())
    for b in benches:
        if b not in COMMANDS:
            print(f"[ERROR] Unknown benchmark: {b}")
            sys.exit(1)

    # ---- Discover checkpoints & build job list ----
    # Job = (bench, ckpt_r, ckpt_tick, slice_k, sample_id)
    jobs = []
    total_ckpts = 0
    for bench in benches:
        ckpts = discover_checkpoints(bench)
        if not ckpts:
            print(f"  [{bench}] no valid checkpoints, skipping")
            continue
        total_ckpts += len(ckpts)
        print(f"  [{bench}] {len(ckpts)} ckpts")
        for ckpt_r, ckpt_tick in ckpts:
            for k in range(args.slices):
                for sid in range(args.configs):
                    jobs.append((bench, ckpt_r, ckpt_tick, k, sid))

    total = len(jobs)
    print(f"\n=== {total_ckpts} ckpts × {args.slices} slices × {args.configs} configs"
          f" = {total} jobs | workers={workers} ===\n")

    if args.dry_run:
        if jobs:
            bench, ckpt_r, ckpt_tick, k, _ = jobs[0]
            params, bp = sample_random_config()
            dp_dir = os.path.join(output_base, bench, f"ckpt_{ckpt_tick}",
                                   f"slice_{k}", config_to_dirname(params))
            argv = build_gem5_command(
                bench, params, bp,
                os.path.join(CKPT_BASE, bench), ckpt_r,
                dp_dir, os.path.join(dp_dir, f"trace_{bench}.txt"),
                k * SLICE_STRIDE, WARMUP_INSTS, INST_PER_SLICE
            )
            print("Example command:")
            print(f"  cd {GEM5_DIR}")
            print(f"  {' '.join(argv)}")
        return

    # ---- Launch ----
    sem = Semaphore(workers)
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(run_one_sample, b, cr, ct, k, sid, output_base, sem):
            (b, ct, k, sid)
            for b, cr, ct, k, sid in jobs
        }
        for fut in as_completed(futs):
            try:
                if fut.result():
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
            done = ok + fail
            if done % 500 == 0:
                print(f"  --- progress: {done}/{total} (ok={ok}, fail={fail}) ---")

    print(f"\n{'='*50}")
    print(f"DONE  total={total}  ok={ok}  fail={fail}")
    print(f"Output: {output_base}")
    print(f"{'='*50}")

    if not args.skip_aggregate:
        aggregate_results(output_base)


if __name__ == "__main__":
    main()