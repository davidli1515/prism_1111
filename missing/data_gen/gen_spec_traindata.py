#!/usr/bin/env python3
"""
Full Training Data Generation Pipeline
=======================================
Pipeline:  gem5 (with checkpoint restore + random micro-arch config)
         → concord_analytical (trace → concord_data.npy + gem_data.npy)

Usage examples:
  # Run all benchmarks, 5 random configs per checkpoint, 100K instructions
  python3 gen_spec_traindata.py --insts-default 100000 --num-samples 5 --workers 20

  # Run only gcc, 10 samples
  python3 gen_spec_traindata.py --only 602.gcc_s --insts-default 100000 --num-samples 10 --workers 20

  # Specify custom output directory
  python3 gen_spec_traindata.py --insts-default 100000 --num-samples 5 --workers 20 --output-dir /scratch/train_data
"""

import argparse, os, json, shlex, subprocess, hashlib, glob, sys, shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Condition, Lock
import random
import time
import traceback

# ============================================================
# Memory-aware concurrency: 8GB→1 slot, 16GB→2, 32GB→4
# ============================================================
BENCH_MEM_GB = {
    "657.xz_s": 32,
    "603.bwaves_s": 16,
    "654.roms_s": 16,
}


def get_mem_slots(bench):
    return BENCH_MEM_GB.get(bench, 8) // 8


class MemoryBudget:
    def __init__(self, total_slots=50):
        self._lock = Lock()
        self._cond = Condition(self._lock)
        self._available = total_slots

    def acquire(self, slots):
        with self._cond:
            while self._available < slots:
                self._cond.wait()
            self._available -= slots

    def release(self, slots):
        with self._cond:
            self._available += slots
            self._cond.notify_all()

# ============================================================
# Global Configuration
# ============================================================
GEM5_DIR        = "/path/to/gem5"
GEM5_BIN        = os.path.join(GEM5_DIR, "build/ARM/gem5.opt")
GEM5_CONFIG     = os.path.join(GEM5_DIR, "configs/example/se_new.py")
CONCORDE_BIN    = "/path/to/gem5/concord_analytical/concord_analytical"
SPEC_BASE       = "/home/user/spec17/benchspec/CPU"
CKPT_BASE       = "/data/micro_ckpt/m5out_cpts"
DEFAULT_OUT_DIR = "/data/micro_traindata_321"
MAX_CORES       = 20
TIMEOUT_SECS    = 7200   # 2 hours per gem5 run

# Data generation constants (53 ckpts × 400 slices × 10 configs ≈ 212K per round)
SLICES_PER_CKPT   = 800       # max slices (default --slices=400)
CONFIGS_PER_SLICE = 10        # random micro-arch configs per slice per round
INST_PER_SLICE    = 100_000   # 10万 instructions per slice (-I)
SLICE_STRIDE      = 100_000   # 10万 instructions between slice starts
BASE_WARMUP       = 20_000_000  # 2000万 base warmup instructions
# Actual warmup for slice k = BASE_WARMUP + k * SLICE_STRIDE

# ============================================================
# Benchmark Command Templates  (paths updated to new location)
# ============================================================
# All paths: /home/user/spec17/benchspec/CPU/XXX  →  /home/user/spec17_20_benchmarks_minimal/XXX
# Config:    /path/to/gem5/configs/...          →  configs/... (relative, run from GEM5_DIR)
# NOTE: -I and -P flags are injected dynamically; the -I here is a placeholder.
#       --fast-forward / --standard-switch / -W are REMOVED; checkpoint restore is used instead.

_B = SPEC_BASE  # shorthand

COMMANDS = {
    "602.gcc_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/602.gcc_s/exe/sgcc_base.mytest-64"'
        f' -o "{_B}/602.gcc_s/run/run_base_refspeed_mytest-64.0000/gcc-pp.c'
        f' -O5 -fipa-pta -o gcc-pp.opts-O5_-fipa-pta.s"'
        f' -I 100000'
    ),
    "600.perlbench_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/600.perlbench_s/exe/perlbench_s_base.mytest-64"'
        f' --options="-I{_B}/600.perlbench_s/run/run_base_refspeed_mytest-64.0000/lib'
        f' {_B}/600.perlbench_s/run/run_base_refspeed_mytest-64.0000/checkspam.pl'
        f' 2500 5 25 11 150 1 1 1 1"'
        f' -I 100000'
    ),
    "619.lbm_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/619.lbm_s/exe/lbm_s_base.mytest-64"'
        f' --options="2000 reference.out 0 0 {_B}/619.lbm_s/data/refspeed/input/200_200_260_ldc.of"'
        f' -I 100000'
    ),
    "605.mcf_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/605.mcf_s/exe/mcf_s_base.mytest-64"'
        f' -o "{_B}/605.mcf_s/run/run_base_refspeed_mytest-64.0000/inp.in"'
        f' -I 100000'
    ),
    "620.omnetpp_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/620.omnetpp_s/run/run_base_refspeed_mytest-64.0000/omnetpp_s_base.mytest-64"'
        f' --options="-c General -r 0"'
        f' -I 100000'
    ),
    "623.xalancbmk_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/623.xalancbmk_s/exe/xalancbmk_s_base.mytest-64"'
        f' -o "-v {_B}/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/t5.xml'
        f' {_B}/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/xalanc.xsl"'
        f' -I 100000'
    ),
    "631.deepsjeng_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/631.deepsjeng_s/run/run_base_refspeed_mytest-64.0000/deepsjeng_s_base.mytest-64"'
        f' -o "{_B}/631.deepsjeng_s/run/run_base_refspeed_mytest-64.0000/ref.txt"'
        f' -I 100000'
    ),
    "641.leela_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/641.leela_s/exe/leela_s_base.mytest-64"'
        f' -o "{_B}/641.leela_s/run/run_base_refspeed_mytest-64.0000/ref.sgf"'
        f' -I 100000'
    ),
    "644.nab_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/644.nab_s/run/run_base_refspeed_mytest-64.0000/nab_s_base.mytest-64"'
        f' -o "3j1n 20140317 220"'
        f' -I 100000'
    ),
    "638.imagick_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
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
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=32GB'
        f' -c "{_B}/657.xz_s/run/run_base_refspeed_mytest-64.0000/xz_s_base.mytest-64"'
        f' -o "{_B}/657.xz_s/run/run_base_refspeed_mytest-64.0000/cpu2006docs.tar.xz'
        f' 6643 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae'
        f' 1036078272 1111795472 4"'
        f' -I 100000'
    ),
    "648.exchange2_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/648.exchange2_s/exe/exchange2_s_base.mytest-64"'
        f' -o "6"'
        f' -I 100000'
    ),
    "607.cactuBSSN_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/607.cactuBSSN_s/run/run_base_test_mytest-64.0000/cactuBSSN_s_base.mytest-64"'
        f' -o "{_B}/607.cactuBSSN_s/run/run_base_refspeed_mytest-64.0000/spec_ref.par"'
        f' -I 100000'
    ),
    "621.wrf_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/621.wrf_s/run/run_base_refspeed_mytest-64.0000/wrf_s_base.mytest-64"'
        f' -I 100000'
    ),
    "508.namd_r": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/508.namd_r/run/run_base_refrate_mytest-64.0000/namd_r_base.mytest-64"'
        f' -o "--input {_B}/508.namd_r/run/run_base_refrate_mytest-64.0000/apoa1.input'
        f' --output apoa1.ref.output --iterations 65"'
        f' -I 100000'
    ),
    "526.blender_r": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/526.blender_r/run/run_base_refrate_mytest-64.0000/blender_r_base.mytest-64"'
        f' -o "{_B}/526.blender_r/data/refrate/input/sh3_no_char.blend'
        f' --render-output sh3_no_char_ --threads 1 -b -F RAWTGA -s 849 -e 849 -a"'
        f' -I 100000'
    ),
    "511.povray_r": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
        f' -c "{_B}/511.povray_r/run/run_base_refrate_mytest-64.0000/povray_r_base.mytest-64"'
        f' -o "SPEC-benchmark-ref.ini +L{_B}/511.povray_r/run/run_base_refrate_mytest-64.0000"'
        f' -I 100000'
    ),
    "603.bwaves_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=16GB'
        f' -c "{_B}/603.bwaves_s/run/run_base_refspeed_mytest-64.0000/speed_bwaves_base.mytest-64"'
        f' -o "bwaves_1"'
        f' --input "{_B}/603.bwaves_s/run/run_base_refspeed_mytest-64.0000/bwaves_1.in"'
        f' -I 100000'
    ),
    "654.roms_s": (
        f'{GEM5_BIN} {GEM5_CONFIG}'
        f' --cpu-type=DerivO3CPU --caches --mem-size=16GB'
        f' -c "{_B}/654.roms_s/run/run_base_refspeed_mytest-64.0000/sroms_base.mytest-64"'
        f' --input "{_B}/654.roms_s/run/run_base_refspeed_mytest-64.0000/ocean_benchmark3.in.x"'
        f' -I 100000'
    ),
}

# Benchmarks that need to run from their 'run' directory (relative path inputs)
BENCH_CWD = {
    "600.perlbench_s": f"{_B}/600.perlbench_s/run/run_base_refspeed_mytest-64.0000",
    "511.povray_r":    f"{_B}/511.povray_r/run/run_base_refrate_mytest-64.0000",
    "620.omnetpp_s":   f"{_B}/620.omnetpp_s/run/run_base_refspeed_mytest-64.0000",
    "621.wrf_s":       f"{_B}/621.wrf_s/run/run_base_refspeed_mytest-64.0000",
    "644.nab_s":       f"{_B}/644.nab_s/run/run_base_refspeed_mytest-64.0000",
    "648.exchange2_s": f"{_B}/648.exchange2_s/run/run_base_refspeed_mytest-64.0000",
    "654.roms_s":      f"{_B}/654.roms_s/run/run_base_refspeed_mytest-64.0000",
}


def get_gem5_cwd(bench):
    """Return the working directory for running gem5 on this benchmark."""
    return BENCH_CWD.get(bench, GEM5_DIR)


# ============================================================
# Checkpoint Discovery
# ============================================================
def discover_checkpoints(bench_name):
    """Find valid SimPoint checkpoints for a benchmark.
    Each tick subdirectory (e.g., 420400000000/) is a standalone gem5 output dir
    containing cpt.None.TICK. We pass the tick dir as --checkpoint-dir with -r 1.
    Returns list of ckpt_tick_name strings for valid dirs only.
    """
    bench_dir = os.path.join(CKPT_BASE, bench_name)
    if not os.path.isdir(bench_dir):
        return []
    subdirs = sorted([
        d for d in os.listdir(bench_dir)
        if os.path.isdir(os.path.join(bench_dir, d)) and d.isdigit()
    ], key=lambda x: int(x))
    return [d for d in subdirs
            if glob.glob(os.path.join(bench_dir, d, "cpt.*"))]


# ============================================================
# Random Micro-architecture Sampling (from your paper's Table 3)
# ============================================================
def sample_random_config(seed=None):
    """Sample a random micro-architecture configuration with structural constraints.

    Constraints:
      1. Front-End Monotonicity: fetch >= decode >= rename >= dispatch (all >= 2)
      2. Back-End Provisioning:  issue, writeback, commit >= dispatch (all >= 2)
      3. Resource Lower Bounds:  ROB >= 64, IQ >= 32, LQ >= 16, SQ >= 16
      4. Capacity Inclusion:     ROB >= 2*issue + IQ + LQ + SQ
    """
    rng = random.Random(seed)
    l1_options = ["16", "32", "64", "128", "256"]
    l2_options = ["512", "1024", "2048", "4096"]
    assoc_options = ["2", "4", "8", "16", "32"]
    bp_choice = rng.choice(["BiModeBP", "TAGE"])

    # Front-end monotonicity: fetch >= decode >= rename >= dispatch, all in [2,12]
    fw = sorted([rng.randint(4, 12) for _ in range(4)], reverse=True)
    fetchW, decodeW, renameW, dispatchW = fw

    # Back-end: issue == dispatch (gem5 constraint), wb/commit >= dispatch
    issueW  = dispatchW
    wbW     = rng.randint(dispatchW, 12)
    commitW = rng.randint(dispatchW, 12)

    # Queues first (needed for ROB lower bound)
    iqN = rng.randint(32, 128)
    lqN = rng.randint(16, 256)
    sqN = rng.randint(16, 256)

    # Capacity inclusion: ROB >= DIQ + IQ + LQ + SQ, where DIQ = 2 * issueWidth
    rob_min = max(64, 2 * issueW + iqN + lqN + sqN)
    robN = rng.randint(rob_min, max(rob_min, 1024))

    params = {
        "fetchWidth":    str(fetchW),
        "decodeWidth":   str(decodeW),
        "renameWidth":   str(renameW),
        "dispatchWidth": str(dispatchW),
        "issueWidth":    str(issueW),
        "wbWidth":       str(wbW),
        "commitWidth":   str(commitW),
        "numROBEntries":   str(robN),
        "numIQEntries":    str(iqN),
        "LQEntries":       str(lqN),
        "SQEntries":       str(sqN),
        "fetchBufferSize": str(rng.choice([8, 16, 32, 64])),
        "IntALU":     str(rng.randint(1, 8)),
        "IntMultDiv": str(rng.randint(1, 8)),
        "FP_ALU":     str(rng.randint(1, 8)),
        "FP_MultDiv": str(rng.randint(1, 8)),
        "RdWrPort":   str(rng.randint(1, 8)),
        "l1_size":     rng.choice(l1_options),
        "l2_size":     rng.choice(l2_options),
        "l1d_assoc":   rng.choice(assoc_options),
        "l1i_assoc":   rng.choice(assoc_options),
        "l2_assoc":    rng.choice(assoc_options),
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


def make_seed(bench, ckpt_tick, slice_k, global_sid):
    """Deterministic seed per job. Different global_sid → different config.
    global_sid = round * configs_per_slice + local_sid."""
    raw = f"{bench}_{ckpt_tick}_{slice_k}_{global_sid}"
    return int(hashlib.md5(raw.encode()).hexdigest(), 16) % (2**31)


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
                       ckpt_dir, out_dir, warmup, inst_count):
    """Construct the full gem5 argv list.

    ckpt_dir: the specific tick subdirectory (e.g., .../508.namd_r/420400000000/)
              which contains cpt.None.TICK. Always restored with -r 1.
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

    # Checkpoint restore + standard-switch (Atomic→TimingSimple→O3) + warmup + inst count
    argv += [
        f"--checkpoint-dir={ckpt_dir}",
        "-r", "1",
        "--standard-switch=1",
        "-W", str(warmup),
        "-I", str(inst_count),
    ]

    # O3 trace dump path: o3trace.txt in the datapoint directory
    o3trace_path = os.path.join(out_dir, "o3trace")
    cpu = "system.switch_cpus_1[0]"
    argv += [
        "-P", f"{cpu}.dump_inst_trace_path='{o3trace_path}'",
    ]

    # Random micro-arch parameters
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
        "-P", f"{cpu}.fuPool.FUList[0].count={params['IntALU']}",
        "-P", f"{cpu}.fuPool.FUList[1].count={params['IntMultDiv']}",
        "-P", f"{cpu}.fuPool.FUList[2].count={params['FP_ALU']}",
        "-P", f"{cpu}.fuPool.FUList[3].count={params['FP_MultDiv']}",
        "-P", f"{cpu}.fuPool.FUList[8].count={params['RdWrPort']}",
        f"--bp-type={bp_choice}",
        "--l2cache",
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
# Run concord_analytical
# ============================================================


def run_concorde_features(dp_dir, ckpt_dir, warmup):
    """Run concord_analytical on a completed gem5 datapoint.

    Interface:
      ./concord_analytical <dp_dir> --ckpt-dir <ckpt_dir> --concord_warmup <insts>

    Reads from dp_dir:  o3trace.txt, arch_params.npy
    Reads from ckpt_dir: simpletrace.txt
    Writes to dp_dir:   concord_data.npy (3874,), gem_data.npy (4697,)
    """
    cmd = [
        CONCORDE_BIN,
        dp_dir,
        "--ckpt-dir", ckpt_dir,
        "--concord_warmup", str(warmup),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=GEM5_DIR,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            print(f"    [WARN] concord_analytical failed (rc={result.returncode}): "
                  f"{result.stderr[:500]}")
            return False

        # ---- Load and verify output files ----
        concord_path = os.path.join(dp_dir, "concord_data.npy")
        gem_path     = os.path.join(dp_dir, "gem_data.npy")

        ok = True
        if os.path.isfile(concord_path):
            concord = np.load(concord_path).astype(np.float32)
                np.save(concord_path, concord)
        else:
            print(f"    [WARN] concord_data.npy not found in {dp_dir}")
            ok = False

        if not os.path.isfile(gem_path):
            print(f"    [WARN] gem_data.npy not found in {dp_dir}")
            ok = False

        return ok

    except subprocess.TimeoutExpired:
        print(f"    [WARN] concord_analytical timed out for {dp_dir}")
        return False
    except Exception as e:
        print(f"    [WARN] concord_analytical exception: {e}")
        traceback.print_exc()
        return False


# ============================================================
# Single Sample Runner
# ============================================================
MAX_RETRIES = 50  # max attempts per (bench, ckpt, slice) before giving up


def run_one_sample(bench, ckpt_tick, slice_k, global_sid,
                   output_base, mem_budget):
    """Run one datapoint: gem5 → concord_analytical → concord_data.npy + gem_data.npy.
    Retries with a different config if gem5 or concord_analytical fails."""
    warmup = BASE_WARMUP + slice_k * SLICE_STRIDE
    ckpt_dir = os.path.join(CKPT_BASE, bench, ckpt_tick)
    slots = get_mem_slots(bench)

    for retry in range(MAX_RETRIES):
        seed = make_seed(bench, ckpt_tick, slice_k, global_sid * 1000 + retry)
        params, bp_choice = sample_random_config(seed)
        tag = f"{bench}/ckpt{ckpt_tick}/s{slice_k}/g{global_sid}"

        dp_dir = os.path.join(
            output_base, bench, f"ckpt_{ckpt_tick}",
            f"slice_{slice_k}", config_to_dirname(params)
        )

        mem_budget.acquire(slots)
        try:
            os.makedirs(dp_dir, exist_ok=True)

            argv = build_gem5_command(
                bench, params, bp_choice,
                ckpt_dir, dp_dir, warmup, INST_PER_SLICE
            )

            p = subprocess.Popen(
                argv, cwd=get_gem5_cwd(bench),
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
                print(f"  [{tag}] gem5 FAIL (retry {retry})")
                shutil.rmtree(dp_dir, ignore_errors=True)
                continue

            np.save(os.path.join(dp_dir, "arch_params.npy"), params_to_vector(params))
            concord_ok = run_concorde_features(dp_dir, ckpt_dir, warmup)
            if not concord_ok:
                print(f"  [{tag}] concord FAIL (retry {retry})")
                shutil.rmtree(dp_dir, ignore_errors=True)
                continue

            # Sanity check only (no ordering constraint): concord_analytical writes an
            # all-zero MSCS when the trace has no stage stats, which would make
            # CPI = 1 / (Base_commit * W_min) blow up downstream.
            mscs = np.load(os.path.join(dp_dir, "gem_data.npy"))[-16:]
            if not np.all(np.isfinite(mscs)) or mscs[12] <= 0:
                print(f"  [{tag}] MSCS missing (retry {retry})")
                shutil.rmtree(dp_dir, ignore_errors=True)
                continue

            # Keep the .npy outputs and the MSCS diagnostics (o3trace.mscs.txt).
            # missing_stamps and base_outside_window must be 0; *_fallback_slots
            # counts slots outside the Table 1 conditions (charged to BP).
            for fname in os.listdir(dp_dir):
                if not (fname.endswith('.npy') or fname.endswith('.mscs.txt')):
                    fpath = os.path.join(dp_dir, fname)
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                    elif os.path.isdir(fpath):
                        shutil.rmtree(fpath, ignore_errors=True)
            print(f"  [{tag}] OK (retry {retry})")
            return True

        except Exception as e:
            print(f"  [{tag}] ERR: {e} (retry {retry})")
            shutil.rmtree(dp_dir, ignore_errors=True)
            continue
        finally:
            mem_budget.release(slots)

    print(f"  [{tag}] GAVE UP after {MAX_RETRIES} retries")
    return False


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Training data pipeline: gem5 → concorde_features → .npy\n\n"
                    "Two-phase generation:\n"
                    "  Round 0: python3 gen_spec_traindata.py --round 0  (first 212K)\n"
                    "  Round 1: python3 gen_spec_traindata.py --round 1  (next 212K)\n"
                    "Different rounds produce different configs (deterministic seeds).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--round", type=int, required=True,
                    help="Round number (0,1,...). Each round generates slices×configs datapoints "
                         "per ckpt with non-overlapping random configs.")
    ap.add_argument("--only", nargs="*",
                    help="Subset of benchmarks (e.g., 602.gcc_s 605.mcf_s)")
    ap.add_argument("--workers", type=int, default=MAX_CORES)
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--slices", type=int, default=400,
                    help="Slices per checkpoint (default=400)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    workers = args.workers
    output_base = args.output_dir
    rnd = args.round
    os.makedirs(output_base, exist_ok=True)

    benches = args.only if args.only else list(COMMANDS.keys())
    for b in benches:
        if b not in COMMANDS:
            print(f"[ERROR] Unknown benchmark: {b}")
            sys.exit(1)

    # ---- Server A remaining work (80K dp) ----
    # (bench, ckpt_tick, sid_start, sid_end)
    # sid_start..sid_end-1, each × 400 slices
    MAX_CONFIGS = 60
    REMAINING_JOBS = [
        # 1-ckpt benches: phase1 ckpt sid 0-19 done, need sid 20-59
        ("657.xz_s",         "108200000000",   20, 60),   # 16K (32GB)
        ("654.roms_s",       "306300000000",    20, 60),   # 16K (16GB)
        ("644.nab_s",        "217200000000",    20, 60),   # 16K
        ("648.exchange2_s",  "444700000000",    20, 60),   # 16K
        # 3-ckpt bench: phase2 ckpts, sid 0-19
        ("526.blender_r",    "580900000000",     0, 20),   # 8K
        ("526.blender_r",    "43600000000",      0, 20),   # 8K
    ]

    jobs = []
    for bench, ckpt_tick, sid_start, sid_end in REMAINING_JOBS:
        n_dp = (sid_end - sid_start) * args.slices
        print(f"  {bench}/ckpt_{ckpt_tick}: sid {sid_start}-{sid_end-1} × {args.slices} slices = {n_dp} dp")
        for k in range(args.slices):
            for sid in range(sid_start, sid_end):
                global_sid = rnd * MAX_CONFIGS + sid
                jobs.append((bench, ckpt_tick, k, global_sid))

    total = len(jobs)
    print(f"\n=== Round {rnd}: {total} jobs | workers={workers} ===\n")

    if args.dry_run:
        if jobs:
            bench, ckpt_tick, k, gsid = jobs[0]
            seed = make_seed(bench, ckpt_tick, k, gsid)
            params, bp = sample_random_config(seed)
            warmup = BASE_WARMUP + k * SLICE_STRIDE
            dp_dir = os.path.join(output_base, bench, f"ckpt_{ckpt_tick}",
                                   f"slice_{k}", config_to_dirname(params))
            argv = build_gem5_command(
                bench, params, bp,
                os.path.join(CKPT_BASE, bench, ckpt_tick),
                dp_dir, warmup, INST_PER_SLICE
            )
            print(f"Example (round={rnd}, global_sid={gsid}, seed={seed}, warmup={warmup}):")
            print(f"  cd {GEM5_DIR}")
            print(f"  {' '.join(argv)}")
        return

    # ---- Build per-ckpt job counts for tracking ----
    from collections import defaultdict
    ckpt_total = defaultdict(int)    # (bench, ckpt_tick) → total jobs
    ckpt_ok    = defaultdict(int)    # (bench, ckpt_tick) → completed ok
    ckpt_fail  = defaultdict(int)
    # Ordered list of unique ckpts for progress display
    ckpt_order = []
    seen = set()
    for b, ct, k, gsid in jobs:
        key = (b, ct)
        ckpt_total[key] += 1
        if key not in seen:
            ckpt_order.append(key)
            seen.add(key)

    # ---- Launch ----
    mem_budget = MemoryBudget(50)  # 50 slots × 8GB = 400GB
    ok = fail = 0
    t_start = time.time()
    last_report = t_start

    def print_report():
        elapsed = time.time() - t_start
        rate = (ok + fail) / elapsed if elapsed > 0 else 0
        eta = (total - ok - fail) / rate / 3600 if rate > 0 else 0
        print(f"\n--- Status @ {time.strftime('%H:%M:%S')} "
              f"({elapsed/3600:.1f}h elapsed, ETA {eta:.1f}h) ---", flush=True)
        print(f"  Total: {ok+fail}/{total}  ok={ok} fail={fail}  {rate:.1f} jobs/s", flush=True)
        # Per-ckpt status
        done_ckpts = []
        active_ckpts = []
        pending_ckpts = []
        for key in ckpt_order:
            b, ct = key
            done_c = ckpt_ok[key] + ckpt_fail[key]
            tot_c = ckpt_total[key]
            if done_c == tot_c:
                done_ckpts.append(key)
            elif done_c > 0:
                active_ckpts.append((key, done_c, tot_c))
            else:
                pending_ckpts.append(key)
        print(f"  DONE ({len(done_ckpts)}/{len(ckpt_order)}):", flush=True)
        for key in done_ckpts:
            print(f"    {key[0]}/ckpt_{key[1]}  ok={ckpt_ok[key]} fail={ckpt_fail[key]}", flush=True)
        for key, done_c, tot_c in active_ckpts:
            print(f"    ACTIVE {key[0]}/ckpt_{key[1]}: {done_c}/{tot_c}"
                  f" ({done_c*100//tot_c}%)", flush=True)
        print(f"  PENDING ({len(pending_ckpts)}):", flush=True)
        for key in pending_ckpts:
            print(f"    {key[0]}/ckpt_{key[1]}", flush=True)
        print("---", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {}
        for b, ct, k, gsid in jobs:
            fut = pool.submit(run_one_sample, b, ct, k, gsid, output_base, mem_budget)
            futs[fut] = (b, ct)

        for fut in as_completed(futs):
            b, ct = futs[fut]
            key = (b, ct)
            try:
                if fut.result():
                    ok += 1
                    ckpt_ok[key] += 1
                else:
                    fail += 1
                    ckpt_fail[key] += 1
            except Exception:
                fail += 1
                ckpt_fail[key] += 1

            done = ok + fail

            # Progress bar every 100 jobs
            if done % 100 == 0 or done == total:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate / 3600 if rate > 0 else 0
                pct = done * 100 // total
                bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                msg = (f"  [{bar}] {pct:3d}% ({done}/{total})"
                       f"  ok={ok} fail={fail}"
                       f"  {rate:.1f} jobs/s  ETA={eta:.1f}h")
                if sys.stdout.isatty():
                    print(f"\r{msg}", end="", flush=True)
                else:
                    print(msg, flush=True)

            # Detailed report every 30 minutes
            now = time.time()
            if now - last_report >= 1800:
                last_report = now
                print_report()

    # Final report
    print_report()
    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Round {rnd} DONE  total={total}  ok={ok}  fail={fail}")
    print(f"Wall time: {elapsed/3600:.1f}h")
    print(f"Output: {output_base}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()