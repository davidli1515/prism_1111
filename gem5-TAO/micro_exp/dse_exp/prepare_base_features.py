"""
Prepare o3trace files for DSE (one-time per benchmark)

Runs 400 gem5 simulations (each 100K instructions, different warmup offsets),
saves o3trace.txt for each slice. These traces are reused by every DSE
evaluation — only concord_analytical needs to be re-run per config.

simpletrace.txt must already exist in the checkpoint directory.

Usage:
    python prepare_base_features.py --benchmark 605.mcf_s --workers 24
    python prepare_base_features.py --benchmark 623.xalancbmk_s --workers 24
    python prepare_base_features.py --benchmark 605.mcf_s --dry-run

Output:
    {TRACE_DIR}/{benchmark}/slice_0/o3trace.txt
    {TRACE_DIR}/{benchmark}/slice_1/o3trace.txt
    ...
    {TRACE_DIR}/{benchmark}/slice_399/o3trace.txt
"""

import argparse
import os
import sys
import shlex
import shutil
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
import time
import traceback

# ============================================================
# Paths
# ============================================================

GEM5_DIR     = "/scratch/gem5-TAO"
GEM5_BIN     = os.path.join(GEM5_DIR, "build/ARM/gem5.opt")
GEM5_CONFIG  = os.path.join(GEM5_DIR, "configs/example/se_new.py")
SPEC_BASE    = "/home/wl730/spec17/benchspec/CPU"
CKPT_BASE    = "/scratch2/wl730/micro_ckpt/m5out_cpts"

TRACE_DIR    = "/scratch/gem5-TAO/micro_exp/dse_exp/traces"

# Slice parameters (same as gen_spec_traindata.py)
INST_PER_SLICE = 100_000      # 10万 per slice
SLICE_STRIDE   = 100_000      # 10万 between slice starts
BASE_WARMUP    = 20_000_000   # 2000万 base warmup
N_SLICES       = 400          # 400 slices
GEM5_TIMEOUT   = 7200

# Default config (trace content depends on config, but we use default
# for the base traces — DSE re-runs concord_analytical per config)
DEFAULT_BP = "BiModeBP"
DEFAULT_PARAMS = {
    "fetchWidth": "8", "decodeWidth": "8", "renameWidth": "8",
    "dispatchWidth": "8", "issueWidth": "8", "wbWidth": "8", "commitWidth": "8",
    "numROBEntries": "192", "numIQEntries": "64",
    "LQEntries": "32", "SQEntries": "32",
    "fetchBufferSize": "32",
    "IntALU": "2", "IntMultDiv": "1",
    "FP_ALU": "2", "FP_MultDiv": "1", "RdWrPort": "2",
    "l1_size": "64", "l2_size": "1024",
    "l1d_assoc": "2", "l1i_assoc": "2", "l2_assoc": "8",
}

# ============================================================
# Benchmark Configs
# ============================================================

_B = SPEC_BASE

BENCHMARKS = {
    "605.mcf_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/605.mcf_s/exe/mcf_s_base.mytest-64"'
            f' -o "{_B}/605.mcf_s/run/run_base_refspeed_mytest-64.0000/inp.in"'
        ),
        "ckpt_tick": "917300000000",
        "cwd": GEM5_DIR,
    },
    "623.xalancbmk_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/623.xalancbmk_s/exe/xalancbmk_s_base.mytest-64"'
            f' -o "-v {_B}/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/t5.xml'
            f' {_B}/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/xalanc.xsl"'
        ),
        "ckpt_tick": "537100000000",
        "cwd": GEM5_DIR,
    },
    "641.leela_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/641.leela_s/exe/leela_s_base.mytest-64"'
            f' -o "{_B}/641.leela_s/run/run_base_refspeed_mytest-64.0000/ref.sgf"'
        ),
        "ckpt_tick": "1510100000000",
        "cwd": GEM5_DIR,
    },
    "600.perlbench_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c {_B}/600.perlbench_s/exe/perlbench_s_base.mytest-64'
            f' "--options=-I{_B}/600.perlbench_s/run/run_base_refspeed_mytest-64.0000/lib'
            f' {_B}/600.perlbench_s/run/run_base_refspeed_mytest-64.0000/checkspam.pl'
            f' 2500 5 25 11 150 1 1 1 1"'
        ),
        "ckpt_tick": "240100000000",
        "cwd": GEM5_DIR,
    },
    "508.namd_r": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/508.namd_r/run/run_base_refrate_mytest-64.0000/namd_r_base.mytest-64"'
            f' -o "--input {_B}/508.namd_r/run/run_base_refrate_mytest-64.0000/apoa1.input'
            f' --output apoa1.ref.output --iterations 65"'
        ),
        "ckpt_tick": "1200100000000",
        "cwd": GEM5_DIR,
    },
    "526.blender_r": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/526.blender_r/run/run_base_refrate_mytest-64.0000/blender_r_base.mytest-64"'
            f' -o "{_B}/526.blender_r/data/refrate/input/sh3_no_char.blend'
            f' --render-output sh3_no_char_ --threads 1 -b -F RAWTGA -s 849 -e 849 -a"'
        ),
        "ckpt_tick": "43600000000",
        "cwd": GEM5_DIR,
    },
    "602.gcc_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/602.gcc_s/exe/sgcc_base.mytest-64"'
            f' -o "{_B}/602.gcc_s/run/run_base_refspeed_mytest-64.0000/gcc-pp.c'
            f' -O5 -fipa-pta -o gcc-pp.opts-O5_-fipa-pta.s"'
        ),
        "ckpt_tick": "430300000000",
        "cwd": GEM5_DIR,
    },
    "607.cactuBSSN_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/607.cactuBSSN_s/run/run_base_test_mytest-64.0000/cactuBSSN_s_base.mytest-64"'
            f' -o "{_B}/607.cactuBSSN_s/run/run_base_refspeed_mytest-64.0000/spec_ref.par"'
        ),
        "ckpt_tick": "295000000000",
        "cwd": GEM5_DIR,
    },
    "619.lbm_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/619.lbm_s/exe/lbm_s_base.mytest-64"'
            f' -o "2000 reference.out 0 0'
            f' {_B}/619.lbm_s/data/refspeed/input/200_200_260_ldc.of"'
        ),
        "ckpt_tick": "1900300000000",
        "cwd": GEM5_DIR,
    },
    "620.omnetpp_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/620.omnetpp_s/run/run_base_refspeed_mytest-64.0000/omnetpp_s_base.mytest-64"'
            f' "--options=-c General -r 0"'
        ),
        "ckpt_tick": "85300000000",
        "cwd": GEM5_DIR,
    },
    "621.wrf_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/621.wrf_s/run/run_base_refspeed_mytest-64.0000/wrf_s_base.mytest-64"'
        ),
        "ckpt_tick": "162800000000",
        "cwd": GEM5_DIR,
    },
    "631.deepsjeng_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/631.deepsjeng_s/run/run_base_refspeed_mytest-64.0000/deepsjeng_s_base.mytest-64"'
            f' -o "{_B}/631.deepsjeng_s/run/run_base_refspeed_mytest-64.0000/ref.txt"'
        ),
        "ckpt_tick": "1154700000000",
        "cwd": GEM5_DIR,
    },
    "638.imagick_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/638.imagick_s/exe/imagick_s_base.mytest-64"'
            f' -o "-limit disk 0'
            f' {_B}/638.imagick_s/run/run_base_refspeed_mytest-64.0000/refspeed_input.tga'
            f' -resize 817% -rotate -2.76 -shave 540x375 -alpha remove -auto-level'
            f' -contrast-stretch 1x1% -colorspace Lab -channel R -equalize +channel'
            f' -colorspace sRGB -define histogram:unique-colors=false -adaptiveblur 0x5'
            f' -despeckle -auto-gamma -adaptive-sharpen 55 -enhance -brightness-contrast 10x10'
            f' -resize 30%'
            f' {_B}/638.imagick_s/run/run_base_refspeed_mytest-64.0000/refspeed_output.tga"'
        ),
        "ckpt_tick": "41800000000",
        "cwd": GEM5_DIR,
    },
    "644.nab_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/644.nab_s/run/run_base_refspeed_mytest-64.0000/nab_s_base.mytest-64"'
            f' -o "3j1n 20140317 220"'
        ),
        "ckpt_tick": "217200000000",
        "cwd": GEM5_DIR,
    },
    "648.exchange2_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=8GB'
            f' -c "{_B}/648.exchange2_s/exe/exchange2_s_base.mytest-64"'
            f' -o "6"'
        ),
        "ckpt_tick": "444700000000",
        "cwd": GEM5_DIR,
    },
    "654.roms_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=16GB'
            f' -c "{_B}/654.roms_s/run/run_base_refspeed_mytest-64.0000/sroms_base.mytest-64"'
        ),
        "ckpt_tick": "306300000000",
        "cwd": GEM5_DIR,
    },
    "657.xz_s": {
        "command": (
            f'{GEM5_BIN} {GEM5_CONFIG}'
            f' --cpu-type=DerivO3CPU --caches --mem-size=32GB'
            f' -c "{_B}/657.xz_s/run/run_base_refspeed_mytest-64.0000/xz_s_base.mytest-64"'
            f' -o "{_B}/657.xz_s/run/run_base_refspeed_mytest-64.0000/cpu2006docs.tar.xz'
            f' 6643 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae'
            f' 1036078272 1111795472 4"'
        ),
        "ckpt_tick": "108200000000",
        "cwd": GEM5_DIR,
    },
}


# ============================================================
# Run one slice: gem5 only, keep o3trace.txt
# ============================================================

def run_one_slice(bench, ckpt_dir, slice_k, trace_dir, sem):
    """
    Run gem5 for one slice (100K instructions), keep o3trace.txt.
    Returns True on success.
    """
    warmup = BASE_WARMUP + slice_k * SLICE_STRIDE
    slice_dir = os.path.join(trace_dir, f"slice_{slice_k}")
    tag = f"slice_{slice_k}"
    bench_info = BENCHMARKS[bench]

    # Skip if o3trace.txt already exists
    o3trace_file = os.path.join(slice_dir, "o3trace.txt")
    if os.path.isfile(o3trace_file) and os.path.getsize(o3trace_file) > 0:
        return True

    with sem:
        try:
            os.makedirs(slice_dir, exist_ok=True)

            # Build gem5 command
            argv = shlex.split(bench_info["command"], posix=True)

            # Remove template -I if present
            i = 0
            while i < len(argv):
                if argv[i] == "-I" and i + 1 < len(argv):
                    del argv[i:i + 2]
                else:
                    i += 1

            argv.insert(1, "-d")
            argv.insert(2, slice_dir)

            o3trace_path = os.path.join(slice_dir, "o3trace")
            cpu = "system.switch_cpus_1[0]"
            p = DEFAULT_PARAMS

            argv += [
                f"--checkpoint-dir={ckpt_dir}",
                "-r", "1",
                "--standard-switch=1",
                "-W", str(warmup),
                "-I", str(INST_PER_SLICE),
                "-P", f"{cpu}.dump_inst_trace_path='{o3trace_path}'",
                "-P", f"{cpu}.fetchWidth={p['fetchWidth']}",
                "-P", f"{cpu}.decodeWidth={p['decodeWidth']}",
                "-P", f"{cpu}.renameWidth={p['renameWidth']}",
                "-P", f"{cpu}.dispatchWidth={p['dispatchWidth']}",
                "-P", f"{cpu}.issueWidth={p['issueWidth']}",
                "-P", f"{cpu}.wbWidth={p['wbWidth']}",
                "-P", f"{cpu}.commitWidth={p['commitWidth']}",
                "-P", f"{cpu}.LQEntries={p['LQEntries']}",
                "-P", f"{cpu}.SQEntries={p['SQEntries']}",
                "-P", f"{cpu}.numROBEntries={p['numROBEntries']}",
                "-P", f"{cpu}.numIQEntries={p['numIQEntries']}",
                "-P", f"{cpu}.fetchBufferSize={p['fetchBufferSize']}",
                "-P", f"{cpu}.fuPool.FUList[0].count={p['IntALU']}",
                "-P", f"{cpu}.fuPool.FUList[1].count={p['IntMultDiv']}",
                "-P", f"{cpu}.fuPool.FUList[2].count={p['FP_ALU']}",
                "-P", f"{cpu}.fuPool.FUList[3].count={p['FP_MultDiv']}",
                "-P", f"{cpu}.fuPool.FUList[8].count={p['RdWrPort']}",
                f"--bp-type={DEFAULT_BP}",
                "--l1d_size", f"{p['l1_size']}kB",
                "--l1i_size", f"{p['l1_size']}kB",
                "--l2_size",  f"{p['l2_size']}kB",
                "--l3_size",  "4096kB",
                "--l1d_assoc", p['l1d_assoc'],
                "--l1i_assoc", p['l1i_assoc'],
                "--l2_assoc",  p['l2_assoc'],
            ]

            # Run gem5
            result = subprocess.run(argv, cwd=bench_info["cwd"],
                                    capture_output=True, text=True,
                                    timeout=GEM5_TIMEOUT)
            if result.returncode != 0:
                print(f"  [{tag}] gem5 FAIL (rc={result.returncode})")
                return False

            # Verify o3trace.txt exists
            if not os.path.isfile(o3trace_file):
                print(f"  [{tag}] o3trace.txt not generated")
                return False

            # Cleanup: remove everything except o3trace.txt
            for fname in os.listdir(slice_dir):
                if fname == "o3trace.txt":
                    continue
                fpath = os.path.join(slice_dir, fname)
                if os.path.isfile(fpath):
                    os.remove(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath, ignore_errors=True)

            return True

        except subprocess.TimeoutExpired:
            print(f"  [{tag}] TIMEOUT")
            return False
        except Exception as e:
            print(f"  [{tag}] ERROR: {e}")
            traceback.print_exc()
            return False


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate o3trace files for DSE (one-time per benchmark)")
    parser.add_argument("--benchmark", required=True,
                        choices=list(BENCHMARKS.keys()))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--n-slices", type=int, default=N_SLICES)
    parser.add_argument("--output-dir", default=TRACE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bench = args.benchmark
    n_slices = args.n_slices
    bench_info = BENCHMARKS[bench]
    ckpt_tick = bench_info["ckpt_tick"]
    ckpt_dir = os.path.join(CKPT_BASE, bench, ckpt_tick)

    # Verify simpletrace.txt exists
    simpletrace_path = os.path.join(ckpt_dir, "simpletrace.txt")
    if not os.path.isfile(simpletrace_path):
        print(f"ERROR: simpletrace.txt not found at {simpletrace_path}")
        sys.exit(1)

    trace_dir = os.path.join(args.output_dir, bench)
    st_size = os.path.getsize(simpletrace_path) / 1e6

    print(f"Benchmark:    {bench}")
    print(f"Checkpoint:   {ckpt_dir}")
    print(f"simpletrace:  {st_size:.0f} MB")
    print(f"Slices:       {n_slices} × {INST_PER_SLICE//1000}K insts")
    print(f"Output:       {trace_dir}/slice_*/o3trace.txt")
    print(f"Workers:      {args.workers}")

    if args.dry_run:
        est_min = n_slices * 60 / args.workers / 60
        print(f"\n[DRY RUN] Est time: ~{est_min:.0f} min")
        return

    os.makedirs(trace_dir, exist_ok=True)

    # Check how many already done
    existing = sum(1 for k in range(n_slices)
                   if os.path.isfile(os.path.join(trace_dir, f"slice_{k}", "o3trace.txt")))
    if existing > 0:
        print(f"\n  {existing}/{n_slices} slices already exist, will skip them")

    print(f"\nRunning {n_slices} gem5 slices...")
    t0 = time.time()

    sem = Semaphore(args.workers)
    ok = fail = skip = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for k in range(n_slices):
            fut = pool.submit(run_one_slice, bench, ckpt_dir, k, trace_dir, sem)
            futures[fut] = k

        done = 0
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                if fut.result():
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                print(f"  [slice_{k}] EXCEPTION: {e}")
                fail += 1

            done += 1
            if done % 50 == 0 or done == n_slices:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_slices - done) / rate / 60 if rate > 0 else 0
                print(f"  Progress: {done}/{n_slices} (ok={ok}, fail={fail}) "
                      f"{elapsed:.0f}s, ETA {eta:.0f}min")

    elapsed = time.time() - t0

    # Report total trace size
    total_bytes = 0
    for k in range(n_slices):
        p = os.path.join(trace_dir, f"slice_{k}", "o3trace.txt")
        if os.path.isfile(p):
            total_bytes += os.path.getsize(p)

    print(f"\nDone: {ok} ok, {fail} fail, {elapsed/60:.1f} min")
    print(f"Total trace size: {total_bytes/1e9:.2f} GB")
    print(f"Output: {trace_dir}/slice_*/o3trace.txt")


if __name__ == "__main__":
    main()