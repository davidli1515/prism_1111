#!/usr/bin/env python3
"""
Pre-generate simpletrace.txt for each checkpoint.
One gem5 run per ckpt: warmup 60M instructions (TimingSimpleCPU), -I 1000.
Output: <ckpt_dir>/simpletrace.txt

Usage:
  python3 gen_simpletrace.py --workers 20
  python3 gen_simpletrace.py --only 602.gcc_s 508.namd_r
  python3 gen_simpletrace.py --dry-run
"""

import argparse, os, sys, time, subprocess, shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from gen_spec_traindata import (
    GEM5_DIR, GEM5_BIN, GEM5_CONFIG, CKPT_BASE,
    COMMANDS, BENCH_CWD,
    discover_checkpoints, get_gem5_cwd,
)

WARMUP_FOR_SIMPLETRACE = 60_000_000  # 6000万 = covers first 400 slices


def build_simpletrace_cmd(bench, ckpt_dir, simpletrace_path):
    """Build gem5 command for simpletrace generation.
    Micro-arch params don't matter (warmup is TimingSimpleCPU).
    """
    argv = shlex.split(COMMANDS[bench], posix=True)

    # Remove template -I
    i = 0
    while i < len(argv):
        if argv[i] == "-I" and i + 1 < len(argv):
            del argv[i:i + 2]
        else:
            i += 1

    # Output dir = ckpt_dir (stats.txt etc. go here, will be overwritten but that's fine)
    m5out = os.path.join(ckpt_dir, "simpletrace_m5out")
    argv.insert(1, "-d")
    argv.insert(2, m5out)

    argv += [
        f"--checkpoint-dir={ckpt_dir}",
        "-r", "1",
        "--standard-switch=1",
        "-W", str(WARMUP_FOR_SIMPLETRACE),
        "-I", "1000",
        "-P", f"system.switch_cpus[0].dump_inst_trace_path='{simpletrace_path}'",
    ]
    return argv


def run_one(bench, ckpt_tick, sem):
    """Generate simpletrace.txt for one checkpoint."""
    ckpt_dir = os.path.join(CKPT_BASE, bench, ckpt_tick)
    simpletrace_path = os.path.join(ckpt_dir, "simpletrace.txt")
    tag = f"{bench}/ckpt_{ckpt_tick}"

    # Skip if already exists
    if os.path.isfile(simpletrace_path):
        size_mb = os.path.getsize(simpletrace_path) / 1e6
        print(f"  [SKIP] {tag} (simpletrace.txt exists, {size_mb:.0f}MB)")
        return True

    argv = build_simpletrace_cmd(bench, ckpt_dir, simpletrace_path)

    with sem:
        t0 = time.time()
        try:
            p = subprocess.Popen(
                argv, cwd=get_gem5_cwd(bench),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            stdout, _ = p.communicate(timeout=7200)
            elapsed = time.time() - t0
            ok = (p.returncode == 0)

            if ok and os.path.isfile(simpletrace_path):
                size_mb = os.path.getsize(simpletrace_path) / 1e6
                print(f"  [OK]   {tag}  {elapsed:.0f}s  {size_mb:.0f}MB")
            else:
                print(f"  [FAIL] {tag}  {elapsed:.0f}s  rc={p.returncode}")
                log_path = os.path.join(ckpt_dir, "simpletrace_gen_log.txt")
                with open(log_path, "w") as f:
                    f.write(stdout or "")
            return ok

        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate(timeout=10)
            print(f"  [TIMEOUT] {tag}")
            return False
        except Exception as e:
            print(f"  [ERR] {tag}: {e}")
            return False


def main():
    ap = argparse.ArgumentParser(description="Pre-generate simpletrace.txt for each checkpoint")
    ap.add_argument("--only", nargs="*", help="Subset of benchmarks")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    benches = args.only if args.only else list(COMMANDS.keys())
    for b in benches:
        if b not in COMMANDS:
            print(f"[ERROR] Unknown benchmark: {b}")
            sys.exit(1)

    # Discover jobs
    jobs = []
    for bench in benches:
        for ckpt_tick in discover_checkpoints(bench):
            jobs.append((bench, ckpt_tick))

    print(f"Found {len(jobs)} checkpoints to process, {args.workers} workers")

    if args.dry_run:
        for bench, ckpt_tick in jobs:
            ckpt_dir = os.path.join(CKPT_BASE, bench, ckpt_tick)
            st_path = os.path.join(ckpt_dir, "simpletrace.txt")
            argv = build_simpletrace_cmd(bench, ckpt_dir, st_path)
            print(f"\n# {bench}/ckpt_{ckpt_tick}")
            print(f"cd {get_gem5_cwd(bench)} && \\")
            print(f"  {' '.join(argv)}")
        print(f"\nTotal: {len(jobs)} jobs")
        return

    sem = Semaphore(args.workers)
    t0 = time.time()
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, b, ct, sem): (b, ct) for b, ct in jobs}
        for fut in as_completed(futs):
            if fut.result():
                ok += 1
            else:
                fail += 1

    print(f"\n{'='*50}")
    print(f"DONE  ok={ok}  fail={fail}  wall={time.time()-t0:.0f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()