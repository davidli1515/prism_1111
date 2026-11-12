#!/usr/bin/env python3
"""
Validate every checkpoint: 1 random config per ckpt, slice_k=0.
Ensures all 53 ckpts can run gem5 + concorde_features before large-scale launch.

Usage:
  python3 validate_pipeline.py                # test all ckpts, 20 workers
  python3 validate_pipeline.py --workers 10   # fewer workers
"""

import argparse, os, sys, time, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from gen_spec_traindata import (
    COMMANDS, CKPT_BASE, GEM5_DIR,
    INST_PER_SLICE, BASE_WARMUP, ARCH_PARAM_DIM,
    discover_checkpoints, sample_random_config, config_to_dirname,
    build_gem5_command, run_concorde_features,
    params_to_vector, get_gem5_cwd,
)

VAL_OUT = os.path.join(GEM5_DIR, "validate_pipeline")


def validate_one_ckpt(bench, ckpt_tick, output_base, sem):
    """Run 1 job for this checkpoint (slice_k=0, warmup=BASE_WARMUP)."""
    params, bp = sample_random_config()
    ckpt_dir = os.path.join(CKPT_BASE, bench, ckpt_tick)
    dp_dir = os.path.join(output_base, bench, f"ckpt_{ckpt_tick}", config_to_dirname(params))
    os.makedirs(dp_dir, exist_ok=True)

    argv = build_gem5_command(
        bench, params, bp,
        ckpt_dir,
        dp_dir, BASE_WARMUP, INST_PER_SLICE
    )

    r = {"bench": bench, "ckpt": ckpt_tick, "dp_dir": dp_dir,
         "gem5_ok": False, "concorde_ok": False,
         "t_gem5": 0, "t_concorde": 0, "error": None, "params": params}

    with sem:
        try:
            # gem5
            t1 = time.time()
            p = subprocess.Popen(argv, cwd=get_gem5_cwd(bench),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            stdout, _ = p.communicate(timeout=3600)
            r["t_gem5"] = time.time() - t1
            r["gem5_ok"] = (p.returncode == 0)

            if not r["gem5_ok"]:
                r["error"] = f"gem5 rc={p.returncode}"
                with open(os.path.join(dp_dir, "gem5_log.txt"), "w") as f:
                    f.write(stdout or "")
                return r

            # concord_analytical
            t2 = time.time()
            np.save(os.path.join(dp_dir, "arch_params.npy"), params_to_vector(params))
            concord_ok = run_concorde_features(dp_dir, ckpt_dir, BASE_WARMUP)
            r["t_concorde"] = time.time() - t2
            r["concorde_ok"] = concord_ok

            # Quick shape check
            issues = []
            for name, shape in [("concord_data.npy", (3874,)), ("gem_data.npy", (4697,))]:
                path = os.path.join(dp_dir, name)
                if not os.path.isfile(path):
                    issues.append(f"MISSING {name}")
                elif np.load(path).shape != shape:
                    issues.append(f"{name} bad shape={np.load(path).shape}")

            # MSCS monotonicity check — store results for categorized output later
            gem_path = os.path.join(dp_dir, "gem_data.npy")
            if os.path.isfile(gem_path):
                data = np.load(gem_path)
                mscs = data[-16:].reshape(4, 4)
                base = mscs[:, 0]
                fp = mscs[:, 1]
                bp_col = mscs[:, 2]
                sp = mscs[:, 3]
                tol = 1e-6  # non-strict ordering (paper Eq. 5), %.6f rounding
                base_ok = bool(np.allclose(base, base[0], atol=tol))
                fp_ok = bool(np.all(np.diff(fp) <= tol))
                bp_ok = bool(np.all(np.diff(bp_col) >= -tol))
                sp_ok = bool(np.all(np.diff(sp) <= tol))
                r["mscs_pattern"] = (base_ok, fp_ok, bp_ok, sp_ok)
                r["mscs_matrix"] = mscs.copy()
                if not (base_ok and fp_ok and bp_ok and sp_ok):
                    issues.append("MSCS violated")

            if issues:
                r["error"] = "; ".join(issues)

        except Exception as e:
            r["error"] = str(e)

    return r


def main():
    ap = argparse.ArgumentParser(description="Validate every checkpoint (1 job each)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--output-dir", default=VAL_OUT)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all valid checkpoints
    jobs = []
    for bench in COMMANDS:
        for item in discover_checkpoints(bench):
            # Handle both old format (ckpt_r, tick) and new format (tick string)
            ckpt_tick = item[1] if isinstance(item, tuple) else item
            jobs.append((bench, ckpt_tick))

    total = len(jobs)
    print(f"Found {total} valid checkpoints across {len(COMMANDS)} benchmarks")
    print(f"Running 1 validation job per checkpoint, {args.workers} workers\n")

    sem = Semaphore(args.workers)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(validate_one_ckpt, b, ct, args.output_dir, sem): (b, ct)
            for b, ct in jobs
        }
        results = []
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            ok = r["gem5_ok"] and r["concorde_ok"]
            sym = "OK" if ok else "FAIL"
            err = f"  {r['error']}" if r["error"] else ""
            print(f"  [{sym}] {r['bench']:20s} ckpt={r['ckpt']:>15s}"
                  f"  gem5={r['t_gem5']:5.1f}s  conc={r['t_concorde']:5.1f}s{err}")

    wall = time.time() - t0

    # Summary
    ok_list = [r for r in results if r["gem5_ok"] and r["concorde_ok"]]
    fail_list = [r for r in results if r not in ok_list]

    print(f"\n{'='*60}")
    print(f"PASSED: {len(ok_list)}/{total}")
    if fail_list:
        print(f"FAILED: {len(fail_list)}")
        for r in fail_list:
            print(f"  - {r['bench']}/ckpt_{r['ckpt']}: {r['error']}")
    print(f"Wall time: {wall:.0f}s")

    if ok_list:
        avg = np.mean([r["t_gem5"] + r["t_concorde"] for r in ok_list])
        print(f"Avg job time: {avg:.1f}s")
        for w in [20, 30]:
            for n, label in [(212_000, "212K"), (424_000, "424K")]:
                h = n * avg / w / 3600
                print(f"  -> {label} @ {w} workers: {h:.1f}h ({h/24:.1f}d)")

    print(f"{'='*60}")

    # ---- Categorized MSCS report ----
    from collections import defaultdict
    mscs_categories = defaultdict(list)  # pattern → list of results
    mscs_pass = 0
    for r in results:
        if "mscs_pattern" not in r:
            continue
        pat = r["mscs_pattern"]  # (base_ok, fp_ok, bp_ok, sp_ok)
        if all(pat):
            mscs_pass += 1
        else:
            mscs_categories[pat].append(r)

    mscs_total = mscs_pass + sum(len(v) for v in mscs_categories.values())
    print(f"\n{'='*60}")
    print(f"MSCS REPORT: {mscs_pass}/{mscs_total} passed all checks")

    if mscs_categories:
        for pat in sorted(mscs_categories.keys(), key=lambda p: -len(mscs_categories[p])):
            items = mscs_categories[pat]
            base_ok, fp_ok, bp_ok, sp_ok = pat
            label = f"Base={'T' if base_ok else 'F'} FP={'T' if fp_ok else 'F'} BP={'T' if bp_ok else 'F'} SP={'T' if sp_ok else 'F'}"
            print(f"\n  [{label}] count={len(items)}")
            # Show each example with full matrix
            for r in items:
                mscs = r["mscs_matrix"]
                fp = mscs[:, 1]
                bp_col = mscs[:, 2]
                sp = mscs[:, 3]
                base = mscs[:, 0]
                p = r["params"]
                print(f"    {r['bench']}/ckpt_{r['ckpt']}:")
                print(f"      Widths: fetch={p['fetchWidth']} decode={p['decodeWidth']}"
                      f" rename={p['renameWidth']} dispatch={p['dispatchWidth']}"
                      f" issue={p['issueWidth']} wb={p['wbWidth']} commit={p['commitWidth']}")
                for i, stage in enumerate(["decode", "dispatch", "issue", "commit"]):
                    print(f"      {stage:10s}: {mscs[i,0]:.6f} {mscs[i,1]:.6f} {mscs[i,2]:.6f} {mscs[i,3]:.6f}")
                print(f"      Base: {base[0]:.6f}=={base[1]:.6f}=={base[2]:.6f}=={base[3]:.6f} ? {base_ok}")
                print(f"      FP: {fp[0]:.6f}>{fp[1]:.6f}>{fp[2]:.6f}>{fp[3]:.6f} ? {fp_ok}")
                print(f"      BP: {bp_col[0]:.6f}<{bp_col[1]:.6f}<{bp_col[2]:.6f}<{bp_col[3]:.6f} ? {bp_ok}")
                print(f"      SP: {sp[0]:.6f}>{sp[1]:.6f}>{sp[2]:.6f}>{sp[3]:.6f} ? {sp_ok}")

    print(f"{'='*60}")

    if fail_list:
        print("\nFix failed checkpoints before running gen_spec_traindata.py!")
        sys.exit(1)


if __name__ == "__main__":
    main()