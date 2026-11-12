#!/usr/bin/env python3
"""
Validate joint 17-benchmark DSE Pareto configs against gem5 ground truth.

For a few representative Pareto-optimal configs, run gem5 on all 17 SPEC17
benchmarks, compute geometric-mean CPI (matching the surrogate objective),
and compare against the surrogate's predicted gmean CPI.

Usage:
  python validate_joint_dse.py \
      --json results_17bm_54/joint_17_gem_bo_mscs_results.json \
      --n-points 6 --insts 40000000 --workers 16 --output-dir validate_joint

  python validate_joint_dse.py --plot-only --output-dir validate_joint
"""

import argparse, os, json, shlex, subprocess, sys, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

GEM5_DIR    = "/scratch/gem5-TAO"
GEM5_BIN    = os.path.join(GEM5_DIR, "build/ARM/gem5.opt")
GEM5_CONFIG = os.path.join(GEM5_DIR, "configs/example/se_new.py")
_B          = "/home/wl730/spec17/benchspec/CPU"
CKPT        = "/scratch2/wl730/micro_ckpt/m5out_cpts"
BASE_WARMUP = 20_000_000
TIMEOUT     = 14400

# Per-benchmark: (checkpoint tick, mem-size, program args as -c/-o/--options tail)
# 'cmd_tail' is everything after the gem5 binary + config, EXCEPT -d/-P/cache/ckpt/switch.
BENCH = {
    "605.mcf_s": {
        "tick": "917300000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/605.mcf_s/exe/mcf_s_base.mytest-64',
                 '-o', f'{_B}/605.mcf_s/run/run_base_refspeed_mytest-64.0000/inp.in'],
    },
    "623.xalancbmk_s": {
        "tick": "537100000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/623.xalancbmk_s/exe/xalancbmk_s_base.mytest-64',
                 '-o', f'-v {_B}/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/t5.xml '
                       f'{_B}/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/xalanc.xsl'],
    },
    "641.leela_s": {
        "tick": "1510100000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/641.leela_s/exe/leela_s_base.mytest-64',
                 '-o', f'{_B}/641.leela_s/run/run_base_refspeed_mytest-64.0000/ref.sgf'],
    },
    "600.perlbench_s": {
        "tick": "240100000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/600.perlbench_s/exe/perlbench_s_base.mytest-64',
                 f'--options=-I{_B}/600.perlbench_s/run/run_base_refspeed_mytest-64.0000/lib '
                 f'{_B}/600.perlbench_s/run/run_base_refspeed_mytest-64.0000/checkspam.pl '
                 f'2500 5 25 11 150 1 1 1 1'],
        "cwd": f'{_B}/600.perlbench_s/run/run_base_refspeed_mytest-64.0000',
    },
    "508.namd_r": {
        "tick": "1200100000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/508.namd_r/run/run_base_refrate_mytest-64.0000/namd_r_base.mytest-64',
                 '-o', f'--input {_B}/508.namd_r/run/run_base_refrate_mytest-64.0000/apoa1.input '
                       f'--output apoa1.ref.output --iterations 65'],
    },
    "526.blender_r": {
        "tick": "43600000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/526.blender_r/run/run_base_refrate_mytest-64.0000/blender_r_base.mytest-64',
                 '-o', f'{_B}/526.blender_r/data/refrate/input/sh3_no_char.blend '
                       f'--render-output sh3_no_char_ --threads 1 -b -F RAWTGA -s 849 -e 849 -a'],
    },
    "602.gcc_s": {
        "tick": "430300000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/602.gcc_s/exe/sgcc_base.mytest-64',
                 '-o', f'{_B}/602.gcc_s/run/run_base_refspeed_mytest-64.0000/gcc-pp.c '
                       f'-O5 -fipa-pta -o gcc-pp.opts-O5_-fipa-pta.s'],
    },
    "607.cactuBSSN_s": {
        "tick": "295000000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/607.cactuBSSN_s/run/run_base_test_mytest-64.0000/cactuBSSN_s_base.mytest-64',
                 '-o', f'{_B}/607.cactuBSSN_s/run/run_base_refspeed_mytest-64.0000/spec_ref.par'],
    },
    "619.lbm_s": {
        "tick": "1900300000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/619.lbm_s/exe/lbm_s_base.mytest-64',
                 '-o', f'2000 reference.out 0 0 '
                       f'{_B}/619.lbm_s/data/refspeed/input/200_200_260_ldc.of'],
    },
    "620.omnetpp_s": {
        "tick": "85300000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/620.omnetpp_s/run/run_base_refspeed_mytest-64.0000/omnetpp_s_base.mytest-64',
                 '--options=-c General -r 0'],
        "cwd": f'{_B}/620.omnetpp_s/run/run_base_refspeed_mytest-64.0000',
    },
    "621.wrf_s": {
        "tick": "162800000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/621.wrf_s/run/run_base_refspeed_mytest-64.0000/wrf_s_base.mytest-64'],
        "cwd": f'{_B}/621.wrf_s/run/run_base_refspeed_mytest-64.0000',
    },
    "631.deepsjeng_s": {
        "tick": "1154700000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/631.deepsjeng_s/run/run_base_refspeed_mytest-64.0000/deepsjeng_s_base.mytest-64',
                 '-o', f'{_B}/631.deepsjeng_s/run/run_base_refspeed_mytest-64.0000/ref.txt'],
    },
    "638.imagick_s": {
        "tick": "41800000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/638.imagick_s/exe/imagick_s_base.mytest-64',
                 '-o', f'-limit disk 0 {_B}/638.imagick_s/run/run_base_refspeed_mytest-64.0000/refspeed_input.tga '
                       f'-resize 817% -rotate -2.76 -shave 540x375 -alpha remove -auto-level '
                       f'-contrast-stretch 1x1% -colorspace Lab -channel R -equalize +channel '
                       f'-colorspace sRGB -define histogram:unique-colors=false -adaptiveblur 0x5 '
                       f'-despeckle -auto-gamma -adaptive-sharpen 55 -enhance -brightness-contrast 10x10 '
                       f'-resize 30% {_B}/638.imagick_s/run/run_base_refspeed_mytest-64.0000/refspeed_output.tga'],
    },
    "644.nab_s": {
        "tick": "217200000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/644.nab_s/run/run_base_refspeed_mytest-64.0000/nab_s_base.mytest-64',
                 '-o', '3j1n 20140317 220'],
    },
    "648.exchange2_s": {
        "tick": "444700000000", "mem": "8GB",
        "prog": ['-c', f'{_B}/648.exchange2_s/exe/exchange2_s_base.mytest-64', '-o', '6'],
    },
    "654.roms_s": {
        "tick": "306300000000", "mem": "16GB",
        "prog": ['-c', f'{_B}/654.roms_s/run/run_base_refspeed_mytest-64.0000/sroms_base.mytest-64'],
    },
    "657.xz_s": {
        "tick": "108200000000", "mem": "32GB",
        "prog": ['-c', f'{_B}/657.xz_s/run/run_base_refspeed_mytest-64.0000/xz_s_base.mytest-64',
                 '-o', f'{_B}/657.xz_s/run/run_base_refspeed_mytest-64.0000/cpu2006docs.tar.xz '
                       f'6643 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae '
                       f'1036078272 1111795472 4'],
    },
}


def config_pflags(config):
    """Build -P and cache flags from a DSE config dict."""
    p = dict(config)
    if "issueWidth" not in p:  p["issueWidth"]  = p["dispatchWidth"]
    if "renameWidth" not in p: p["renameWidth"] = p["dispatchWidth"]
    if "wbWidth" not in p:     p["wbWidth"]     = p["dispatchWidth"]
    cpu = "system.switch_cpus_1[0]"
    bp = "TAGE" if int(float(p.get("bp_type", 0))) == 1 else "BiModeBP"
    g = lambda k: str(p[k])
    flags = [
        "-P", f"{cpu}.fetchWidth={g('fetchWidth')}",
        "-P", f"{cpu}.decodeWidth={g('decodeWidth')}",
        "-P", f"{cpu}.renameWidth={g('renameWidth')}",
        "-P", f"{cpu}.dispatchWidth={g('dispatchWidth')}",
        "-P", f"{cpu}.issueWidth={g('issueWidth')}",
        "-P", f"{cpu}.wbWidth={g('wbWidth')}",
        "-P", f"{cpu}.commitWidth={g('commitWidth')}",
        "-P", f"{cpu}.LQEntries={g('LQEntries')}",
        "-P", f"{cpu}.SQEntries={g('SQEntries')}",
        "-P", f"{cpu}.numROBEntries={g('numROBEntries')}",
        "-P", f"{cpu}.numIQEntries={g('numIQEntries')}",
        "-P", f"{cpu}.fetchBufferSize={g('fetchBufferSize')}",
        "-P", f"{cpu}.fuPool.FUList[0].count={g('IntALU')}",
        "-P", f"{cpu}.fuPool.FUList[1].count={g('IntMultDiv')}",
        "-P", f"{cpu}.fuPool.FUList[2].count={g('FP_ALU')}",
        "-P", f"{cpu}.fuPool.FUList[3].count={g('FP_MultDiv')}",
        "-P", f"{cpu}.fuPool.FUList[8].count={g('RdWrPort')}",
        f"--bp-type={bp}",
        "--l1d_size", f"{g('l1_size')}kB",
        "--l1i_size", f"{g('l1_size')}kB",
        "--l2_size",  f"{g('l2_size')}kB",
        "--l3_size",  "4096kB",
        "--l1d_assoc", g('l1d_assoc'),
        "--l1i_assoc", g('l1i_assoc'),
        "--l2_assoc",  g('l2_assoc'),
    ]
    return flags


def build_cmd(bench, config, out_dir, warmup, insts):
    info = BENCH[bench]
    argv = [GEM5_BIN, "-d", out_dir, GEM5_CONFIG,
            "--cpu-type=DerivO3CPU", "--caches", f"--mem-size={info['mem']}"]
    argv += info["prog"]
    argv += [f"--checkpoint-dir={CKPT}/{bench}/{info['tick']}",
             "-r", "1", "--standard-switch=1",
             "-W", str(warmup), "-I", str(insts)]
    argv += config_pflags(config)
    return argv


def parse_cpi(stats_path):
    if not os.path.isfile(stats_path):
        return None
    insts = cycles = cpi_direct = None
    with open(stats_path) as f:
        for line in f:
            if "switch_cpus_1" not in line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key, val = parts[0], parts[1]
            try:
                v = float(val)
            except ValueError:
                continue
            if key.endswith(".cpi"):
                cpi_direct = v
            elif key.endswith(".numInsts") or key.endswith(".committedInsts"):
                insts = v
            elif key.endswith(".numCycles"):
                cycles = v
    if cpi_direct is not None:
        return cpi_direct
    if insts and cycles and insts > 0:
        return cycles / insts
    return None


def run_one(job):
    cfg_id, bench, config, out_base, warmup, insts = job
    out_dir = os.path.abspath(os.path.join(out_base, "runs", f"cfg{cfg_id}", bench))
    os.makedirs(out_dir, exist_ok=True)
    argv = build_cmd(bench, config, out_dir, warmup, insts)
    log = os.path.join(out_dir, "run.log")
    run_cwd = BENCH[bench].get("cwd")  # some benchmarks need a specific working dir
    t0 = time.time()
    try:
        with open(log, "w") as lf:
            subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=TIMEOUT, check=False, cwd=run_cwd)
    except subprocess.TimeoutExpired:
        return {"cfg_id": cfg_id, "bench": bench, "cpi": None,
                "error": "timeout", "elapsed": time.time() - t0}
    cpi = parse_cpi(os.path.join(out_dir, "stats.txt"))
    return {"cfg_id": cfg_id, "bench": bench, "cpi": cpi,
            "error": None if cpi else "parse_failed", "elapsed": time.time() - t0}


def select_pareto(data, n_points):
    """Pick n Pareto configs spread across the area range."""
    pidx = data["pareto_indices"]
    areas = [data["areas"][i] for i in pidx]
    order = np.argsort(areas)
    pidx_sorted = [pidx[i] for i in order]
    if len(pidx_sorted) <= n_points:
        sel = pidx_sorted
    else:
        take = np.linspace(0, len(pidx_sorted) - 1, n_points).astype(int)
        sel = [pidx_sorted[i] for i in sorted(set(take))]
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="joint DSE results JSON")
    ap.add_argument("--n-points", type=int, default=6,
                    help="number of Pareto configs to validate")
    ap.add_argument("--insts", type=int, default=40_000_000)
    ap.add_argument("--warmup", type=int, default=BASE_WARMUP)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--output-dir", default="validate_joint")
    ap.add_argument("--only-benchmarks", nargs="+", default=None,
                    help="only run these benchmarks; merge into existing raw results")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    res_path = os.path.join(args.output_dir, "joint_validation_raw.json")
    sum_path = os.path.join(args.output_dir, "joint_validation_summary.json")

    if not args.plot_only:
        with open(args.json) as f:
            data = json.load(f)
        sel = select_pareto(data, args.n_points)
        print(f"Validating {len(sel)} Pareto configs × 17 benchmarks "
              f"= {len(sel)*17} gem5 runs ({args.insts:,} insts each)")

        # surrogate gmean cpi for each selected config
        surrogate = {ci: data["cpis"][ci] for ci in sel}
        areas = {ci: data["areas"][ci] for ci in sel}

        jobs = []
        bench_list = args.only_benchmarks if args.only_benchmarks else list(BENCH)
        for ci in sel:
            cfg = data["configs"][ci]
            for bench in bench_list:
                jobs.append((ci, bench, cfg, args.output_dir, args.warmup, args.insts))

        # Load existing raw results if merging (partial re-run)
        existing = []
        if args.only_benchmarks and os.path.isfile(res_path):
            with open(res_path) as f:
                existing = json.load(f)
            # drop old entries for the benchmarks we're re-running
            rerun = set(args.only_benchmarks)
            existing = [r for r in existing if r["bench"] not in rerun]
            print(f"Merging: kept {len(existing)} existing results, "
                  f"re-running {rerun}")

        raw = list(existing)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, j): j for j in jobs}
            for fut in as_completed(futs):
                r = fut.result()
                raw.append(r)
                st = f"CPI={r['cpi']:.4f}" if r["cpi"] else r["error"]
                print(f"  cfg{r['cfg_id']} {r['bench']:<18} {st} [{r['elapsed']:.0f}s]")
        with open(res_path, "w") as f:
            json.dump(raw, f, indent=2)

        # aggregate: gmean CPI per config
        summary = []
        for ci in sel:
            cpis = [r["cpi"] for r in raw if r["cfg_id"] == ci and r["cpi"]]
            n_ok = len(cpis)
            gmean = float(np.exp(np.mean(np.log(cpis)))) if n_ok == 17 else None
            summary.append({
                "cfg_id": ci, "surrogate_gmean": surrogate[ci],
                "gem5_gmean": gmean, "area": areas[ci],
                "n_ok": n_ok,
            })
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {res_path}\n       {sum_path}")
    else:
        with open(sum_path) as f:
            summary = json.load(f)

    make_plot(summary, args.output_dir)
    print_summary(summary)


def print_summary(summary):
    ok = [s for s in summary if s.get("gem5_gmean")]
    print(f"\n{'cfg':>5} {'area':>8} {'surrogate':>10} {'gem5':>10} "
          f"{'err%':>7} {'n_ok':>5}")
    print("-" * 50)
    for s in sorted(summary, key=lambda x: x["area"]):
        g = s.get("gem5_gmean")
        if g:
            err = (s["surrogate_gmean"] - g) / g * 100
            print(f"{s['cfg_id']:>5} {s['area']:>8.1f} "
                  f"{s['surrogate_gmean']:>10.4f} {g:>10.4f} {err:>+6.2f}% "
                  f"{s['n_ok']:>5}")
        else:
            print(f"{s['cfg_id']:>5} {s['area']:>8.1f} "
                  f"{s['surrogate_gmean']:>10.4f} {'--':>10} {'--':>7} "
                  f"{s['n_ok']:>5}")
    if len(ok) >= 2:
        su = np.array([s["surrogate_gmean"] for s in ok])
        ge = np.array([s["gem5_gmean"] for s in ok])
        mape = np.mean(np.abs(su - ge) / ge) * 100
        pear = np.corrcoef(su, ge)[0, 1]
        from scipy import stats as st
        rho, _ = st.spearmanr(su, ge)
        print("-" * 50)
        print(f"  MAPE={mape:.2f}%  Pearson={pear:.4f}  Spearman={rho:.4f}")


def make_plot(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11, 'axes.labelsize': 13, 'axes.linewidth': 1.5,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'figure.dpi': 600,
    })
    ok = [s for s in summary if s.get("gem5_gmean")]
    if len(ok) < 2:
        print("Not enough points to plot.")
        return
    su = np.array([s["surrogate_gmean"] for s in ok])
    ge = np.array([s["gem5_gmean"] for s in ok])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(ge, su, s=90, c="#e67e22", edgecolors='black',
               linewidth=0.8, zorder=5)
    lo = min(su.min(), ge.min()) * 0.98
    hi = max(su.max(), ge.max()) * 1.02
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, alpha=0.7, label='y = x')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("gem5 geomean CPI (ground truth)", fontweight='bold')
    ax.set_ylabel("Surrogate geomean CPI", fontweight='bold')
    ax.set_title("Joint 17-benchmark DSE: surrogate vs gem5", fontweight='bold')
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    mape = np.mean(np.abs(su - ge) / ge) * 100
    pear = np.corrcoef(su, ge)[0, 1]
    ax.text(0.05, 0.92, f"MAPE = {mape:.1f}%\nPearson r = {pear:.3f}",
            transform=ax.transAxes, fontsize=12, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFDF5',
                      edgecolor='black', linewidth=1.2))
    leg = ax.legend(loc='lower right', frameon=True, edgecolor='black',
                    fancybox=False, framealpha=1.0)
    leg.get_frame().set_linewidth(1.2)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(output_dir, f"joint_validation_scatter.{ext}")
        fig.savefig(p, dpi=600)
        print(f"  Saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()