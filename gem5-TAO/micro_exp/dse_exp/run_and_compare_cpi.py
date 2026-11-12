#!/usr/bin/env python3
import os
import shlex
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 全局路径和参数配置
# ============================================================
GEM5_DIR = "/scratch/gem5-TAO"
GEM5_BIN = os.path.join(GEM5_DIR, "build/ARM/gem5.opt")
GEM5_CONFIG = os.path.join(GEM5_DIR, "configs/example/se_new.py")
SPEC_BASE = "/home/wl730/spec17/benchspec/CPU"
OUT_BASE = "/scratch2/wl730/arm_n1_ablation_auto"

WARMUP_INSTS = 20_000_000
RUN_INSTS = 40_000_000

BENCH_INFO = {
    "mcf": {
        "ckpt_dir": "/scratch2/wl730/micro_ckpt/m5out_cpts/605.mcf_s/917300000000",
        "cmd_template": (
            f'{GEM5_BIN} {GEM5_CONFIG} --cpu-type=DerivO3CPU --caches --mem-size=8GB '
            f'-c "{SPEC_BASE}/605.mcf_s/exe/mcf_s_base.mytest-64" '
            f'-o "{SPEC_BASE}/605.mcf_s/run/run_base_refspeed_mytest-64.0000/inp.in"'
        )
    },
    "xalan": {
        "ckpt_dir": "/scratch2/wl730/micro_ckpt/m5out_cpts/623.xalancbmk_s/377600000000",
        "cmd_template": (
            f'{GEM5_BIN} {GEM5_CONFIG} --cpu-type=DerivO3CPU --caches --mem-size=8GB '
            f'-c "{SPEC_BASE}/623.xalancbmk_s/exe/xalancbmk_s_base.mytest-64" '
            f'-o "-v {SPEC_BASE}/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/t5.xml '
            f'{SPEC_BASE}/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/xalanc.xsl"'
        )
    }
}

BASE_ARM_N1_PARAMS = {
    "fetchWidth": "4", "decodeWidth": "4", "renameWidth": "4", "dispatchWidth": "4",
    "issueWidth": "8", "wbWidth": "8", "commitWidth": "8",
    "numROBEntries": "128", "numIQEntries": "64", "LQEntries": "72", "SQEntries": "48",
    "fetchBufferSize": "8",
    "IntALU": "3", "IntMultDiv": "1", "FP_ALU": "2", "FP_MultDiv": "2", "RdWrPort": "2",
    "l1_size": "64", "l2_size": "1024", "l1d_assoc": "4", "l1i_assoc": "4", "l2_assoc": "8",
    "bp_type": "TAGE",
}

TEST_CASES = [
    {"id": 1, "bench": "mcf",   "desc": "LQ 16 -> 48",       "base_ov": {"LQEntries": "16"}, "mod_ov": {"LQEntries": "48"}},
    {"id": 2, "bench": "mcf",   "desc": "IntALU 3 -> 6",     "base_ov": {"IntALU": "3"},     "mod_ov": {"IntALU": "6"}},
    {"id": 3, "bench": "mcf",   "desc": "ROB 128 -> 256",    "base_ov": {"numROBEntries": "128"}, "mod_ov": {"numROBEntries": "256"}},
    {"id": 4, "bench": "xalan", "desc": "IntALU 3 -> 6",     "base_ov": {"IntALU": "3"},     "mod_ov": {"IntALU": "6"}},
    {"id": 5, "bench": "xalan", "desc": "LQ 16 -> 48",       "base_ov": {"LQEntries": "16"}, "mod_ov": {"LQEntries": "48"}},
    {"id": 6, "bench": "xalan", "desc": "fetchWidth 4 -> 8", "base_ov": {"fetchWidth": "4"}, "mod_ov": {"fetchWidth": "8"}},
    {"id": 7, "bench": "mcf",   "desc": "IQ 32 -> 64",       "base_ov": {"numIQEntries": "32"}, "mod_ov": {"numIQEntries": "64"}},
]

def build_gem5_command_list(bench_name, params, out_dir):
    info = BENCH_INFO[bench_name]
    argv = shlex.split(info["cmd_template"], posix=True)
    
    argv.insert(1, "-d")
    argv.insert(2, out_dir)
    
    argv += [
        f"--checkpoint-dir={info['ckpt_dir']}",
        "-r", "1",
        "--standard-switch=1",
        "-W", str(WARMUP_INSTS),
        "-I", str(RUN_INSTS),
    ]
    
    # 【修复重点】采用显式且与你的训练代码一致的参数注入，避免拼装出错
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
        "-P", f"{cpu}.fuPool.FUList[0].count={params['IntALU']}",
        "-P", f"{cpu}.fuPool.FUList[1].count={params['IntMultDiv']}",
        "-P", f"{cpu}.fuPool.FUList[2].count={params['FP_ALU']}",
        "-P", f"{cpu}.fuPool.FUList[3].count={params['FP_MultDiv']}",
        "-P", f"{cpu}.fuPool.FUList[8].count={params['RdWrPort']}",
        f"--bp-type={params['bp_type']}",
        "--l1d_size", f"{params['l1_size']}kB",   # 这里正确拆分了 l1d_size 和 l1i_size
        "--l1i_size", f"{params['l1_size']}kB",
        "--l2_size",  f"{params['l2_size']}kB",
        "--l3_size",  "4096kB",
        "--l1d_assoc", str(params['l1d_assoc']),
        "--l1i_assoc", str(params['l1i_assoc']),
        "--l2_assoc",  str(params['l2_assoc']),
    ]
    return argv

def run_task(task_name, cmd_list, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "gem5_stdout.log")
    
    with open(log_path, "w") as log_file:
        subprocess.run(cmd_list, cwd=GEM5_DIR, stdout=log_file, stderr=subprocess.STDOUT)
    return task_name, out_dir

def extract_cpi(out_dir):
    stats_path = os.path.join(out_dir, "stats.txt")
    if not os.path.exists(stats_path):
        return None
    with open(stats_path, "r") as f:
        for line in f:
            if "system.switch_cpus_1.cpi" in line:
                match = re.search(r'system\.switch_cpus_1\.cpi\s+([0-9\.]+)', line)
                if match: return float(match.group(1))
    return None

def get_error_log(out_dir):
    """提取报错日志的最后10行，方便快速定位"""
    log_path = os.path.join(out_dir, "gem5_stdout.log")
    if not os.path.exists(log_path): return "未找到日志文件"
    with open(log_path, "r") as f:
        lines = f.readlines()
        return "".join(lines[-10:])

if __name__ == "__main__":
    tasks = []
    for case in TEST_CASES:
        base_p = BASE_ARM_N1_PARAMS.copy()
        base_p.update(case["base_ov"])
        base_dir = os.path.join(OUT_BASE, f"test{case['id']}_{case['bench']}", "baseline")
        tasks.append((f"Test {case['id']} [Base]", build_gem5_command_list(case["bench"], base_p, base_dir), base_dir))
        
        mod_p = base_p.copy()
        mod_p.update(case["mod_ov"])
        mod_dir = os.path.join(OUT_BASE, f"test{case['id']}_{case['bench']}", "modified")
        tasks.append((f"Test {case['id']} [Mod ]", build_gem5_command_list(case["bench"], mod_p, mod_dir), mod_dir))

    print(f"准备提交 {len(tasks)} 个仿真任务...")
    
    completed_dirs = {}
    with ThreadPoolExecutor(max_workers=14) as executor:
        futures = {executor.submit(run_task, name, cmd, out): (name, out) for name, cmd, out in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            name, out_dir = future.result()
            print(f"[{i}/{len(tasks)}] 完成: {name}")
            completed_dirs[name] = out_dir

    print("\n" + "="*50)
    print("                 CPI 最终对比结果")
    print("="*50)
    
    for case in TEST_CASES:
        base_name = f"Test {case['id']} [Base]"
        mod_name  = f"Test {case['id']} [Mod ]"
        
        base_dir = completed_dirs.get(base_name)
        mod_dir  = completed_dirs.get(mod_name)
        
        base_cpi = extract_cpi(base_dir) if base_dir else None
        mod_cpi  = extract_cpi(mod_dir) if mod_dir else None
        
        print(f"Test {case['id']} ({case['bench']}: {case['desc']})")
        
        if base_cpi and mod_cpi:
            diff = mod_cpi - base_cpi
            pct = (diff / base_cpi) * 100
            print(f"  Baseline CPI: {base_cpi:.4f}")
            print(f"  Modified CPI: {mod_cpi:.4f}")
            trend = "↓ (性能提升)" if diff < 0 else "↑ (性能下降)" if diff > 0 else "持平"
            print(f"  Delta CPI   : {diff:+.4f} ({pct:+.2f}%) {trend}")
        else:
            print("  [错误] 仿真失败。")
            if not base_cpi and base_dir:
                print(f"  >>> Baseline 报错日志 ({base_name}):")
                print("      " + get_error_log(base_dir).replace("\n", "\n      "))
            elif not mod_cpi and mod_dir:
                print(f"  >>> Modified 报错日志 ({mod_name}):")
                print("      " + get_error_log(mod_dir).replace("\n", "\n      "))
        print("-" * 50)