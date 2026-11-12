#!/usr/bin/env python3
import argparse, shlex, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time

TIMEOUT_SECS = 3600 

# 你的原始命令（保持不动）
COMMANDS = {
    "602.gcc_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/602.gcc_s/exe/sgcc_base.mytest-64" -o "/home/wl730/spec17/benchspec/CPU/602.gcc_s/run/run_base_refspeed_mytest-64.0000/gcc-pp.c -O5 -fipa-pta -o gcc-pp.opts-O5_-fipa-pta.s" -I 1000000''',
    "600.perlbench_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/600.perlbench_s/exe/perlbench_s_base.mytest-64" --options="-I/home/wl730/spec17/benchspec/CPU/600.perlbench_s/run/run_base_refspeed_mytest-64.0000/lib /home/wl730/spec17/benchspec/CPU/600.perlbench_s/run/run_base_refspeed_mytest-64.0000/checkspam.pl 2500 5 25 11 150 1 1 1 1" -I 10000000''',
    "619.lbm_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/619.lbm_s/exe/lbm_s_base.mytest-64" --options="2000 reference.out 0 0 /home/wl730/spec17/benchspec/CPU/619.lbm_s/data/refspeed/input/200_200_260_ldc.of"  -I 100000''',
    "605.mcf_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/605.mcf_s/exe/mcf_s_base.mytest-64" -o "/home/wl730/spec17/benchspec/CPU/605.mcf_s/run/run_base_refspeed_mytest-64.0000/inp.in"  -I 100000''',
    "620.omnetpp_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/620.omnetpp_s/run/run_base_refspeed_mytest-64.0000/omnetpp_s_base.mytest-64" --options="-c General -r 0" -I 100000''',
    "623.xalancbmk_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/623.xalancbmk_s/exe/xalancbmk_s_base.mytest-64" -o "-v /home/wl730/spec17/benchspec/CPU/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/t5.xml /home/wl730/spec17/benchspec/CPU/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/xalanc.xsl" -I 100000''',
    # "625.x264_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/623.xalancbmk_s/exe/xalancbmk_s_base.mytest-64" --options="-v /home/wl730/spec17/benchspec/CPU/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/t5.xml /home/wl730/spec17/benchspec/CPU/623.xalancbmk_s/run/run_base_refspeed_mytest-64.0000/xalanc.xsl" -I 100000''',
    "631.deepsjeng_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=16GB -c "/home/wl730/spec17/benchspec/CPU/631.deepsjeng_s/run/run_base_refspeed_mytest-64.0000/deepsjeng_s_base.mytest-64" -o "/home/wl730/spec17/benchspec/CPU/631.deepsjeng_s/run/run_base_refspeed_mytest-64.0000/ref.txt" -I 100000''',
    "641.leela_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/641.leela_s/exe/leela_s_base.mytest-64" -o "/home/wl730/spec17/benchspec/CPU/641.leela_s/run/run_base_refspeed_mytest-64.0000/ref.sgf" -I 100000''',
    "644.nab_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/644.nab_s/run/run_base_refspeed_mytest-64.0000/nab_s_base.mytest-64" -o "3j1n 20140317 220" -I 10000''',
    "638.imagick_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/638.imagick_s/exe/imagick_s_base.mytest-64" -o "-limit disk 0 /home/wl730/spec17/benchspec/CPU/638.imagick_s/run/run_base_refspeed_mytest-64.0000/refspeed_input.tga -resize 817% -rotate -2.76 -shave 540x375 -alpha remove -auto-level -contrast-stretch 1x1% -colorspace Lab -channel R -equalize +channel -colorspace sRGB -define histogram:unique-colors=false -adaptiveblur 0x5 -despeckle -auto-gamma -adaptive-sharpen 55 -enhance -brightness-contrast 10x10 -resize 30% /home/wl730/spec17/benchspec/CPU/638.imagick_s/run/run_base_refspeed_mytest-64.0000/refspeed_output.tga" -I 10000''',
    # "649.fotonik3d_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=O3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/649.fotonik3d_s/exe/fotonik3d_s_base.mytest-64" -I 10000''',
    "657.xz_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=16GB -c "/home/wl730/spec17/benchspec/CPU/657.xz_s/run/run_base_refspeed_mytest-64.0000/xz_s_base.mytest-64" -o "/home/wl730/spec17/benchspec/CPU/657.xz_s/run/run_base_refspeed_mytest-64.0000/cpu2006docs.tar.xz 6643 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1036078272 1111795472 4" -I 10000''',
    "648.exchange2_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/648.exchange2_s/exe/exchange2_s_base.mytest-64" -o "6" -I 10000''',
    "607.cactuBSSN_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/607.cactuBSSN_s/run/run_base_test_mytest-64.0000/cactuBSSN_s_base.mytest-64" -o "/home/wl730/spec17/benchspec/CPU/607.cactuBSSN_s/run/run_base_refspeed_mytest-64.0000/spec_ref.par" -I 10000''',
    "621.wrf_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/621.wrf_s/run/run_base_refspeed_mytest-64.0000/wrf_s_base.mytest-64" -I 10000''',
    # "628.pop2_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/648.exchange2_s/exe/exchange2_s_base.mytest-64" -o "6" -I 10000''',
    "508.namd_r": r'''build/ARM/gem5.opt --debug-flags=O3PipeView --debug-file=arm_timetrace.out  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/508.namd_r/run/run_base_refrate_mytest-64.0000/namd_r_base.mytest-64"   -o "--input /home/wl730/spec17/benchspec/CPU/508.namd_r/run/run_base_refrate_mytest-64.0000/apoa1.input --output apoa1.ref.output --iterations 65" -I 10000''',
    "526.blender_r": r'''build/ARM/gem5.opt --debug-flags=O3PipeView --debug-file=arm_timetrace.out  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/526.blender_r/run/run_base_refrate_mytest-64.0000/blender_r_base.mytest-64"   -o "/home/wl730/spec17/benchspec/CPU/526.blender_r/data/refrate/input/sh3_no_char.blend --render-output sh3_no_char_ --threads 1 -b -F RAWTGA -s 849 -e 849 -a" -I 10000''',
    "511.povray_r": r'''build/ARM/gem5.opt --debug-flags=O3PipeView --debug-file=arm_timetrace.out  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/511.povray_r/run/run_base_refrate_mytest-64.0000/povray_r_base.mytest-64"   -o "SPEC-benchmark-ref.ini +L/home/wl730/spec17/benchspec/CPU/511.povray_r/run/run_base_refrate_mytest-64.0000" -I 10000''',
    "603.bwaves_s": r'''build/ARM/gem5.opt --debug-flags=O3PipeView --debug-file=arm_timetrace.out  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/603.bwaves_s/run/run_base_refspeed_mytest-64.0000/speed_bwaves_base.mytest-64"  -o "bwaves_1"  --input "/home/wl730/spec17/benchspec/CPU/603.bwaves_s/run/run_base_refspeed_mytest-64.0000/bwaves_1.in" -I 10000''',
    "654.roms_s ": r'''build/ARM/gem5.opt --debug-flags=O3PipeView --debug-file=arm_timetrace.out  /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=DerivO3CPU --caches --mem-size=4GB -c "/home/wl730/spec17/benchspec/CPU/654.roms_s/run/run_base_refspeed_mytest-64.0000/sroms_base.mytest-64"   --input "/home/wl730/spec17/benchspec/CPU/654.roms_s/run/run_base_refspeed_mytest-64.0000/ocean_benchmark3.in.x" -I 10000'''
}

# # python3 gen_spec_traindata.py \
# #   --only 602.gcc_s \
# #   --insts-default 10000 \
# #   --inst 602.gcc_s=10000 \
# #   --workers 1

def parse_inst_overrides(pairs):
    if not pairs: return {}
    def to_int(v):
        s = str(v).strip().lower().replace('_','')
        mult = 1
        if s.endswith('k'): mult, s = 1000, s[:-1]
        elif s.endswith('m'): mult, s = 1_000_000, s[:-1]
        elif s.endswith('g'): mult, s = 1_000_000_000, s[:-1]
        return int(float(s) * mult)
    out = {}
    for p in pairs:
        name, val = p.split('=',1)
        out[name.strip()] = to_int(val)
    return out

def inject_tail_param_and_I(cmd_str, bench, inst_val):
    argv = shlex.split(cmd_str, posix=True)

    # 清掉已有的 -P / -I
    i = 0
    while i < len(argv):
        if argv[i] == "-P" and i+1 < len(argv):
            del argv[i:i+2]
        else:
            i += 1
    i = 0
    while i < len(argv):
        if argv[i] == "-I" and i+1 < len(argv):
            del argv[i:i+2]
        else:
            i += 1

    param = f"system.switch_cpus_1[0].dump_inst_trace_path='/scratch2/wl730/train_data_20000000/custrom_o3_trace_{bench}.txt'"
    # argv += ["-P", param, "-I", str(inst_val), "--fast-forward=1000000000 -W 500000000 --standard-switch=1000000000"]
    argv += [
    "-P", param, 
    "-I", str(inst_val), 
    "--fast-forward=1000000000", 
    "-W", "500000000", 
    "--standard-switch=1000000000"
    ]

    current_time_seed = time.time()
    random.seed(current_time_seed)

    # ==========================================
    # 1. 定义数值范围 (基于 Table 1)
    # ==========================================
    l1_options = ["16", "32", "64", "128", "256"]
    l2_options = ["512", "1024", "2048", "4096"]
    bp_choice = random.choice(["BiModeBP", "TAGE"])
    pref_choice = random.choice(["NULL", "StridePrefetcher"])
    assoc_options = ["2", "4", "8", "16", "32"]

    # ==========================================
    # 2. 随机生成参数并存入字典
    # ==========================================
    current_params = {
        # Pipeline Widths (1-12)
        "fetchWidth": str(random.randint(1, 12)),
        "decodeWidth": str(random.randint(1, 12)),
        "renameWidth": str(random.randint(1, 12)),
        "dispatchWidth": str(random.randint(1, 12)),
        "issueWidth": str(random.randint(1, 12)),
        "wbWidth": str(random.randint(1, 12)),
        "commitWidth": str(random.randint(1, 12)),
    
        # Buffer/Queue Sizes
        "numROBEntries": str(random.randint(1, 1024)),
        "LQEntries": str(random.randint(1, 256)),
        "SQEntries": str(random.randint(1, 256)),
        "numIQEntries": str(random.randint(16, 128)), # 估算值，通常小于ROB
        "fetchBufferSize": str(random.randint(1, 8)),
    
        # Functional Units (1-8)
        "IntALU": str(random.randint(1, 8)),
        "IntMultDiv": str(random.randint(1, 8)),
        "FP_ALU": str(random.randint(1, 8)),
        "FP_MultDiv": str(random.randint(1, 8)),
        "RdWrPort": str(random.randint(1, 8)),
    
        # Caches & BP 
        "l1d_size": random.choice(l1_options),
        "l1i_size": random.choice(l1_options),
        "l2_size": random.choice(l2_options),
        "l1d_assoc": random.choice(assoc_options), 
        "l1i_assoc": random.choice(assoc_options), 
        "l2_assoc": random.choice(assoc_options),
        "bp_type": 0 if bp_choice == "BiModeBP" else 1,
        "prefetcher": 0 if pref_choice == "NULL" else 1
    }

    # ==========================================
    # 3. 将本次组合保存到 List (核心修改)
    # ==========================================
    
    current_val_combo = list(current_params.values())

    config_log = [] 
    config_log.append(current_params)

    # 如果你只是想单纯看一眼保存的数据，可以取消下面这行的注释
    # print("Generated Config:", config_log)

    # ==========================================
    # 4. 生成 gem5 需要的 argv
    # ==========================================
    argv += [
        "-P", f"system.switch_cpus_1[0].fetchWidth={current_params['fetchWidth']}",
        "-P", f"system.switch_cpus_1[0].decodeWidth={current_params['decodeWidth']}",
        "-P", f"system.switch_cpus_1[0].renameWidth={current_params['renameWidth']}",
        "-P", f"system.switch_cpus_1[0].dispatchWidth={current_params['dispatchWidth']}",
        "-P", f"system.switch_cpus_1[0].issueWidth={current_params['issueWidth']}",
        "-P", f"system.switch_cpus_1[0].wbWidth={current_params['wbWidth']}",
        "-P", f"system.switch_cpus_1[0].commitWidth={current_params['commitWidth']}",
        "-P", f"system.switch_cpus_1[0].LQEntries={current_params['LQEntries']}",
        "-P", f"system.switch_cpus_1[0].SQEntries={current_params['SQEntries']}",
        "-P", f"system.switch_cpus_1[0].numROBEntries={current_params['numROBEntries']}",
        "-P", f"system.switch_cpus_1[0].numIQEntries={current_params['numIQEntries']}",
        "-P", f"system.switch_cpus_1[0].fetchBufferSize={current_params['fetchBufferSize']}",
        "-P", f"system.switch_cpus_1[0].fupool.FUList[0].count={current_params['IntALU']}",
        "-P", f"system.switch_cpus_1[0].fupool.FUList[1].count={current_params['IntMultDiv']}",
        "-P", f"system.switch_cpus_1[0].fupool.FUList[2].count={current_params['FP_ALU']}",
        "-P", f"system.switch_cpus_1[0].fupool.FUList[3].count={current_params['FP_MultDiv']}",
        "-P", f"system.switch_cpus_1[0].fupool.FUList[8].count={current_params['RdWrPort']}",
        "--bp-type=", bp_choice,
        "--l1d_size", f"{current_params['l1d_size']}kB",
        "--l1i_size", f"{current_params['l1i_size']}kB",
        "--l2_size", f"{current_params['l2_size']}kB",
        "--l3_size", "4096kB",
    
        "--l1d_assoc", current_params['l1d_assoc'],
        "--l1i_assoc", current_params['l1i_assoc'],
        "--l2_assoc", current_params['l2_assoc']
    ]
    
    # 只有当预取器不是 NULL 时才加入该 flag
    if current_params['prefetcher'] != "NULL":
        argv += ["--l1d-hwp-type=", pref_choice]

    return current_val_combo, argv

def run_one(bench, insts_default, inst_overrides):
    cmd = COMMANDS[bench]
    inst_val = inst_overrides.get(bench, insts_default)
    current_val_combo, argv = inject_tail_param_and_I(cmd, bench, inst_val)
    print(current_val_combo)

    # try:
    #     # 用 Popen + communicate(timeout=...)，超时就杀
    #     p = subprocess.Popen(
    #         argv,
    #         text=True,
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.STDOUT
    #     )
    #     try:
    #         out, _ = p.communicate(timeout=TIMEOUT_SECS)
    #         ok = (p.returncode == 0)
    #         return bench, ok, ("" if ok else out[-1200:])
    #     except subprocess.TimeoutExpired:
    #         # 超时强杀
    #         p.kill()
    #         try:
    #             out, _ = p.communicate(timeout=5)
    #         except Exception:
    #             out = ""
    #         return bench, False, f"[TIMEOUT > {TIMEOUT_SECS}s] {out[-800:]}"
    # except Exception as e:
    #     return bench, False, str(e)

def main():
    ap = argparse.ArgumentParser(description="Parallel gem5 runner (append -P/-I after config script)")
    ap.add_argument("--only", nargs="*", help="subset of benchmarks to run (keys like 602.gcc_s)")
    ap.add_argument("--insts-default", required=True, help="default -I (e.g., 1000000 / 500k / 2m)")
    ap.add_argument("--inst", action="extend", nargs="+", default=[],
                    help="per-bench override: --inst 602.gcc_s=2m 600.perlbench_s=10m")
    ap.add_argument("--workers", type=int, default=4)  # 先留着，用不到也无所谓
    args = ap.parse_args()

    insts_default = parse_inst_overrides([f"all={args.insts_default}"])["all"]
    inst_overrides = parse_inst_overrides(args.inst)

    benches = args.only if args.only else list(COMMANDS.keys())

    ok_list, fail_list = [], []

    for b in benches:
        try:
            b, ok, info = run_one(b, insts_default, inst_overrides)
            print(f"[{'OK' if ok else 'FAIL'}] {b}")
            if not ok and info:
                print(info)
            (ok_list if ok else fail_list).append(b)
        except Exception as e:
            print(f"[FAIL] {b}")
            print(e)
            fail_list.append(b)

    print("\n=== Summary ===")
    print("Success:", ok_list)
    print("Fail   :", fail_list)

if __name__ == "__main__":
    main()