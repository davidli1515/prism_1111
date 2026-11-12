#!/bin/bash
# 6个benchmark各自最小ckpt: 直接并行跑 (新服务器版)
# 内存: perlbench/povray/exchange2=8G, bwaves/roms=16G, xz=32G
# 用法: nohup ./run_6min.sh &> run_6min.txt &

export OMP_NUM_THREADS=1

GEM5=/scratch/gem5-TAO/build/ARM/gem5.opt
SE_PY=/scratch/gem5-TAO/configs/example/se_new.py
BASE=/home/wl730/spec17/benchspec/CPU
OUT=/scratch2/wl730/micro_ckpt/m5out_cpts

PIDS=(); NAMES=(); LOGS=()
FAIL=0; SUCC=0; IDX=0

reg() {
    PIDS[$IDX]=$!; NAMES[$IDX]="$1"; LOGS[$IDX]="$2"
    IDX=$((IDX+1))
    echo "[$(date +%H:%M:%S)] ▶️ ($IDX/6) $1  PID=$!"
}

echo "===== 开始: $(date) ====="

# 600.perlbench_s @ 240.1B (8GB, CWD修复)
R=${BASE}/600.perlbench_s/run/run_base_refspeed_mytest-64.0000
mkdir -p ${OUT}/600.perlbench_s/240100000000
L=${OUT}/600.perlbench_s/240100000000/atomic_run.log
(cd $R && $GEM5 --outdir=${OUT}/600.perlbench_s/240100000000 $SE_PY --cpu-type=AtomicSimpleCPU --mem-size=8GB \
    -c "${BASE}/600.perlbench_s/exe/perlbench_s_base.mytest-64" \
    --options="-I${R}/lib ${R}/checkspam.pl 2500 5 25 11 150 1 1 1 1" \
    --take-checkpoints=240100000000 --at-instruction --max-checkpoints=1 > $L 2>&1) &
reg "600.perlbench_s/240100000000" "$L"

# 511.povray_r @ 1104.9B (8GB, CWD修复)
R=${BASE}/511.povray_r/run/run_base_refrate_mytest-64.0000
mkdir -p ${OUT}/511.povray_r/1104900000000
L=${OUT}/511.povray_r/1104900000000/atomic_run.log
(cd $R && $GEM5 --outdir=${OUT}/511.povray_r/1104900000000 $SE_PY --cpu-type=AtomicSimpleCPU --mem-size=8GB \
    -c "$R/povray_r_base.mytest-64" -o "SPEC-benchmark-ref.ini +L$R" \
    --take-checkpoints=1104900000000 --at-instruction --max-checkpoints=1 > $L 2>&1) &
reg "511.povray_r/1104900000000" "$L"

# 603.bwaves_s @ 1177.0B (16GB, OOM修复)
mkdir -p ${OUT}/603.bwaves_s/1177000000000
L=${OUT}/603.bwaves_s/1177000000000/atomic_run.log
$GEM5 --outdir=${OUT}/603.bwaves_s/1177000000000 $SE_PY --cpu-type=AtomicSimpleCPU --mem-size=16GB \
    -c "${BASE}/603.bwaves_s/run/run_base_refspeed_mytest-64.0000/speed_bwaves_base.mytest-64" \
    -o "bwaves_1" --input "${BASE}/603.bwaves_s/run/run_base_refspeed_mytest-64.0000/bwaves_1.in" \
    --take-checkpoints=1177000000000 --at-instruction --max-checkpoints=1 > $L 2>&1 &
reg "603.bwaves_s/1177000000000" "$L"

# 648.exchange2_s @ 444.7B (8GB, CWD修复)
R=${BASE}/648.exchange2_s/run/run_base_refspeed_mytest-64.0000
mkdir -p ${OUT}/648.exchange2_s/444700000000
L=${OUT}/648.exchange2_s/444700000000/atomic_run.log
(cd $R && $GEM5 --outdir=${OUT}/648.exchange2_s/444700000000 $SE_PY --cpu-type=AtomicSimpleCPU --mem-size=8GB \
    -c "${BASE}/648.exchange2_s/exe/exchange2_s_base.mytest-64" -o "6" \
    --take-checkpoints=444700000000 --at-instruction --max-checkpoints=1 > $L 2>&1) &
reg "648.exchange2_s/444700000000" "$L"

# 654.roms_s @ 306.3B (16GB, CWD+OOM修复)
R=${BASE}/654.roms_s/run/run_base_refspeed_mytest-64.0000
mkdir -p ${OUT}/654.roms_s/306300000000
L=${OUT}/654.roms_s/306300000000/atomic_run.log
(cd $R && $GEM5 --outdir=${OUT}/654.roms_s/306300000000 $SE_PY --cpu-type=AtomicSimpleCPU --mem-size=16GB \
    -c "$R/sroms_base.mytest-64" --input "$R/ocean_benchmark3.in.x" \
    --take-checkpoints=306300000000 --at-instruction --max-checkpoints=1 > $L 2>&1) &
reg "654.roms_s/306300000000" "$L"

# 657.xz_s @ 108.2B (32GB, OOM修复)
mkdir -p ${OUT}/657.xz_s/108200000000
L=${OUT}/657.xz_s/108200000000/atomic_run.log
$GEM5 --outdir=${OUT}/657.xz_s/108200000000 $SE_PY --cpu-type=AtomicSimpleCPU --mem-size=32GB \
    -c "${BASE}/657.xz_s/run/run_base_refspeed_mytest-64.0000/xz_s_base.mytest-64" \
    -o "${BASE}/657.xz_s/run/run_base_refspeed_mytest-64.0000/cpu2006docs.tar.xz 6643 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1036078272 1111795472 1" \
    --take-checkpoints=108200000000 --at-instruction --max-checkpoints=1 > $L 2>&1 &
reg "657.xz_s/108200000000" "$L"

echo "⏳ 等待 6 个任务结束..."
wait

echo ""
echo "===== 检查结果: $(date) ====="
for i in $(seq 0 $((IDX-1))); do
    dir=$(dirname "${LOGS[$i]}")
    if find "$dir" -maxdepth 1 -name "cpt.*" -type d 2>/dev/null | grep -q .; then
        echo "  ✔ ${NAMES[$i]}"; SUCC=$((SUCC+1))
    else
        ERR=$(grep -Ei "fatal:|Out of memory|Cannot open|No such file|exit code not 0" "${LOGS[$i]}" 2>/dev/null | head -1 | cut -c1-100)
        echo "  ✗ ${NAMES[$i]}"; echo "    原因: ${ERR:-未知}"; echo "    Log:  ${LOGS[$i]}"
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "===== 汇总: 总计:6  成功:$SUCC  失败:$FAIL ====="
echo "===== 结束: $(date) ====="