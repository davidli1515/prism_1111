#!/bin/bash

# 确保脚本遇到错误时不直接退出
set +e 

echo "============================================================"
echo "                       [ 测试 A ]                           "
echo "============================================================"
echo "正在静默运行 gem5 (A: fetchWidth=8)... 请稍候。"

build/ARM/gem5.opt --debug-file=arm_timetrace.out \
    /scratch/gem5-TAO/configs/example/se_new.py \
    --cpu-type=DerivO3CPU \
    --caches \
    --mem-size=4GB \
    -c "/home/wl730/spec17/benchspec/CPU/602.gcc_s/exe/sgcc_base.mytest-64" \
    -o "/home/wl730/spec17/benchspec/CPU/602.gcc_s/run/run_base_refspeed_mytest-64.0000/gcc-pp.c -O5 -fipa-pta -o gcc-pp.opts-O5_-fipa-pta.s" \
    -P "system.switch_cpus_1[0].fetchWidth=8" \
    -P "system.switch_cpus_1[0].dump_inst_trace_path='/scratch/gem5-TAO/0115_602_gcc_s_test.txt'" \
    -I 10000 --fast-forward=10000000 -W 1000000 --standard-switch=10000000 --bp-type=TAGE_SC_L_8KB > /dev/null 2>&1

python3 process_test_result.py
echo -e "\n"


echo "============================================================"
echo "                       [ 测试 B ]                           "
echo "============================================================"
echo "正在静默运行 gem5 (B: fetchWidth=2)... 请稍候。"

build/ARM/gem5.opt --debug-file=arm_timetrace.out \
    /scratch/gem5-TAO/configs/example/se_new.py \
    --cpu-type=DerivO3CPU \
    --caches \
    --mem-size=4GB \
    -c "/home/wl730/spec17/benchspec/CPU/602.gcc_s/exe/sgcc_base.mytest-64" \
    -o "/home/wl730/spec17/benchspec/CPU/602.gcc_s/run/run_base_refspeed_mytest-64.0000/gcc-pp.c -O5 -fipa-pta -o gcc-pp.opts-O5_-fipa-pta.s" \
    -P "system.switch_cpus_1[0].fetchWidth=2" \
    -P "system.switch_cpus_1[0].dump_inst_trace_path='/scratch/gem5-TAO/0115_602_gcc_s_test.txt'" \
    -I 10000 --fast-forward=10000000 -W 1000000 --standard-switch=10000000 --bp-type=TAGE_SC_L_8KB > /dev/null 2>&1

python3 process_test_result.py
echo -e "\n"


echo "============================================================"
echo "                       [ 测试 C ]                           "
echo "============================================================"
echo "正在静默运行 gem5 (C: fetchWidth=8, dispatchWidth=4, issueWidth=4)... 请稍候。"

build/ARM/gem5.opt --debug-file=arm_timetrace.out \
    /scratch/gem5-TAO/configs/example/se_new.py \
    --cpu-type=DerivO3CPU \
    --caches \
    --mem-size=4GB \
    -c "/home/wl730/spec17/benchspec/CPU/602.gcc_s/exe/sgcc_base.mytest-64" \
    -o "/home/wl730/spec17/benchspec/CPU/602.gcc_s/run/run_base_refspeed_mytest-64.0000/gcc-pp.c -O5 -fipa-pta -o gcc-pp.opts-O5_-fipa-pta.s" \
    -P "system.switch_cpus_1[0].fetchWidth=8" \
    -P "system.switch_cpus_1[0].dispatchWidth=4" \
    -P "system.switch_cpus_1[0].issueWidth=4" \
    -P "system.switch_cpus_1[0].dump_inst_trace_path='/scratch/gem5-TAO/0115_602_gcc_s_test.txt'" \
    -I 10000 --fast-forward=10000000 -W 1000000 --standard-switch=10000000 --bp-type=TAGE_SC_L_8KB > /dev/null 2>&1

python3 process_test_result.py
echo -e "\n"

echo "============================================================"
echo "                   所有测试已依次执行完毕！                 "
echo "============================================================"