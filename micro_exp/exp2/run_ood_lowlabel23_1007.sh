#!/bin/bash
# Low-label OOD extension (exp4 formula --norm subset + BN fix; Concorde/V2 unaffected by --norm).
#  A: 15 remaining targets, seed 42, src 5% and 20%, N 32/256/1024   (8 targets already in *_1006 / *_1007)
#  B: all 23 targets, seeds 123 456, src 20%, N 256/1024
cd /path/to/gem5/micro_exp/exp2
OUT=/path/to/gem5/micro_exp/exp2/results_ood_lowlabel23
LAM="--lam-task 10 --lam-const 1 --lam-mono 0.01 --lam-unlab 1 --norm subset"
NEW="526.blender_r 600.perlbench_s 619.lbm_s 620.omnetpp_s 621.wrf_s 623.xalancbmk_s 631.deepsjeng_s 641.leela_s 644.nab_s 648.exchange2_s 654.roms_s brotli gumbo jsonnet wuffs "
ALL="508.namd_r 526.blender_r 600.perlbench_s 602.gcc_s 605.mcf_s 607.cactuBSSN_s 619.lbm_s 620.omnetpp_s 621.wrf_s 623.xalancbmk_s 631.deepsjeng_s 638.imagick_s 641.leela_s 644.nab_s 648.exchange2_s 654.roms_s 657.xz_s brotli gumbo jsonnet re2 snappy wuffs "
TARGETS="$NEW" SHOTS="32 256 1024" SEEDS="42" RESULTS_DIR=$OUT/src5  LAM_ARGS="$LAM --src-ratio 0.05" GPUS="0 1 2 3" MAX_JOBS=10 bash run_ood_lowlabel.sh > $OUT.A5.log 2>&1 &
TARGETS="$NEW" SHOTS="32 256 1024" SEEDS="42" RESULTS_DIR=$OUT/src20 LAM_ARGS="$LAM --src-ratio 0.20" GPUS="0 1 2 3" MAX_JOBS=10 bash run_ood_lowlabel.sh > $OUT.A20.log 2>&1 &
TARGETS="$ALL" SHOTS="256 1024"   SEEDS="123 456" RESULTS_DIR=$OUT/src20 LAM_ARGS="$LAM --src-ratio 0.20" GPUS="0 1 2 3" MAX_JOBS=10 bash run_ood_lowlabel.sh > $OUT.B20.log 2>&1 &
wait
echo "ALL DONE $(date)"
