#!/bin/bash
# Low-label OOD grid, step 1: source ratios 20% and 100%; N in {32,256,1024,2048}; 23 targets; seed 42.
#   prism    = V3 + 40k extra unlabeled (test/val windows excluded)  -> all N
#   concorde/supervised: 20% only N=2048 (32/256/1024 exist in results_ood_lowlabel / lowlabel23_1007); 100% all N
# Runs on GPUs 0 1 2. Each sub-launcher skips results already in this grid dir.
cd /path/to/gem5/micro_exp/exp2
OUT=/path/to/gem5/micro_exp/exp2/results_ood_lowlabel_grid
ALL=$(ls /data/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/\.npy$//' | tr '\n' ' ')
LAM="--lam-task 10 --lam-const 1 --lam-mono 0.01 --lam-unlab 1 --norm subset --extra40k"
G="0 1 2"
go() {  # $1 ratio  $2 tag  $3 variants  $4 shots  $5 max_jobs  $6 logname
  TARGETS="$ALL" SHOTS="$4" SEEDS="42" VARIANTS="$3" RESULTS_DIR=$OUT/$2 \
  LAM_ARGS="$LAM --src-ratio $1" GPUS="$G" MAX_JOBS=$5 \
  bash run_ood_lowlabel_extra.sh > $OUT.$6.log 2>&1
}
mkdir -p $OUT
go 0.20 src20  "prism"               "32 256 1024 2048" 9 src20_prism &
go 0.20 src20  "concorde supervised" "2048"             6 src20_base  &
go 1.00 src100 "concorde supervised prism" "32 256 1024 2048" 9 src100 &
wait
echo "ALL DONE $(date)"
