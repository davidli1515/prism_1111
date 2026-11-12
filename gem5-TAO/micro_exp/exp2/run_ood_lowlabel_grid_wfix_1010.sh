#!/bin/bash
# PRISM with a fixed per-sample unlabeled:labeled weight ratio r=0.1 (lambda_unlab = r*n_unl/n_lab),
# source ratios 50/80/100%, N in {32,256,1024,2048}, 23 targets, seed 42. Everything else as
# results_ood_lowlabel_grid_1008 (window isolation, 40k extras, --norm subset, BN fix).
# Compare against results_ood_lowlabel_grid_1008 (same runs with lambda_unlab=1).
# Usage: GPUS="0 1 2 3" MAX_JOBS=8 nohup bash run_ood_lowlabel_grid_wfix_1010.sh > run_ood_lowlabel_grid_wfix_1010.log 2>&1 &
cd /scratch/gem5-TAO/micro_exp/exp2
OUT=/scratch/gem5-TAO/micro_exp/exp2/results_ood_lowlabel_grid_wfix_1010
ALL=$(ls /scratch2/wl730/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/\.npy$//' | tr '\n' ' ')
LAM="--lam-task 10 --lam-const 1 --lam-mono 0.01 --norm subset --extra40k --unlab-weight-ratio 0.1"
mkdir -p $OUT
for r in 0.50 0.80 1.00; do
  tag=src$(printf '%.0f' $(echo "$r*100" | bc))
  TARGETS="$ALL" SHOTS="32 256 1024 2048" SEEDS="42" VARIANTS="prism" RESULTS_DIR=$OUT/$tag \
  LAM_ARGS="$LAM --src-ratio $r" GPUS="${GPUS:-0 1 2 3}" MAX_JOBS=${MAX_JOBS:-8} \
  bash run_ood_lowlabel_wfix.sh > $OUT.$tag.log 2>&1 &
done
wait
echo "ALL DONE $(date)"
