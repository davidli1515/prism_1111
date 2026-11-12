#!/bin/bash
# Low-label OOD + 40k extra unlabeled (exp4 PRISM design), prism only, 23 targets, seed 42.
# Concorde / V2 / V3 baselines: results_ood_lowlabel_1006, _subset_1007, lowlabel23_1007.
# Usage: GPUS="0 1 2 3" bash run_ood_lowlabel_extra_1008.sh
cd /scratch/gem5-TAO/micro_exp/exp2
OUT=/scratch/gem5-TAO/micro_exp/exp2/results_ood_lowlabel_extra_1008
ALL=$(ls /scratch2/wl730/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/\.npy$//' | tr '\n' ' ')
LAM="--lam-task 10 --lam-const 1 --lam-mono 0.01 --lam-unlab 1 --norm subset --extra40k"
for r in 0.05 0.20; do
  tag=src$(printf '%.0f' $(echo "$r*100" | bc))
  TARGETS="$ALL" SHOTS="32 256 1024" SEEDS="42" VARIANTS="prism" RESULTS_DIR=$OUT/$tag \
  LAM_ARGS="$LAM --src-ratio $r" GPUS="${GPUS:-0 1 2 3}" MAX_JOBS=${MAX_JOBS:-8} \
  bash run_ood_lowlabel_extra.sh > $OUT.$tag.log 2>&1 &
done
wait
echo "ALL DONE $(date)"
