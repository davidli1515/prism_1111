#!/bin/bash
# Retry OOM-failed low-label OOD jobs on GPU 2 (sequential ratios, fewer concurrent). Skips finished runs.
cd /scratch/gem5-TAO/micro_exp/exp2
for r in 0.05 0.20; do
  tag=src$(printf '%.0f' $(echo "$r*100" | bc))
  RESULTS_DIR=/scratch/gem5-TAO/micro_exp/exp2/results_ood_lowlabel_1006/$tag \
  LAM_ARGS="--lam-task 10 --lam-const 1 --lam-mono 0.01 --lam-unlab 1 --norm batch --src-ratio $r" \
  GPUS="2" MAX_JOBS=4 bash run_ood_lowlabel.sh > run_ood_lowlabel_1006_${tag}_retry.log 2>&1
done
echo "ALL DONE $(date)"
