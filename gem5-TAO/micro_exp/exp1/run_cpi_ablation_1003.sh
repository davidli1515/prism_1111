#!/bin/bash
# Exp1 CPI-supervision ablation: 5 variants x 3 seeds on GPUs 0,1,3 (all concurrent)
cd /scratch/gem5-TAO/micro_exp/exp1
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
OUT=results/cpi_ablation_1003
GPUS=(0 1 3); i=0
for v in v1 v1_cpi_commit v1_cpi_mean v2_const v2_full; do
  for s in 42 123 456; do
    [ -f $OUT/${v}_s${s}.json ] && continue
    g=${GPUS[$((i % 3))]}; i=$((i+1))
    python train_exp1_cpi_ablation.py --variant $v --seed $s --gpu $g --out $OUT > $OUT/logs/${v}_s${s}.log 2>&1 &
  done
done
wait
echo "ALL DONE $(date)"
