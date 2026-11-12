#!/bin/bash
# Variant B (Commit anchor + labeled self-consistency), forms var/rel x 3 seeds on GPUs 0,1,2
cd /scratch/gem5-TAO/micro_exp/exp1
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
OUT=results/cpi_ablation_selfcons_1007; GPUS=(0 1 2); i=0
for f in var rel; do for s in 42 123 456; do
  g=${GPUS[$((i % 3))]}; i=$((i+1))
  python train_exp1_cpi_selfcons.py --form $f --seed $s --gpu $g --out $OUT > $OUT/logs/B_${f}_s${s}.log 2>&1 &
done; done
wait; echo "ALL DONE $(date)"
