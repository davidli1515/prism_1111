#!/bin/bash
# CPI-supervision ablation at the final setting (no-D3 3873 features, lam_task 10, CPI weight 1, no L_mono):
# A, B_var, B_rel x seeds 42/123/456 = 9 runs on GPUs 0-3. C = results/lambda_sweep_nod3/t10_c1_m0_s*_nod3 (not rerun).
# Finished runs are skipped; the trainer refuses to overwrite.
# Usage: nohup bash run_cpi_ablation_nod3_1110.sh > results/cpi_ablation_nod3/launcher.log 2>&1 &
cd "$(dirname "$0")"
PY=/data/conda_envs/hybrid/bin/python
OUT=results/cpi_ablation_nod3; mkdir -p $OUT/logs
GPUS=(0 1 2 3); i=0
for v in A B_var B_rel; do for s in 42 123 456; do
  [ -e $OUT/${v}_s${s}_nod3.json ] && continue
  g=${GPUS[$((i % 4))]}; i=$((i+1))
  $PY train_exp1_cpi_ablation_nod3_1110.py --variant $v --seed $s --gpu $g --out $OUT > $OUT/logs/${v}_s${s}.log 2>&1 &
  sleep 20
done; done
wait; echo "ALL DONE $(date)"
