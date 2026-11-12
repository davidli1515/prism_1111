#!/bin/bash
# DSE v3 joint, seed 42, CPU only (no GPU). 4 methods in parallel.
cd /scratch/gem5-TAO/micro_exp/dse_v2
PY=/scratch2/wl730/conda_envs/hybrid/bin/python
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
OUT=results_v3_1006; mkdir -p $OUT/logs
for m in bo bo_v3_mscs bo_v3_scalar bo_v3_mscsavg; do
  [ -f $OUT/${m}_s42.json ] && continue
  $PY run_dse_v3.py --method $m --seed 42 --device cpu --total-queries 5000 --bo-queries 500 \
      --out $OUT > $OUT/logs/${m}_s42.log 2>&1 &
  echo "[$(date +%H:%M)] started $m (cpu)"
done
wait
$PY summarize_dse_v2.py $OUT --checkpoints 500 1000 2000 3000 5000 > $OUT/summary.txt 2>&1
echo "ALL DONE $(date)"
