#!/bin/bash
# DSE v3 (joint, finite differences): 4 methods x 3 seeds, GPU 0 / 3 (surrogate), McPAT on CPU.
# NOT started automatically. Launch: setsid nohup bash run_v3_1005.sh > run_v3_1005.log 2>&1 < /dev/null &
cd /scratch/gem5-TAO/micro_exp/dse_v2
PY=/scratch2/wl730/conda_envs/hybrid/bin/python
export OMP_NUM_THREADS=2
OUT=results_v3_1005; mkdir -p $OUT/logs
JOBS=${JOBS:-6}; i=0
for s in 42 123 456; do
  for m in bo bo_v3_scalar bo_v3_mscs bo_v3_mscsavg; do
    [ -f $OUT/${m}_s${s}.json ] && continue
    while [ $(jobs -rp | wc -l) -ge $JOBS ]; do sleep 30; done
    g=$([ $((i % 2)) -eq 0 ] && echo 0 || echo 3); i=$((i+1))
    $PY run_dse_v3.py --method $m --seed $s --device cuda:$g --out $OUT > $OUT/logs/${m}_s${s}.log 2>&1 &
  done
done
wait
$PY summarize_dse_v2.py $OUT --checkpoints 500 1000 2000 3000 5000
echo "ALL DONE $(date)"
