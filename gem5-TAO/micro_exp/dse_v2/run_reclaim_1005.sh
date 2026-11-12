#!/bin/bash
# Post-BO reclaim comparison, 4 methods in parallel (GPU 0: mscs, random; GPU 3: scalar_jac, exhaustive)
cd /scratch/gem5-TAO/micro_exp/dse_v2
PY=/scratch2/wl730/conda_envs/hybrid/bin/python
export OMP_NUM_THREADS=4
OUT=results_v2_1005/reclaim; mkdir -p $OUT/logs
for spec in "mscs 0" "random 0" "scalar_jac 3" "exhaustive 3"; do
  set -- $spec
  [ -f $OUT/$1_s42.json ] && continue
  $PY mscs_reclaim.py --method $1 --device cuda:$2 --budget 1000 --out $OUT > $OUT/logs/$1.log 2>&1 &
done
wait
echo "ALL DONE $(date)"
