#!/bin/bash
# Reclaim v2 (benefit/cost ranking): GPU 0: mscs_area, area_only; GPU 3: scalar_area, exhaustive
cd /scratch/gem5-TAO/micro_exp/dse_v2
PY=/scratch2/wl730/conda_envs/hybrid/bin/python
export OMP_NUM_THREADS=4
OUT=results_v2_1005/reclaim_v2; mkdir -p $OUT/logs
for spec in "mscs_area 0" "area_only 0" "scalar_area 3" "exhaustive 3"; do
  set -- $spec
  [ -f $OUT/$1_s42.json ] && continue
  $PY mscs_reclaim_v2.py --method $1 --device cuda:$2 --budget 1000 --out $OUT > $OUT/logs/$1.log 2>&1 &
done
wait
$PY summarize_reclaim_v2.py > reclaim_v2_summary_1005.txt 2>&1
echo "ALL DONE $(date)"
