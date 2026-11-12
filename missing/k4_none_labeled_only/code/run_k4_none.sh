#!/bin/bash
# Concorde 4 heads, labeled only (1521 steps at <=40%, same as C1), 5/10/20/50% x 3 seeds, 4 GPUs. Output: /scratch/gem5-TAO/missing/k4_none_labeled_only.
D=/scratch/gem5-TAO/missing/k4_none_labeled_only
PY=/scratch2/wl730/conda_envs/hybrid/bin/python; cd /scratch/gem5-TAO/micro_exp/exp4
mkdir -p $D/results $D/logs; i=0
for s in 42 123 456; do for r in 0.05 0.10 0.20 0.50; do
  $PY train_exp4_concorde_multihead.py --heads 4 --unlab none --ratio $r --seed $s --gpu $((i % 4)) \
    --out $D/results > $D/logs/r${r}_s$s.log 2>&1 &
  i=$((i+1)); sleep 5
done; done; wait
grep -h DONE $D/logs/*.log
