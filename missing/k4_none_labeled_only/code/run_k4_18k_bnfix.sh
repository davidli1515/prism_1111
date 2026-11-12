#!/bin/bash
# Concorde 4 heads + 18k unlabeled rows (agreement loss, lam 1); same steps and labeled batches as the
# single-head labeled-only baseline; unlabeled rows forwarded with eval-mode BN. 5/10/20/50% x 3 seeds, 4 GPUs.
D=/scratch/gem5-TAO/missing/k4_none_labeled_only
PY=/scratch2/wl730/conda_envs/hybrid/bin/python; cd /scratch/gem5-TAO/micro_exp/exp4
mkdir -p $D/results_18k_bnfix $D/logs_18k_bnfix; i=0
for s in 42 123 456; do for r in 0.05 0.10 0.20 0.50; do
  $PY $D/code/train_k4_18k_bnfix.py --heads 4 --lam 1 --ratio $r --seed $s --gpu $((i % 4)) \
    --out $D/results_18k_bnfix > $D/logs_18k_bnfix/r${r}_s$s.log 2>&1 &
  i=$((i+1)); sleep 5
done; done; wait
grep -h DONE $D/logs_18k_bnfix/*.log
