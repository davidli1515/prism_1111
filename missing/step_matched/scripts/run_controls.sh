#!/bin/bash
# Two controls on the ORIGINAL soft model (no hard-coding), lam 10/1/0, no D3, 5/10/20/50% x 3 seeds.
#  A (v3_u0):       V3 with --lam-unlab 0: unlabeled rows stay in the batch, L_const_self off.
#  B (v2_stepmatch): V2, labeled rows only, batch = 50000*ratio -> same 6084 steps as V3.
# 24 runs, 4 GPUs. Output only in this scratchpad.
SP=/tmp/claude-1017/-scratch-gem5-TAO/90b1d870-5e44-4246-a121-ddc22036bb0a/scratchpad
PY=/scratch2/wl730/conda_envs/hybrid/bin/python
COMMON="--lam-task 10 --lam-const 1 --lam-mono 0 --input concorde"
mkdir -p $SP/v3_u0 $SP/v3_u0_logs $SP/v2_stepmatch $SP/v2_stepmatch_logs
i=0
for s in 42 123 456; do for r in 0.05 0.10 0.20 0.50; do
  g=$((i % 4)); i=$((i + 1))
  $PY /scratch/gem5-TAO/micro_exp/exp4/train_exp4_lambda_sens.py --variant v3 --lam-unlab 0 --ratio $r \
    $COMMON --seed $s --gpu $g --out $SP/v3_u0 > $SP/v3_u0_logs/r${r}_s${s}.log 2>&1 &
  $PY $SP/train_v2_stepmatch.py --variant v2 --ratio $r \
    $COMMON --seed $s --gpu $g --out $SP/v2_stepmatch > $SP/v2_stepmatch_logs/r${r}_s${s}.log 2>&1 &
  sleep 10
done; done
wait
grep -h DONE $SP/v3_u0_logs/*.log $SP/v2_stepmatch_logs/*.log
