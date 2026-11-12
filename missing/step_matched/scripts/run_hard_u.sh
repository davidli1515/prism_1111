#!/bin/bash
# H-U: hard-constraint MSCS + 18k unlabeled rows, same settings as V3 (lam 10/1/0, no D3).
# 5/10/20/50% x seeds 42/123/456 = 12 runs, 4 GPUs. Output only in this scratchpad.
SP=/tmp/claude-1017/-scratch-gem5-TAO/90b1d870-5e44-4246-a121-ddc22036bb0a/scratchpad
OUT=$SP/hard_u; LOG=$SP/hard_u_logs; mkdir -p $OUT $LOG
PY=/scratch2/wl730/conda_envs/hybrid/bin/python
i=0
for s in 42 123 456; do for r in 0.05 0.10 0.20 0.50; do
  g=$((i % 4)); i=$((i + 1))
  $PY $SP/train_hard_mscs.py --variant v3 --ratio $r --lam-task 10 --lam-const 1 --lam-mono 0 \
    --input concorde --seed $s --gpu $g --out $OUT > $LOG/r${r}_s${s}.log 2>&1 &
  sleep 10
done; done
wait
grep -h DONE $LOG/*.log
