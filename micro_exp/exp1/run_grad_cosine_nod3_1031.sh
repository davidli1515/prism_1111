#!/bin/bash
# Fig. 6 re-measurement: 4 (lam_task, lam_const, lam_mono) settings x seeds 42/123/456 = 12 runs, 4 GPUs.
#   (1, 1, 0.01) vs (1, 0, 0.01): differ only in L_const
#   (0.1, 1, 0) and (10, 1, 0):   L_mono not trained, its gradient only measured; (10, 1, 0) = chosen
# DRY=1 only prints the commands.
cd /path/to/gem5/micro_exp/exp1
OUT=results/grad_cosine_nod3; LOG=$OUT/logs
PY=/data/conda_envs/hybrid/bin/python
[ -z "$DRY" ] && mkdir -p $LOG
i=0
for s in 42 123 456; do for cfg in "1 1 0.01" "1 0 0.01" "0.1 1 0" "10 1 0"; do
  read t c m <<< "$cfg"
  g=$((i % 4)); i=$((i + 1))
  cmd="$PY grad_cosine_nod3_1031.py --lam-task $t --lam-const $c --lam-mono $m --seed $s --gpu $g --out $OUT"
  echo "$cmd"
  [ -z "$DRY" ] && { $cmd > $LOG/t${t}_c${c}_m${m}_s${s}.log 2>&1 & sleep 10; }
done; done
[ -n "$DRY" ] && exit 0
wait
grep -h DONE $LOG/*.log
