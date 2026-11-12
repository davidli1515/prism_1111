#!/bin/bash
# Fig. 5, [256,128] with the no-D3 setting: 3 (lam_task, lam_const, lam_mono) x seeds 42/123/456 = 9 runs, 4 GPUs.
# Same as lambda_sweep_nod3_1026 except for the hidden layers. DRY=1 only prints the commands.
cd /scratch/gem5-TAO/micro_exp/exp1
OUT=results/loss_ablation_256_nod3_1101; LOG=$OUT/logs
PY=/scratch2/wl730/conda_envs/hybrid/bin/python
[ -z "$DRY" ] && mkdir -p $LOG
i=0
for s in 42 123 456; do for cfg in "1 1 0.01" "1 1 0" "1 0 0.01"; do
  read t c m <<< "$cfg"
  g=$((i % 4)); i=$((i + 1))
  cmd="$PY train_exp1_lambda_sens.py --lam-task $t --lam-const $c --lam-mono $m --seed $s --gpu $g --input concorde --hidden 256 128 --out $OUT"
  echo "$cmd"
  [ -z "$DRY" ] && { $cmd > $LOG/t${t}_c${c}_m${m}_s${s}.log 2>&1 & sleep 10; }
done; done
[ -n "$DRY" ] && exit 0
wait
grep -h DONE $LOG/*.log
