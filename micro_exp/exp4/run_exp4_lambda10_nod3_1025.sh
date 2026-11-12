#!/bin/bash
# exp4 label efficiency WITHOUT the 808 D3 features (input = first 3873 Concorde features,
# --input concorde), aligned with the Concorde baseline. 3 seeds.
#   V1    : L_task only (lam_task 1), as train_exp4_vanilla.py, ratios 5-90%  -> results_lambda10_nod3_main
#   V2/V3 : lam_task 10, lam_const 1, lam_mono 0, ratios 5-90%                -> results_lambda10_nod3_main
#   PRISM : same weights, V3 + 40k extras (original split), ratios 5-100%     -> results_lambda10_nod3_prism
# Shared lock-protected queues; 4 workers per GPU; PRISM (41 GB host RAM at load) only on
# 2 workers per GPU. Finished runs (json exists) are skipped; trainers refuse to overwrite.
# Usage: mkdir -p logs_lambda10_nod3; nohup bash run_exp4_lambda10_nod3_1025.sh > logs_lambda10_nod3/launcher.log 2>&1 &
cd "$(dirname "$0")"
PY=/data/conda_envs/hybrid/bin/python
MAIN=results_lambda10_nod3_main; PRISM=results_lambda10_nod3_prism; LOG=logs_lambda10_nod3
GPUS=(0 1 2 3); SEEDS=(42 123 456)
RATIOS="0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80 0.90"
W10="--lam-task 10 --lam-const 1 --lam-mono 0"; W1="--lam-task 1 --lam-const 0 --lam-mono 0"
mkdir -p $MAIN $PRISM $LOG
QP=$LOG/queue_prism.txt; QM=$LOG/queue_main.txt; LOCK=$LOG/.lock
[ -e $QP ] || [ -e $QM ] && { echo "queue files exist in $LOG, refusing"; exit 1; }

pct() { python3 -c "print(round($1*100))"; }
for s in ${SEEDS[@]}; do
  for r in $RATIOS 1.0; do
    t=prism_r$(pct $r)_t10_c1_m0_u1_s${s}_nod3
    [ -e $PRISM/$t.json ] || echo "$t train_exp4_prism_lambda.py --ratio $r --out $PRISM --seed $s $W10" >> $QP
  done
  for r in $RATIOS; do
    t=v1_r$(pct $r)_t1_c0_m0_u1_s${s}_nod3
    [ -e $MAIN/$t.json ] || echo "$t train_exp4_lambda_sens.py --variant v1 --ratio $r --out $MAIN --seed $s $W1" >> $QM
    for v in v2 v3; do
      t=${v}_r$(pct $r)_t10_c1_m0_u1_s${s}_nod3
      [ -e $MAIN/$t.json ] || echo "$t train_exp4_lambda_sens.py --variant $v --ratio $r --out $MAIN --seed $s $W10" >> $QM
    done
  done
done
touch $QP $QM
echo "queued: prism $(wc -l < $QP), v1/v2/v3 $(wc -l < $QM)"

pop() {  # pop first line of queue $1 under lock
  flock $LOCK bash -c "l=\$(head -n1 $1); [ -n \"\$l\" ] && sed -i 1d $1; echo \"\$l\""
}
worker() {  # $1 gpu, $2 queues (space separated)
  local g=$1 q line t
  for q in $2; do
    while line=$(pop $q); [ -n "$line" ]; do
      t=${line%% *}
      echo "$(date '+%F %T') gpu$g start $t"
      $PY ${line#* } --input concorde --gpu $g > $LOG/$t.log 2>&1
      echo "$(date '+%F %T') gpu$g end   $t rc=$?"
    done
  done
}
for g in ${GPUS[@]}; do
  worker $g "$QP $QM" & worker $g "$QP $QM" &
  worker $g "$QM" & worker $g "$QM" &
done
wait
echo "$(date '+%F %T') all done"
