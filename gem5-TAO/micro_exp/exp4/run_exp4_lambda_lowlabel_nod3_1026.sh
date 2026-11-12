#!/bin/bash
# exp4 low-label loss-weight sweep WITHOUT the 808 D3 features (--input concorde, 3873 dims).
# Same runs as results_lambda_lowlabel_1004 (60) + results_lambda_probe_1005 (28), seed 42:
#   A  v2: lam_task {0.1,1,10} x lam_mono {0,0.01};  v3: lam_task {0.1,1,10} x lam_mono {0,0.01,0.1}
#      ratios 5/10/20/50%
#   B  same lam_task {0.1,1,10} x lam_mono {0,0.01} for v2 and v3 at ratios 1/2%
#   C  v3, lam_task {30,100} x lam_mono {0,0.01} at 5%
#   all lam_const 1, lam_unlab 1.  -> results_lambda_lowlabel_nod3_1026
# SEEDS="42 123 456" adds seeds. DRY=1 only prints the queue.
# Shared lock-protected queue, PER_GPU workers per GPU; finished runs (json exists) are skipped;
# the trainer refuses to overwrite.
# Usage: mkdir -p logs_lambda_lowlabel_nod3_1026
#        nohup bash run_exp4_lambda_lowlabel_nod3_1026.sh > logs_lambda_lowlabel_nod3_1026/launcher.log 2>&1 &
cd "$(dirname "$0")"
PY=/scratch2/wl730/conda_envs/hybrid/bin/python
OUT=results_lambda_lowlabel_nod3_1026; LOG=logs_lambda_lowlabel_nod3_1026
GPUS=(${GPUS:-0 1 2 3}); SEEDS=(${SEEDS:-42}); PER_GPU=${PER_GPU:-4}
f() { python3 -c "print(('%g' % $1).replace('.', 'p'))"; }
pct() { python3 -c "print(round($1*100))"; }
JOBS=()
add() {  # variant ratio lam_task lam_mono seed
  local t=$1_r$(pct $2)_t$(f $3)_c1_m$(f $4)_u1_s$5_nod3
  [ -e $OUT/$t.json ] || JOBS+=("$t --variant $1 --ratio $2 --lam-task $3 --lam-const 1 --lam-mono $4 --lam-unlab 1 --seed $5")
}
for s in ${SEEDS[@]}; do
  for r in 0.05 0.10 0.20 0.50 0.01 0.02; do for t in 0.1 1 10; do
    for m in 0 0.01; do add v2 $r $t $m $s; add v3 $r $t $m $s; done
    case $r in 0.01|0.02) ;; *) add v3 $r $t 0.1 $s ;; esac
  done; done
  for t in 30 100; do for m in 0 0.01; do add v3 0.05 $t $m $s; done; done
done
echo "$(date '+%F %T') queued ${#JOBS[@]} jobs (seeds ${SEEDS[*]}, GPUs ${GPUS[*]} x $PER_GPU)"
if [ -n "$DRY" ]; then printf '%s\n' "${JOBS[@]}" | cut -d' ' -f1; exit 0; fi

mkdir -p $OUT $LOG
Q=$LOG/queue.txt; LOCK=$LOG/.lock
[ -e $Q ] && { echo "$Q exists, refusing"; exit 1; }
printf '%s\n' "${JOBS[@]}" > $Q
pop() { flock $LOCK bash -c "l=\$(head -n1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\""; }
worker() {
  local g=$1 line t
  while line=$(pop); [ -n "$line" ]; do
    t=${line%% *}
    echo "$(date '+%F %T') gpu$g start $t"
    $PY train_exp4_lambda_sens.py ${line#* } --input concorde --gpu $g --out $OUT > $LOG/$t.log 2>&1
    echo "$(date '+%F %T') gpu$g end   $t rc=$?"
  done
}
for g in ${GPUS[@]}; do for k in $(seq $PER_GPU); do worker $g & done; done
wait
echo "$(date '+%F %T') all done; $(ls $OUT/*.json 2>/dev/null | wc -l) results"
