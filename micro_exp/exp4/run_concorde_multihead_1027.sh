#!/bin/bash
# Control for PRISM's semi-supervised gain: Concorde + K CPI heads + head-disagreement penalty on
# the same unlabeled data (train_exp4_concorde_multihead.py, 3873 Concorde features), 3 seeds.
#   C1      : K=1, labeled only (Concorde baseline, same GPU-resident pipeline as PRISM), 5/10/20/50%
#   C4-18k  : K=4, + 18k unlabeled rows, lam_agree 1 (V3 counterpart),   5/10/20/50%
#   C4-40k  : K=4, + 40k extra unlabeled rows, lam_agree 1 (PRISM counterpart), 5/10/20/50%
# -> results_concorde_multihead. Shared lock-protected queues; 4 workers per GPU, of which
# 2 also take 40k jobs (41 GB host RAM at load). Finished runs are skipped; the trainer refuses
# to overwrite. DRY=1 only prints the queue.
# Usage: mkdir -p logs_concorde_multihead
#        nohup bash run_concorde_multihead_1027.sh > logs_concorde_multihead/launcher.log 2>&1 &
cd "$(dirname "$0")"
PY=/data/conda_envs/hybrid/bin/python
OUT=results_concorde_multihead; LOG=logs_concorde_multihead
GPUS=(${GPUS:-0 1 2 3}); SEEDS=(${SEEDS:-42 123 456})
LOW="0.05 0.10 0.20 0.50"
pct() { python3 -c "print(round($1*100))"; }
QB=(); QS=()
for s in ${SEEDS[@]}; do
  for r in $LOW; do
    t=k4_40k_r$(pct $r)_a1_s$s; [ -e $OUT/$t.json ] || QB+=("$t --heads 4 --unlab 40k --ratio $r --seed $s")
  done
  for r in $LOW; do
    t=k4_18k_r$(pct $r)_a1_s$s; [ -e $OUT/$t.json ] || QS+=("$t --heads 4 --unlab 18k --ratio $r --seed $s")
  done
  for r in $LOW; do
    t=k1_none_r$(pct $r)_a0_s$s; [ -e $OUT/$t.json ] || QS+=("$t --heads 1 --unlab none --ratio $r --seed $s")
  done
done
echo "$(date '+%F %T') queued: 40k ${#QB[@]}, other ${#QS[@]} (seeds ${SEEDS[*]})"
if [ -n "$DRY" ]; then printf '%s\n' "${QB[@]}" "${QS[@]}" | cut -d' ' -f1; exit 0; fi

mkdir -p $OUT $LOG
Q40=$LOG/queue_40k.txt; QOT=$LOG/queue_other.txt; LOCK=$LOG/.lock
[ -e $Q40 ] || [ -e $QOT ] && { echo "queue files exist in $LOG, refusing"; exit 1; }
printf '%s\n' "${QB[@]}" > $Q40; printf '%s\n' "${QS[@]}" > $QOT
pop() { flock $LOCK bash -c "l=\$(head -n1 $1); [ -n \"\$l\" ] && sed -i 1d $1; echo \"\$l\""; }
worker() {  # $1 gpu, $2 queues
  local g=$1 q line t
  for q in $2; do
    while line=$(pop $q); [ -n "$line" ]; do
      t=${line%% *}
      echo "$(date '+%F %T') gpu$g start $t"
      $PY train_exp4_concorde_multihead.py ${line#* } --gpu $g --out $OUT > $LOG/$t.log 2>&1
      echo "$(date '+%F %T') gpu$g end   $t rc=$?"
    done
  done
}
for g in ${GPUS[@]}; do
  worker $g "$Q40 $QOT" & worker $g "$Q40 $QOT" &
  worker $g "$QOT" & worker $g "$QOT" &
done
wait
echo "$(date '+%F %T') all done; $(ls $OUT/*.json 2>/dev/null | wc -l) results"
