#!/bin/bash
# Fill missing results of results_ood_lowlabel_grid step 2 (src 5/50/80) with per-GPU job caps.
# Caps reflect free memory (~8 GB per prism job). Skips any result that already exists.
cd /path/to/gem5/micro_exp/exp2
export PATH=/data/conda_envs/hybrid/bin:$PATH
OUT=/path/to/gem5/micro_exp/exp2/results_ood_lowlabel_grid
LAM="--lam-task 10 --lam-const 1 --lam-mono 0.01 --lam-unlab 1 --norm subset --extra40k"
declare -A CAP=( [0]=${CAP0:-2} [1]=${CAP1:-6} [2]=${CAP2:-0} [3]=${CAP3:-2} )   # ~12 GB per prism job
ALL=$(ls /data/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/\.npy$//')
JOBS=()
add() { # ratio tag variant N target
  [ -f $OUT/$2/$3/$5/n$4_s42.json ] || JOBS+=("$1 $2 $3 $4 $5"); }
for t in $ALL; do
  for n in 32 256 1024 2048; do add 0.05 src5 prism $n $t; done
  for v in concorde supervised; do add 0.05 src5 $v 2048 $t; done
  for r in 0.50:src50 0.80:src80; do for v in concorde supervised prism; do for n in 32 256 1024 2048; do
    add ${r%%:*} ${r##*:} $v $n $t; done; done; done
done
echo "Start $(date) | missing jobs: ${#JOBS[@]}"
declare -A PG
for j in "${JOBS[@]}"; do
  set -- $j
  while :; do
    for p in "${!PG[@]}"; do kill -0 $p 2>/dev/null || unset PG[$p]; done
    g=""; for c in 1 0 3 2; do [ ${CAP[$c]} -gt 0 ] || continue; n=0; for p in "${!PG[@]}"; do [ "${PG[$p]}" = "$c" ] && n=$((n+1)); done; [ $n -lt ${CAP[$c]} ] && { g=$c; break; }; done
    [ -n "$g" ] && break; sleep 15
  done
  mkdir -p $OUT/$2/logs
  python train_ood_lowlabel_extra.py $LAM --src-ratio $1 --target $5 --n-shot $4 --variant $3 --seed 42 \
     --gpu $g --results-dir $OUT/$2 > $OUT/$2/logs/${3}_${5}_n${4}_s42.log 2>&1 &
  PG[$!]=$g; echo "[$(date +%H:%M)] GPU $g: $2 $3 $5 N=$4"
done
wait
echo "ALL DONE $(date) | done: $(ls $OUT/src{5,50,80}/*/*/*.json | wc -l)/690"
