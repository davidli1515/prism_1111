#!/bin/bash
# Low-label lambda sensitivity: ratios {5,10,20,50}% x lam_task {0.1,1,10} x
#   V2 lam_mono {0,0.01}, V3 lam_mono {0,0.01,0.1}; 1 seed. Skips finished runs.
# Usage: GPUS="0 3" PER_GPU=5 nohup bash run_lambda_lowlabel_1004.sh > results_lambda_lowlabel_1004.run.log 2>&1 &
cd /path/to/gem5/micro_exp/exp4
export PATH=/data/conda_envs/hybrid/bin:$PATH
OUT=results_lambda_lowlabel
mkdir -p $OUT/logs
GPUS=(${GPUS:-0 3}); PER_GPU=${PER_GPU:-5}; SEED=${SEED:-42}
fmt() { printf "%g" "$1" | sed 's/\./p/'; }

JOBS=()
for r in 0.05 0.20 0.10 0.50; do          # 5% and 20% first
  for t in 0.1 1 10; do
    for m in 0 0.01;      do JOBS+=("v2 $r $t $m"); done
    for m in 0 0.01 0.1;  do JOBS+=("v3 $r $t $m"); done
  done
done

declare -A PID_GPU
echo "Start $(date) | ${#JOBS[@]} jobs | GPUs ${GPUS[*]} x $PER_GPU"
for j in "${JOBS[@]}"; do
  set -- $j
  tag="$1_r$(printf '%.0f' $(echo "$2*100" | bc))_t$(fmt $3)_c1_m$(fmt $4)_u1_s$SEED"
  [ -f $OUT/$tag.json ] && { echo "skip $tag"; continue; }
  while :; do                               # wait for a free slot
    for p in "${!PID_GPU[@]}"; do kill -0 $p 2>/dev/null || unset PID_GPU[$p]; done
    best=""; bc=999
    for g in "${GPUS[@]}"; do
      c=0; for p in "${!PID_GPU[@]}"; do [ "${PID_GPU[$p]}" = "$g" ] && c=$((c+1)); done
      [ $c -lt $bc ] && { bc=$c; best=$g; }
    done
    [ $bc -lt $PER_GPU ] && break
    sleep 15
  done
  python train_exp4_lambda_sens.py --variant $1 --ratio $2 --lam-task $3 --lam-mono $4 \
      --seed $SEED --gpu $best --out $OUT > $OUT/logs/$tag.log 2>&1 &
  PID_GPU[$!]=$best
  echo "[$(date +%H:%M)] GPU $best: $tag"
done
wait
echo "ALL DONE $(date)"
