#!/bin/bash
# Exp1 lambda sensitivity sweep (job queue, skips finished runs; safe to relaunch)
#   Grid A: lam_task x lam_mono, lam_const = 1
#   Grid B: lam_const sweep at lam_task = 1, lam_mono = 0.01
#   3 seeds each.  Usage: GPUS="0 1 2 3" PER_GPU=4 nohup bash run_lambda_sweep_1004.sh &
cd /scratch/gem5-TAO/micro_exp/exp1
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
OUT=results/lambda_sweep_1004
mkdir -p $OUT/logs
GPUS=(${GPUS:-0 1 2 3}); PER_GPU=${PER_GPU:-4}
SEEDS=(${SEEDS:-42 123 456})

fmt() { printf "%g" "$1" | sed 's/\./p/'; }
JOBS=()
for t in 0.1 0.3 1 3 10; do for m in 0 0.01 0.1 1; do JOBS+=("$t 1 $m"); done; done
for c in 0 0.1 0.3 3 10; do JOBS+=("1 $c 0.01"); done

declare -A PID_GPU
slot() {  # wait for a free slot, echo GPU with fewest jobs
  while :; do
    for p in "${!PID_GPU[@]}"; do kill -0 $p 2>/dev/null || unset PID_GPU[$p]; done
    best=""; bc=999
    for g in "${GPUS[@]}"; do
      c=0; for p in "${!PID_GPU[@]}"; do [ "${PID_GPU[$p]}" = "$g" ] && c=$((c+1)); done
      [ $c -lt $bc ] && { bc=$c; best=$g; }
    done
    [ $bc -lt $PER_GPU ] && { echo $best; return; }
    sleep 10
  done
}

echo "Start $(date) | ${#JOBS[@]} configs x ${#SEEDS[@]} seeds"
for s in "${SEEDS[@]}"; do
  for j in "${JOBS[@]}"; do
    set -- $j
    tag="t$(fmt $1)_c$(fmt $2)_m$(fmt $3)_s$s"
    [ -f $OUT/$tag.json ] && continue
    g=$(slot)
    python train_exp1_lambda_sens.py --lam-task $1 --lam-const $2 --lam-mono $3 --seed $s \
        --gpu $g --out $OUT > $OUT/logs/$tag.log 2>&1 &
    PID_GPU[$!]=$g
    echo "[$(date +%H:%M)] GPU $g: $tag"
  done
done
wait
echo "ALL DONE $(date)"
