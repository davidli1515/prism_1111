#!/bin/bash
# Cross-program OOD lambda sweep, SUPERVISED (V2, no unlabeled rows), sources 20% labeled:
#   lam_task {0.1,10} x lam_mono {0,0.01} x N {256,1024} x 8 targets, seed 42 (32 runs).
# Memory-aware queue: a job starts only on a GPU with >= MIN_FREE_MB free memory and
# < PER_GPU jobs of this queue; otherwise it waits. Skips finished runs (safe to relaunch).
cd /path/to/gem5/micro_exp/exp2
export PATH=/data/conda_envs/hybrid/bin:$PATH
OUT=/path/to/gem5/micro_exp/exp2/results_ood_lambda_v2
GPUS=(${GPUS:-0 1 2 3}); MIN_FREE_MB=${MIN_FREE_MB:-14000}; PER_GPU=${PER_GPU:-4}
BMS=(602.gcc_s 605.mcf_s 607.cactuBSSN_s 638.imagick_s 508.namd_r snappy re2 657.xz_s)
declare -A PID_GPU

reap() { for p in "${!PID_GPU[@]}"; do kill -0 $p 2>/dev/null || unset PID_GPU[$p]; done; }
free_gpu() {   # echo a GPU with enough free memory and spare queue slots, or nothing
  reap
  for g in "${GPUS[@]}"; do
    local c=0; for p in "${!PID_GPU[@]}"; do [ "${PID_GPU[$p]}" = "$g" ] && c=$((c+1)); done
    [ $c -ge $PER_GPU ] && continue
    local f=$(nvidia-smi --id=$g --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    [ "$f" -ge "$MIN_FREE_MB" ] && { echo $g; return; }
  done
}

echo "Start $(date) | GPUs ${GPUS[*]} | min free ${MIN_FREE_MB} MB | per GPU ${PER_GPU}"
for t in 0.1 10; do for m in 0 0.01; do for N in 256 1024; do for BM in "${BMS[@]}"; do
  tag="t$(printf %g $t | sed 's/\./p/')_m$(printf %g $m | sed 's/\./p/')"
  RD=$OUT/$tag/src20; mkdir -p $RD/logs
  [ -f $RD/supervised/$BM/n${N}_s42.json ] && { echo "[skip] $tag $BM N=$N"; continue; }
  g=""; while [ -z "$g" ]; do g=$(free_gpu); [ -z "$g" ] && sleep 60; done
  python train_ood_lowlabel.py --lam-task $t --lam-const 1 --lam-mono $m --lam-unlab 1 --norm subset \
      --src-ratio 0.20 --target $BM --n-shot $N --variant supervised --seed 42 --gpu $g \
      --results-dir $RD > $RD/logs/supervised_${BM}_n${N}_s42.log 2>&1 &
  PID_GPU[$!]=$g
  echo "[$(date +%H:%M)] GPU $g: $tag $BM N=$N"
  sleep 90    # let the job allocate memory before the next free-memory check
done; done; done; done
wait
echo "ALL DONE $(date)"
