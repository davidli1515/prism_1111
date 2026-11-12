#!/bin/bash
# Probe for a real cost of large lam_task (seed 42, lam_const = 1), on one GPU.
#   P1 fewer labels : V2/V3 at 1%, 2% labels x lam_task {0.1,1,10} x lam_mono {0,0.01}   (24)
#   P2 larger lam_task, 5% labels : V3 x lam_task {30,100} x lam_mono {0,0.01}            (4)
#   P3 larger lam_task, 100% labels: exp1 recipe x lam_task {30,100} x lam_mono {0,0.01}  (4)
# Separate output dirs (existing sweeps untouched). Skips finished runs.
# Usage: GPU=3 PER_GPU=5 nohup bash run_lambda_probe_1005.sh > results_lambda_probe_1005.run.log 2>&1 &
cd /scratch/gem5-TAO/micro_exp/exp4
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
GPU=${GPU:-3}; PER_GPU=${PER_GPU:-5}; SEED=42
OUT4=results_lambda_probe_1005
OUT1=/scratch/gem5-TAO/micro_exp/exp1/results/lambda_probe_1005
mkdir -p $OUT4/logs $OUT1/logs
fmt() { printf "%g" "$1" | sed 's/\./p/'; }

JOBS=()   # "exp4 variant ratio t m" | "exp1 t m"
for r in 0.01 0.02; do for t in 0.1 1 10; do for v in v2 v3; do for m in 0 0.01; do
  JOBS+=("exp4 $v $r $t $m"); done; done; done; done
for t in 30 100; do for m in 0 0.01; do JOBS+=("exp4 v3 0.05 $t $m"); JOBS+=("exp1 $t $m"); done; done

PIDS=()
wait_slot() {
  while :; do
    alive=(); for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null && alive+=($p); done; PIDS=("${alive[@]}")
    [ ${#PIDS[@]} -lt $PER_GPU ] && return; sleep 15
  done
}

echo "Start $(date) | ${#JOBS[@]} jobs | GPU $GPU x $PER_GPU"
for j in "${JOBS[@]}"; do
  set -- $j
  if [ "$1" = exp4 ]; then
    tag="$2_r$(printf '%.0f' $(echo "$3*100" | bc))_t$(fmt $4)_c1_m$(fmt $5)_u1_s$SEED"
    [ -f $OUT4/$tag.json ] && { echo "skip $tag"; continue; }
    wait_slot
    python train_exp4_lambda_sens.py --variant $2 --ratio $3 --lam-task $4 --lam-mono $5 \
        --seed $SEED --gpu $GPU --out $OUT4 > $OUT4/logs/$tag.log 2>&1 &
  else
    tag="t$(fmt $2)_c1_m$(fmt $3)_s$SEED"
    [ -f $OUT1/$tag.json ] && { echo "skip $tag"; continue; }
    wait_slot
    (cd ../exp1 && python train_exp1_lambda_sens.py --lam-task $2 --lam-const 1 --lam-mono $3 \
        --seed $SEED --gpu $GPU --out $OUT1 > $OUT1/logs/$tag.log 2>&1) &
  fi
  PIDS+=($!)
  echo "[$(date +%H:%M)] $tag"
done
wait
echo "ALL DONE $(date)"
