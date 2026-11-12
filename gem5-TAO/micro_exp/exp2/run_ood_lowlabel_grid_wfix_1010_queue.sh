#!/bin/bash
# Weight-fix PRISM grid (r=0.1), src 50/80/100%, N 32/256/1024/2048, 23 targets, seed 42.
# Memory-aware queue: launch on a GPU only if it has >= FREE_MB free (other users share the
# machine); ~12 GB per job. Up to 3 rounds; OOM logs moved aside between rounds.
cd /scratch/gem5-TAO/micro_exp/exp2
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
OUT=/scratch/gem5-TAO/micro_exp/exp2/results_ood_lowlabel_grid_wfix_1010
LAM="--lam-task 10 --lam-const 1 --lam-mono 0.01 --norm subset --extra40k --unlab-weight-ratio 0.1"
FREE_MB=${FREE_MB:-15000}; GPUS=(${GPUS:-0 1 2 3}); MAXPER=${MAXPER:-7}
ALL=$(ls /scratch2/wl730/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/\.npy$//')
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1; }
for round in 1 2 3; do
  JOBS=()
  for t in $ALL; do for r in 0.50:src50 0.80:src80 1.00:src100; do for n in 32 256 1024 2048; do
    [ -f $OUT/${r##*:}/prism/$t/n${n}_s42.json ] || JOBS+=("${r%%:*} ${r##*:} $n $t"); done; done; done
  echo "[$(date)] round $round: ${#JOBS[@]} missing"
  [ ${#JOBS[@]} -eq 0 ] && break
  for r in src50 src80 src100; do mkdir -p $OUT/$r/logs $OUT/$r/logs_failed; f=$(grep -l Traceback $OUT/$r/logs/*.log 2>/dev/null); [ -n "$f" ] && mv $f $OUT/$r/logs_failed/; done
  declare -A PG=()
  for j in "${JOBS[@]}"; do
    set -- $j
    while :; do
      for p in "${!PG[@]}"; do kill -0 $p 2>/dev/null || unset PG[$p]; done
      g=""; for c in "${GPUS[@]}"; do n=0; for p in "${!PG[@]}"; do [ "${PG[$p]}" = "$c" ] && n=$((n+1)); done
        [ $n -lt $MAXPER ] && [ $(free_mb $c) -ge $FREE_MB ] && { g=$c; break; }; done
      [ -n "$g" ] && break; sleep 30
    done
    python train_ood_lowlabel_wfix.py $LAM --src-ratio $1 --target $4 --n-shot $3 --variant prism --seed 42 \
       --gpu $g --results-dir $OUT/$2 > $OUT/$2/logs/prism_${4}_n${3}_s42.log 2>&1 &
    PG[$!]=$g; echo "[$(date +%H:%M)] GPU $g: $2 $4 N=$3"
    sleep 60    # let the job allocate before re-reading free memory
  done
  wait
done
echo "ALL DONE $(date): $(ls $OUT/*/prism/*/*.json 2>/dev/null | wc -l)/276"
