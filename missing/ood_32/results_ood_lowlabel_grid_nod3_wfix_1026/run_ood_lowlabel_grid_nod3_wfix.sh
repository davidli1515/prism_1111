#!/bin/bash
# No-D3 PRISM with fixed per-sample unlabeled weight (train_ood_lowlabel_nod3_wfix.py,
# --unlab-weight-ratio 0.1 => lambda_unlab = 0.1 * n_unl / n_lab; ~0.32/0.17/0.11 at src50/80/100).
# Fixes the src80/src100 PRISM collapse of results_ood_lowlabel_grid_nod3_1025 (lambda_unlab=1).
# Same lambdas / norm / extra40k / window split / seed 42 as nod3_1025 otherwise. prism only
# (V2 / Concorde have no unlabeled term; reuse nod3_1025 / grid_1008).
# Defaults: RATIOS="0.50:src50 0.80:src80 1.00:src100", SHOTS="32"  (23 targets -> 69 runs).
# Full grid: SHOTS="32 256 1024 2048" (276 runs). Never overwrites results (trainer refuses too).
# TARGETS="a b ..." limits the targets (default: all 23). LAUNCH_GAP (s, default 30) between launches.
# Usage: GPUS="0 1 2 3" nohup bash run_ood_lowlabel_grid_nod3_wfix.sh > run_ood_lowlabel_grid_nod3_wfix.log 2>&1 &
cd /scratch/gem5-TAO/micro_exp/exp2
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
OUT=/scratch/gem5-TAO/micro_exp/exp2/results_ood_lowlabel_grid_nod3_wfix
LAM="--lam-task 10 --lam-const 1 --lam-mono 0.01 --norm subset --extra40k --unlab-weight-ratio 0.1"
FREE_MB=${FREE_MB:-15000}; FREE_MB_V2=${FREE_MB_V2:-8000}
GPUS=(${GPUS:-0 1 2 3}); MAXPER=${MAXPER:-7}
ALL=${TARGETS:-$(ls /scratch2/wl730/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/\.npy$//')}
RATIOS=${RATIOS:-"0.50:src50 0.80:src80 1.00:src100"}; SHOTS=${SHOTS:-"32"}
for r in $RATIOS; do mkdir -p $OUT/${r##*:}/logs $OUT/${r##*:}/logs_failed; done
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1; }
ours_on() { pgrep -u wl730 -f "^python train_ood_lowlabel_nod3_wfix.py .* --gpu $1 " | wc -l; }   # our jobs on GPU $1 (counted at launch, no nvidia-smi lag)
running() { pgrep -u wl730 -f "^python train_ood_lowlabel_nod3_wfix.py .*--src-ratio $1 --target $2 --n-shot $3 --variant $4 " >/dev/null; }

JOBS=()
for r in $RATIOS; do for n in $SHOTS; do for t in $ALL; do
  JOBS+=("${r%%:*} ${r##*:} prism $n $t"); done; done; done

for round in 1 2 3 4; do
  declare -A SEEN=(); TODO=()
  for j in "${JOBS[@]}"; do
    set -- $j; [ -n "${SEEN[$j]}" ] && continue; SEEN[$j]=1
    [ -f $OUT/$2/$3/$5/n$4_s42.json ] && continue
    running $1 $5 $4 $3 && continue
    TODO+=("$j")
  done
  unset SEEN
  echo "[$(date)] round $round: ${#TODO[@]} to launch"
  [ ${#TODO[@]} -eq 0 ] && break
  for r in $RATIOS; do f=$(grep -l Traceback $OUT/${r##*:}/logs/*.log 2>/dev/null); [ -n "$f" ] && mv $f $OUT/${r##*:}/logs_failed/; done
  for j in "${TODO[@]}"; do
    set -- $j
    need=$FREE_MB; [ "$3" = "supervised" ] && need=$FREE_MB_V2
    while :; do
      g=""; best=999   # least-loaded eligible GPU (spread jobs instead of filling GPU 0 first)
      for c in "${GPUS[@]}"; do k=$(ours_on $c)
        [ $(free_mb $c) -ge $need ] && [ $k -lt $MAXPER ] && [ $k -lt $best ] && { g=$c; best=$k; }; done
      [ -n "$g" ] && break; sleep 30
    done
    python train_ood_lowlabel_nod3_wfix.py $LAM --src-ratio $1 --target $5 --n-shot $4 --variant $3 --seed 42 \
       --gpu $g --results-dir $OUT/$2 > $OUT/$2/logs/${3}_${5}_n${4}_s42.log 2>&1 &
    echo "[$(date +%H:%M)] GPU $g: $2 $3 $5 N=$4"; sleep ${LAUNCH_GAP:-30}
  done
  while pgrep -u wl730 -f "^python train_ood_lowlabel_nod3_wfix.py" >/dev/null; do sleep 60; done
done
n=$(ls $OUT/src*/*/*/n*_s42.json 2>/dev/null | wc -l)
echo "ALL DONE $(date): $n json"
