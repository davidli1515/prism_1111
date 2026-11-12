#!/bin/bash
# No-D3 rerun of the low-label OOD grid (train_ood_lowlabel_nod3.py: V2 / PRISM use only the
# 3873 Concorde input features). Concorde is reused from results_ood_lowlabel_grid_1008 etc.
# Grid: src {5,20,50,80,100}% x N {32,256,1024,2048} x {supervised (V2), prism} x 23 targets
#       = 920 runs, seed 42, same lambdas / norm / extra40k as results_ood_lowlabel_grid_1008.
# Order: N=2048 at src20/50/80 first (the label-saving claim), then the rest.
# Memory-aware queue: launch on a GPU only if it has >= FREE_MB (prism) / FREE_MB_V2 (V2) free
# and fewer than MAXPER of our jobs. Skips finished and running jobs; up to 4 rounds; logs of
# failed runs are moved to logs_failed/. Never overwrites results (the trainer refuses too).
# Usage: GPUS="0 1 2 3" nohup bash run_ood_lowlabel_grid_nod3_1025.sh > run_ood_lowlabel_grid_nod3_1025.log 2>&1 &
cd /scratch/gem5-TAO/micro_exp/exp2
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
OUT=/scratch/gem5-TAO/micro_exp/exp2/results_ood_lowlabel_grid_nod3_1025
LAM="--lam-task 10 --lam-const 1 --lam-mono 0.01 --lam-unlab 1 --norm subset --extra40k"
FREE_MB=${FREE_MB:-15000}; FREE_MB_V2=${FREE_MB_V2:-8000}
GPUS=(${GPUS:-0 1 2 3}); MAXPER=${MAXPER:-7}
ALL=$(ls /scratch2/wl730/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/\.npy$//')
RATIOS="0.05:src5 0.20:src20 0.50:src50 0.80:src80 1.00:src100"
for r in $RATIOS; do mkdir -p $OUT/${r##*:}/logs $OUT/${r##*:}/logs_failed; done
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1; }
ours_on() { pgrep -u wl730 -f "^python train_ood_lowlabel_nod3.py .* --gpu $1 " | wc -l; }   # our jobs on GPU $1 (counted at launch, no nvidia-smi lag)
running() { pgrep -u wl730 -f "^python train_ood_lowlabel_nod3.py .*--src-ratio $1 --target $2 --n-shot $3 --variant $4 " >/dev/null; }

JOBS=()
for r in 0.20:src20 0.50:src50 0.80:src80; do for v in prism supervised; do for t in $ALL; do
  JOBS+=("${r%%:*} ${r##*:} $v 2048 $t"); done; done; done
for r in $RATIOS; do for n in 32 256 1024 2048; do for v in prism supervised; do for t in $ALL; do
  JOBS+=("${r%%:*} ${r##*:} $v $n $t"); done; done; done; done     # duplicates of the first block are skipped below

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
    python train_ood_lowlabel_nod3.py $LAM --src-ratio $1 --target $5 --n-shot $4 --variant $3 --seed 42 \
       --gpu $g --results-dir $OUT/$2 > $OUT/$2/logs/${3}_${5}_n${4}_s42.log 2>&1 &
    echo "[$(date +%H:%M)] GPU $g: $2 $3 $5 N=$4"; sleep ${LAUNCH_GAP:-30}
  done
  while pgrep -u wl730 -f "^python train_ood_lowlabel_nod3.py" >/dev/null; do sleep 60; done
done
n=$(ls $OUT/src*/*/*/n*_s42.json 2>/dev/null | wc -l)
echo "ALL DONE $(date): $n/920"
