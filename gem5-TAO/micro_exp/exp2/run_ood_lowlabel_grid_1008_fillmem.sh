#!/bin/bash
# Memory-aware fill for results_ood_lowlabel_grid_1008 step 2 (src 5/50/80).
# Launch on a GPU only if it has >= FREE_MB free; skip jobs whose result exists or which are
# currently running. Up to 4 rounds; failed logs moved aside. Appends "DRIVER DONE" to the
# old driver log at the end so that run_after_fill_then_wfix.sh starts the weight-fix grid.
cd /scratch/gem5-TAO/micro_exp/exp2
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
OUT=/scratch/gem5-TAO/micro_exp/exp2/results_ood_lowlabel_grid_1008
LAM="--lam-task 10 --lam-const 1 --lam-mono 0.01 --lam-unlab 1 --norm subset --extra40k"
FREE_MB=${FREE_MB:-15000}; GPUS=(${GPUS:-1 0 3 2}); MAXPER=${MAXPER:-7}
ALL=$(ls /scratch2/wl730/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/\.npy$//')
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1; }
ours_on() { nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | \
  grep -c "$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $1)" ; }
running() { pgrep -u wl730 -f "^python train_ood_lowlabel_extra.py .*--src-ratio $1 --target $2 --n-shot $3 --variant $4 " >/dev/null; }
for round in 1 2 3 4; do
  JOBS=()
  for t in $ALL; do
    for n in 32 256 1024 2048; do JOBS+=("0.05 src5 prism $n $t"); done
    for v in concorde supervised; do JOBS+=("0.05 src5 $v 2048 $t"); done
    for r in 0.50:src50 0.80:src80; do for v in concorde supervised prism; do for n in 32 256 1024 2048; do
      JOBS+=("${r%%:*} ${r##*:} $v $n $t"); done; done; done
  done
  TODO=(); for j in "${JOBS[@]}"; do set -- $j; [ -f $OUT/$2/$3/$5/n$4_s42.json ] && continue; running $1 $5 $4 $3 && continue; TODO+=("$j"); done
  echo "[$(date)] round $round: ${#TODO[@]} to launch"
  for r in src5 src50 src80; do mkdir -p $OUT/$r/logs_failed3; f=$(grep -l Traceback $OUT/$r/logs/*.log 2>/dev/null); [ -n "$f" ] && mv $f $OUT/$r/logs_failed3/; done
  for j in "${TODO[@]}"; do
    set -- $j
    while :; do
      g=""; for c in "${GPUS[@]}"; do [ $(free_mb $c) -ge $FREE_MB ] && [ $(ours_on $c) -lt $MAXPER ] && { g=$c; break; }; done
      [ -n "$g" ] && break; sleep 30
    done
    python train_ood_lowlabel_extra.py $LAM --src-ratio $1 --target $5 --n-shot $4 --variant $3 --seed 42 \
       --gpu $g --results-dir $OUT/$2 > $OUT/$2/logs/${3}_${5}_n${4}_s42.log 2>&1 &
    echo "[$(date +%H:%M)] GPU $g: $2 $3 $5 N=$4"; sleep 60
  done
  while pgrep -u wl730 -f "^python train_ood_lowlabel_extra.py" >/dev/null; do sleep 60; done
  [ $(ls $OUT/src5/*/*/*.json $OUT/src50/*/*/*.json $OUT/src80/*/*/*.json | wc -l) -ge 690 ] && break
done
n=$(ls $OUT/src5/*/*/*.json $OUT/src50/*/*/*.json $OUT/src80/*/*/*.json | wc -l)
echo "ALL DONE $(date): $n/690"
echo "DRIVER DONE $(date): $n/690 (fillmem)" >> run_ood_lowlabel_grid_1008_fill_driver.log
