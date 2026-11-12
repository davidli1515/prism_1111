#!/bin/bash
# Wait for running fill jobs, then refill missing results (up to 3 rounds), moving OOM logs aside.
cd /scratch/gem5-TAO/micro_exp/exp2
D=results_ood_lowlabel_grid_1008
for round in 1 2 3; do
  while pgrep -u wl730 -f "^python train_ood_lowlabel_extra.py" >/dev/null; do sleep 60; done
  n=$(ls $D/src5/*/*/*.json $D/src50/*/*/*.json $D/src80/*/*/*.json 2>/dev/null | wc -l)
  echo "[$(date)] round $round start: $n/690"
  [ $n -ge 690 ] && break
  for r in src5 src50 src80; do mkdir -p $D/$r/logs_failed3; f=$(grep -l Traceback $D/$r/logs/*.log 2>/dev/null); [ -n "$f" ] && mv $f $D/$r/logs_failed3/; done
  bash run_ood_lowlabel_grid_1008_fill.sh > run_ood_lowlabel_grid_1008_fill_r$round.log 2>&1
done
echo "DRIVER DONE $(date): $(ls $D/src5/*/*/*.json $D/src50/*/*/*.json $D/src80/*/*/*.json | wc -l)/690"
