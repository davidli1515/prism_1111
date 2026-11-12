#!/bin/bash
# Low-label OOD: src ratio 5% and 20%, both launched concurrently on GPU 2.
cd /path/to/gem5/micro_exp/exp2
for r in 0.05 0.20; do
  tag=src$(printf '%.0f' $(echo "$r*100" | bc))
  RESULTS_DIR=/path/to/gem5/micro_exp/exp2/results_ood_lowlabel/$tag \
  LAM_ARGS="--lam-task 10 --lam-const 1 --lam-mono 0.01 --lam-unlab 1 --norm batch --src-ratio $r" \
  GPUS="2" MAX_JOBS=5 bash run_ood_lowlabel.sh > run_ood_lowlabel_1006_$tag.log 2>&1 &
done
wait
echo "ALL DONE $(date)"
