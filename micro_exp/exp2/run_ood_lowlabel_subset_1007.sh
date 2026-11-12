#!/bin/bash
# Low-label OOD, V3 only, exp4 loss formula (--norm subset) + BN fix. Skips finished runs.
cd /path/to/gem5/micro_exp/exp2
i=0; G=("0 1" "1 2")
for r in 0.05 0.20; do
  tag=src$(printf '%.0f' $(echo "$r*100" | bc))
  RESULTS_DIR=/path/to/gem5/micro_exp/exp2/results_ood_lowlabel_subset/$tag \
  LAM_ARGS="--lam-task 10 --lam-const 1 --lam-mono 0.01 --lam-unlab 1 --norm subset --src-ratio $r" \
  VARIANTS="prism" GPUS="${G[$i]}" MAX_JOBS=8 bash run_ood_lowlabel.sh > run_ood_lowlabel_subset_1007_$tag.log 2>&1 &
  i=$((i+1))
done
wait
echo "ALL DONE $(date)"
