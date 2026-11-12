#!/bin/bash
# Pretrain + Semi-Supervised FT on 18k data
# Usage: nohup bash run_exp2_ft_18k_semi_528.sh &> exp2_ft_18k_semi_528.log &

cd /scratch/gem5-TAO/micro_exp/exp2
mkdir -p /scratch/gem5-TAO/micro_exp/exp5/results_pretrain_18k_semi

echo "Start: $(date)"

for R in 0.05 0.10 0.15 0.20 0.25 0.30 0.40; do
    GPU=$(( RANDOM % 4 ))
    python train_exp2_ft_18k_semi.py --gpu $GPU --ratio $R &
done
wait
echo "Round 1 done: $(date)"

for R in 0.50 0.60 0.70 0.80 0.90 1.00; do
    GPU=$(( RANDOM % 4 ))
    python train_exp2_ft_18k_semi.py --gpu $GPU --ratio $R &
done
wait
echo "All done: $(date)"