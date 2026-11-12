#!/bin/bash
# Type 2: Pretrain (Stage 1) then FT (Stage 2), 18k
# Usage: nohup bash run_exp5_pt_ft_v2.sh &> exp5_pt_ft_v2.log &

cd /path/to/gem5/micro_exp/exp2

echo "Start: $(date)"

# Stage 1: Pretrain (single GPU)
echo "===== PRETRAIN ====="
python train_exp5_pretrain_v2.py --gpu 0
echo "Pretrain done: $(date)"

# Stage 2: FT (parallel)
echo "===== FINETUNE ====="
for R in 0.05 0.10 0.15 0.20 0.25 0.30 0.40; do
    GPU=$(( RANDOM % 4 ))
    python train_exp5_pt_ft_v2.py --gpu $GPU --ratio $R &
done
wait
echo "Round 1 done: $(date)"

for R in 0.50 0.60 0.70 0.80 0.90 1.00; do
    GPU=$(( RANDOM % 4 ))
    python train_exp5_pt_ft_v2.py --gpu $GPU --ratio $R &
done
wait
echo "All done: $(date)"