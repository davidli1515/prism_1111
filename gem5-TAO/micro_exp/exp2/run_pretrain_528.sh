#!/bin/bash
# Run all pretrain + FT on Server B
# Results: /scratch/gem5-TAO/micro_exp/exp2/results_pretrain_528/
# Usage: nohup bash run_pretrain_528.sh &> pretrain_528.log &

cd /scratch/gem5-TAO/micro_exp/exp2
DIR="/scratch/gem5-TAO/micro_exp/exp2/results_pretrain_528"
mkdir -p $DIR
mkdir -p /scratch/gem5-TAO/micro_exp/exp5/results_pretrain_18k

# ============================================================
# 18k Pretrain (~15 min)
# ============================================================
echo "=== 18k Pretrain: $(date) ==="
python train_exp2_pretrain_18k.py --gpu 0
echo "=== 18k Pretrain done: $(date) ==="

# 18k FT, 6 parallel per round
for R in 0.05 0.10 0.15 0.20 0.25 0.30; do
    GPU=$(( RANDOM % 3 ))
    python train_exp2_pretrain_ft_18k.py --gpu $GPU --ratio $R &
done
wait
for R in 0.40 0.50 0.60 0.70 0.80 0.90 1.00; do
    GPU=$(( RANDOM % 3 ))
    python train_exp2_pretrain_ft_18k.py --gpu $GPU --ratio $R &
done
wait
echo "=== 18k FT done: $(date) ==="

# ============================================================
# 40k Pretrain (~30 min)
# ============================================================
echo "=== 40k Pretrain: $(date) ==="
python train_exp2_pretrain_40k.py --gpu 0
echo "=== 40k Pretrain done: $(date) ==="

# 40k FT, 6 parallel per round
for R in 0.05 0.10 0.15 0.20 0.25 0.30; do
    GPU=$(( RANDOM % 3 ))
    python train_exp2_pretrain_ft_40k.py --gpu $GPU --ratio $R &
done
wait
for R in 0.40 0.50 0.60 0.70 0.80 0.90 1.00; do
    GPU=$(( RANDOM % 3 ))
    python train_exp2_pretrain_ft_40k.py --gpu $GPU --ratio $R &
done
wait
echo "=== All done: $(date) ==="
