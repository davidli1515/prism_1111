#!/bin/bash
# run_exp4_vanilla.sh — GEM vanilla (L_task only) for 18k and 40k
#
# 2 datasets × 8 ratios = 16 runs, 8 parallel (4 GPUs × 2), 2 waves
#
# Usage:
#   cd /scratch/gem5-TAO/micro_exp/exp4
#   bash run_exp4_vanilla.sh

set -e

mkdir -p logs_18k logs_40k

R() {
    local DS=$1 RATIO=$2 GPU=$3
    python train_exp4_vanilla.py --dataset $DS --ratio $RATIO --gpu $GPU \
        > "logs_${DS}/vanilla_${RATIO}.log" 2>&1 &
}

echo "============================================"
echo "Exp4 Vanilla: L_task only × 18k + 40k"
echo "  8 parallel, 2 waves"
echo "============================================"

# Wave 1: 5-20%
echo ">>> Wave 1: 5-20%"
R 18k 0.05 0;  R 18k 0.10 1;  R 18k 0.15 2;  R 18k 0.20 3
R 40k 0.05 0;  R 40k 0.10 1;  R 40k 0.15 2;  R 40k 0.20 3
wait; echo "  done."

# Wave 2: 25-50%
echo ">>> Wave 2: 25-50%"
R 18k 0.25 0;  R 18k 0.30 1;  R 18k 0.40 2;  R 18k 0.50 3
R 40k 0.25 0;  R 40k 0.30 1;  R 40k 0.40 2;  R 40k 0.50 3
wait; echo "  done."

echo ""
echo "============================================"
echo "All vanilla runs done!"
echo "  Results: results/exp4_18k_vanilla/"
echo "           results/exp4_40k_vanilla/"
echo "  python plot_concord_gem.py --dataset 18k"
echo "  python plot_concord_gem.py --dataset 40k"
echo "============================================"
