#!/bin/bash
# run_exp4_ablation_all.sh — Ablation (e)(f)(h) + Concorde 60-90% for 18k & 40k
#
# 10-way parallel across 4 GPUs (GPU 0-2: 3 each, GPU 3: 1)
# 56 runs, 6 waves ≈ 5 hours
#
# Usage:
#   cd /path/to/gem5/micro_exp/exp4
#   bash run_exp4_ablation_all.sh

set -e

mkdir -p logs_18k logs_40k

R() {
    local DS=$1 ABL=$2 RATIO=$3 GPU=$4
    python train_exp4_all.py --dataset $DS --ablation $ABL --ratio $RATIO --gpu $GPU \
        > "logs_${DS}/${ABL}_${RATIO}.log" 2>&1 &
}

echo "============================================"
echo "Exp4 Ablation: (e)(f)(h) + Concorde 60-90%"
echo "  18k + 40k, 10 parallel, 6 waves"
echo "============================================"

# Wave 1: const_only 5-25% (10 runs)
echo ">>> Wave 1: const_only 5-25%"
R 18k const_only 0.05 0;  R 18k const_only 0.10 0;  R 18k const_only 0.15 1
R 18k const_only 0.20 1;  R 18k const_only 0.25 2
R 40k const_only 0.05 2;  R 40k const_only 0.10 3;  R 40k const_only 0.15 0
R 40k const_only 0.20 1;  R 40k const_only 0.25 3
wait; echo "  done."

# Wave 2: const_only 30-50% + mono_only 5-15% (10 runs)
echo ">>> Wave 2: const_only 30-50% + mono_only 5-15%"
R 18k const_only 0.30 0;  R 18k const_only 0.40 0;  R 18k const_only 0.50 1
R 40k const_only 0.30 1;  R 40k const_only 0.40 2;  R 40k const_only 0.50 2
R 18k mono_only  0.05 3;  R 18k mono_only  0.10 0;  R 18k mono_only  0.15 1
R 40k mono_only  0.05 3
wait; echo "  done."

# Wave 3: mono_only 20-50% (10 runs)
echo ">>> Wave 3: mono_only 20-50%"
R 18k mono_only 0.20 0;  R 18k mono_only 0.25 0;  R 18k mono_only 0.30 1
R 18k mono_only 0.40 1;  R 18k mono_only 0.50 2
R 40k mono_only 0.10 2;  R 40k mono_only 0.15 3;  R 40k mono_only 0.20 3
R 40k mono_only 0.25 0;  R 40k mono_only 0.30 1
wait; echo "  done."

# Wave 4: mono_only 40-50% + supervised_only 5-25% (10 runs)
echo ">>> Wave 4: mono_only rest + supervised_only 5-25%"
R 40k mono_only  0.40 0;  R 40k mono_only  0.50 0
R 18k supervised_only 0.05 1;  R 18k supervised_only 0.10 1
R 18k supervised_only 0.15 2;  R 18k supervised_only 0.20 2
R 18k supervised_only 0.25 3;  R 40k supervised_only 0.05 3
R 40k supervised_only 0.10 0;  R 40k supervised_only 0.15 1
wait; echo "  done."

# Wave 5: supervised_only 30-50% (10 runs)
echo ">>> Wave 5: supervised_only 30-50%"
R 18k supervised_only 0.30 0;  R 18k supervised_only 0.40 0
R 18k supervised_only 0.50 1;  R 40k supervised_only 0.20 1
R 40k supervised_only 0.25 2;  R 40k supervised_only 0.30 2
R 40k supervised_only 0.40 3;  R 40k supervised_only 0.50 3
R 18k concorde 0.60 0;  R 18k concorde 0.70 1
wait; echo "  done."

# # Wave 6: Concorde 60-90% (6 runs)
# echo ">>> Wave 6: Concorde 80-90% + remaining"
# R 18k concorde 0.80 0;  R 18k concorde 0.90 1
# R 40k concorde 0.60 2;  R 40k concorde 0.70 3
# R 40k concorde 0.80 0;  R 40k concorde 0.90 1
# wait; echo "  done."

echo ""
echo "============================================"
echo "All ablation runs done!"
echo ""
echo "  python summary_exp4.py --dataset 18k"
echo "  python summary_exp4.py --dataset 40k"
echo "  python plot_exp4.py --dataset 18k"
echo "  python plot_exp4.py --dataset 40k"
echo "============================================"
