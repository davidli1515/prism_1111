#!/bin/bash
# run_exp4_all_key.sh — Key experiments: full + concorde × 18k + 40k
#
# 8-way parallel: 4 GPUs × 2 runs each (model is tiny, ~2GB per run)
# 40 total runs, 5 waves ≈ 8-9 hours
#
# Usage:
#   cd /path/to/gem5/micro_exp/exp4
#   bash run_exp4_all_key.sh

set -e

mkdir -p logs_18k logs_40k

R() {
    # R <dataset> <ablation> <ratio> <gpu>
    local DS=$1 ABL=$2 RATIO=$3 GPU=$4
    python train_exp4_all.py --dataset $DS --ablation $ABL --ratio $RATIO --gpu $GPU \
        > "logs_${DS}/${ABL}_${RATIO}.log" 2>&1 &
}

echo "============================================"
echo "Exp4 Key: full + concorde × 18k + 40k"
echo "  4 GPUs × 2 runs = 8 parallel per wave"
echo "  5 waves total"
echo "============================================"

# Wave 1: full 5-20%
echo ">>> Wave 1: full 5-20% (18k + 40k)"
R 18k full 0.05 0;  R 18k full 0.10 1;  R 18k full 0.15 2;  R 18k full 0.20 3
R 40k full 0.05 0;  R 40k full 0.10 1;  R 40k full 0.15 2;  R 40k full 0.20 3
wait; echo "  Wave 1 done."

# Wave 2: full 25-50%
echo ">>> Wave 2: full 25-50% (18k + 40k)"
R 18k full 0.25 0;  R 18k full 0.30 1;  R 18k full 0.40 2;  R 18k full 0.50 3
R 40k full 0.25 0;  R 40k full 0.30 1;  R 40k full 0.40 2;  R 40k full 0.50 3
wait; echo "  Wave 2 done."

# Wave 3: concorde 5-20%
echo ">>> Wave 3: concorde 5-20% (18k + 40k)"
R 18k concorde 0.05 0;  R 18k concorde 0.10 1;  R 18k concorde 0.15 2;  R 18k concorde 0.20 3
R 40k concorde 0.05 0;  R 40k concorde 0.10 1;  R 40k concorde 0.15 2;  R 40k concorde 0.20 3
wait; echo "  Wave 3 done."

# Wave 4: concorde 25-50%
echo ">>> Wave 4: concorde 25-50% (18k + 40k)"
R 18k concorde 0.25 0;  R 18k concorde 0.30 1;  R 18k concorde 0.40 2;  R 18k concorde 0.50 3
R 40k concorde 0.25 0;  R 40k concorde 0.30 1;  R 40k concorde 0.40 2;  R 40k concorde 0.50 3
wait; echo "  Wave 4 done."

# Wave 5: concorde 60-90%
echo ">>> Wave 5: concorde 60-90% (18k + 40k)"
R 18k concorde 0.60 0;  R 18k concorde 0.70 1;  R 18k concorde 0.80 2;  R 18k concorde 0.90 3
R 40k concorde 0.60 0;  R 40k concorde 0.70 1;  R 40k concorde 0.80 2;  R 40k concorde 0.90 3
wait; echo "  Wave 5 done."

echo ""
echo "============================================"
echo "All key runs done!"
echo ""
echo "  python summary_exp4.py --dataset 18k"
echo "  python summary_exp4.py --dataset 40k"
echo "  python plot_concord_gem.py --dataset 18k"
echo "  python plot_concord_gem.py --dataset 40k"
echo "============================================"