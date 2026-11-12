#!/bin/bash
# Exp2 Foundation + Fine-tune pipeline
# Stage 1: Train 17 foundations (~2-3 hours with 12 parallel)
# Stage 2: FT at 1%, 5%, 10%, 20% (~20 min per ratio with 12 parallel)
#
# Usage: nohup bash run_exp2_foundation.sh &> exp2_foundation.log &

cd /scratch/gem5-TAO/micro_exp/exp2

ALL_BMS=($(ls /scratch2/wl730/micro_traindata_packed_all/gem/*.npy | xargs -n1 basename | sed 's/.npy//'))
MAX_JOBS=9
NUM_GPUS=3

echo "========================================="
echo "  Exp2 Foundation Pipeline (Stage 1 only)"
echo "  Benchmarks: ${#ALL_BMS[@]}"
echo "  Parallel: $MAX_JOBS jobs, $NUM_GPUS GPUs (3 per GPU)"
echo "  $(date)"
echo "========================================="

# ==================== Stage 1: Foundations ====================
echo ""
echo "===== STAGE 1: Training foundations (0% target) ====="
echo "$(date)"

for ((i=0; i<${#ALL_BMS[@]}; i+=MAX_JOBS)); do
    for ((j=0; j<MAX_JOBS; j++)); do
        idx=$((i+j))
        [ $idx -ge ${#ALL_BMS[@]} ] && continue
        GPU=$((j % NUM_GPUS))
        echo "  [S1] ${ALL_BMS[$idx]} -> GPU $GPU"
        python train_exp2_foundation.py --gpu $GPU --stage 1 --target "${ALL_BMS[$idx]}" &
    done
    wait
    echo "  [S1] Batch $((i/MAX_JOBS+1)) done: $(date)"
done
echo "===== Stage 1 done: $(date) ====="

echo ""
echo "========== ALL DONE: $(date) =========="
echo "Foundations saved to: foundations/"
echo ""

# List saved checkpoints
echo "--- Saved checkpoints ---"
for d in foundations/*/; do
    bm=$(basename "$d")
    gem="$d/gem_foundation.pt"
    con="$d/con_foundation.pt"
    [ -f "$gem" ] && [ -f "$con" ] && echo "  $bm: OK" || echo "  $bm: MISSING"
done
