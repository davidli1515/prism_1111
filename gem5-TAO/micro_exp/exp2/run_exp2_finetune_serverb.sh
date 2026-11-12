#!/bin/bash
# Server B: 17*6*3 = 306 jobs, 34 parallel (17 per batch)
# Usage: nohup bash run_exp2_finetune_serverB.sh &> exp2_finetune_serverB.log &

cd /scratch/gem5-TAO/micro_exp/exp2

DATA_DIR="/scratch2/wl730/micro_traindata_packed_all/gem"
BMS=($(ls ${DATA_DIR}/*.npy | xargs -n1 basename | sed 's/.npy//'))
SHOTS=(32 256 512 1024 2048 4096)
GPUS=(0 1 2)
MAX_JOBS=34

declare -A PID_GPU

launch_one() {
    local cmd=$1
    local min_count=999
    local best_gpu=${GPUS[0]}
    for g in "${GPUS[@]}"; do
        local count=0
        for pid in "${!PID_GPU[@]}"; do
            [ "${PID_GPU[$pid]}" = "$g" ] && ((count++))
        done
        if [ $count -lt $min_count ]; then
            min_count=$count
            best_gpu=$g
        fi
    done
    echo "[$(date +%H:%M)] GPU $best_gpu: $cmd"
    eval "$cmd --gpu $best_gpu" &
    PID_GPU[$!]=$best_gpu
}

wait_for_slot() {
    while [ ${#PID_GPU[@]} -ge $MAX_JOBS ]; do
        wait -n 2>/dev/null || true
        for pid in "${!PID_GPU[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                unset PID_GPU[$pid]
            fi
        done
    done
}

drain_all() {
    while [ ${#PID_GPU[@]} -gt 0 ]; do
        wait -n 2>/dev/null || true
        for pid in "${!PID_GPU[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                unset PID_GPU[$pid]
            fi
        done
    done
}

echo "Start: $(date)"
echo "Benchmarks: ${#BMS[@]} | Shots: ${SHOTS[*]} | GPUs: ${GPUS[*]} | Max: $MAX_JOBS"
echo "Total jobs: $(( ${#BMS[@]} * ${#SHOTS[@]} * 3 ))"

for N in "${SHOTS[@]}"; do
    echo ""
    echo "==================== N=$N ===================="

    # --- Supervised (GEM + Concorde) ---
    echo "--- N=$N: Supervised ---"
    for bm in "${BMS[@]}"; do
        wait_for_slot
        launch_one "python train_exp2_finetune_serverB.py --target $bm --n_samples $N"
    done
    drain_all
    echo "N=$N supervised done: $(date)"

    # --- Mix (GEM semi-supervised) ---
    echo "--- N=$N: Mix ---"
    for bm in "${BMS[@]}"; do
        wait_for_slot
        launch_one "python train_exp2_finetune_mix_serverB.py --target $bm --n_samples $N"
    done
    drain_all
    echo "N=$N mix done: $(date)"
done

echo ""
echo "End: $(date)"
echo "All done!"
