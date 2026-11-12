#!/bin/bash
# Run gradient conflict analysis for all benchmarks, 4 GPUs parallel
# Usage: nohup bash run_grad_conflict_exp2.sh &> grad_conflict_exp2.log &

cd /scratch/gem5-TAO/micro_exp/exp2

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
BMS=($(ls /scratch2/wl730/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/.npy//'))

echo "Benchmarks: ${#BMS[@]}, GPUs: ${GPUS[*]}"
echo "Start: $(date)"

PIDS=()
for i in "${!BMS[@]}"; do
    GPU=${GPUS[$((i % NUM_GPUS))]}
    python grad_conflict_exp2.py --gpu $GPU --target "${BMS[$i]}" &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid; done

echo "Collecting & averaging..."
python grad_conflict_exp2.py --collect

echo "Plotting..."
python plot_grad_conflict_exp2.py --json results_grad_conflict/grad_conflict_avg.json

echo "End: $(date)"
echo "Done!"
