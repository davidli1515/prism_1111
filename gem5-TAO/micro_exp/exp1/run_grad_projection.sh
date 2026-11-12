#!/bin/bash
# Run gradient projection analysis: ID + OOD (all benchmarks)
# Usage: nohup bash run_grad_projection.sh &> grad_projection.log &

cd /scratch/gem5-TAO/micro_exp/exp2
mkdir -p results_grad_projection

echo "Start: $(date)"

# 1. ID (single run, 1 GPU)
echo "Running ID..."
python grad_projection.py --gpu 0 --mode id &

# 2. OOD (all benchmarks, parallel on remaining GPUs)
echo "Running OOD (all benchmarks)..."
GPUS=(1 2 3)
NUM_GPUS=${#GPUS[@]}
BMS=($(ls /scratch2/wl730/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/.npy//'))

PIDS=()
for i in "${!BMS[@]}"; do
    GPU=${GPUS[$((i % NUM_GPUS))]}
    python grad_projection.py --gpu $GPU --mode ood --target "${BMS[$i]}" &
    PIDS+=($!)
done

wait

# 3. Collect OOD averages
echo "Collecting OOD averages..."
python grad_projection.py --collect

# 4. Plot
echo "Plotting..."
python plot_grad_projection.py \
    --id results_grad_projection/proj_id.json \
    --ood results_grad_projection/proj_ood_avg.json \
    --out results_grad_projection/grad_projection_compare.png

echo "End: $(date)"
echo "Done!"
