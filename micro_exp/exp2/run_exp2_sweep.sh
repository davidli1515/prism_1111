#!/bin/bash
# Run Exp2 Sweep: MAPE vs target train ratio, 4 GPUs parallel
# Usage: nohup bash run_exp2_sweep.sh &> exp2_sweep.log &

cd /path/to/gem5/micro_exp/exp2
mkdir -p results_sweep

NUM_GPUS=4
BMS=($(ls /data/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/.npy//'))
echo "Found ${#BMS[@]} benchmarks, using $NUM_GPUS GPUs"
echo "Ratios: 5%, 10%, 15%, 20%, 30%, 40%"
echo "Start: $(date)"
echo "======================================"

for ((i=0; i<${#BMS[@]}; i+=NUM_GPUS)); do
    PIDS=()
    for j in $(seq 0 $((NUM_GPUS-1))); do
        idx=$((i+j))
        [ $idx -ge ${#BMS[@]} ] && continue
        BM=${BMS[$idx]}
        echo "[$(date +%H:%M)] Launching $BM on GPU $j"
        python train_exp2_sweep.py --gpu $j --target "$BM" &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
    echo "[$(date +%H:%M)] Batch done"
done

echo "======================================"
echo "End: $(date)"
echo "Plotting..."
python plot_exp2_sweep.py --dir results_sweep/
echo "All done!"