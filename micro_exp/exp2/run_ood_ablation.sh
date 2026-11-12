#!/bin/bash
# Run OOD ablation: all benchmarks simultaneously on GPUs 0,1,2
# Usage: nohup bash run_ood_ablation.sh &> ood_ablation.log &

cd /path/to/gem5/micro_exp/exp2
mkdir -p results_ood_ablation

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
BMS=($(ls /data/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/.npy//'))

echo "Benchmarks: ${#BMS[@]}, GPUs: ${GPUS[*]}"
echo "Start: $(date)"

PIDS=()
for i in "${!BMS[@]}"; do
    GPU=${GPUS[$((i % NUM_GPUS))]}
    echo "  ${BMS[$i]} → GPU $GPU"
    python train_exp2_ood_ablation.py --gpu $GPU --target "${BMS[$i]}" \
        > "results_ood_ablation/log_${BMS[$i]}.txt" 2>&1 &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait $pid; done

echo "End: $(date)"
echo "Collecting results..."
python -c "
import json, glob, numpy as np, os
files = sorted(glob.glob('results_ood_ablation/ood_abl_*.json'))
results = [json.load(open(f)) for f in files]
print(f\"{'Benchmark':<20} {'Full':>8} {'Const':>8} {'Mono':>8} {'Concorde':>10}\")
print('-'*60)
for r in sorted(results, key=lambda x: x['target']):
    c = f\"{r['concorde']:>9.2f}%\" if 'concorde' in r else '    ...   '
    print(f\"{r['target']:<20} {r['full']:>7.2f}% {r['const_only']:>7.2f}% {r['mono_only']:>7.2f}% {c}\")
print('-'*60)
done = [r for r in results if 'concorde' in r]
all_r = results
print(f\"{'Average':<20} {np.mean([r['full'] for r in all_r]):>7.2f}% {np.mean([r['const_only'] for r in all_r]):>7.2f}% {np.mean([r['mono_only'] for r in all_r]):>7.2f}%\", end='')
if done: print(f\" {np.mean([r['concorde'] for r in done]):>9.2f}%\")
else: print()
print(f\"{len(files)} benchmarks, {len(done)} with Concorde done\")
"
echo "All done!"