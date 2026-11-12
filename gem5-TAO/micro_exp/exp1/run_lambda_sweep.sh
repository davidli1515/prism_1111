#!/bin/bash
# Sweep LAMBDA_MONO: 6 values on 4 GPUs
# Usage: nohup bash run_lambda_sweep.sh &> lambda_sweep.log &

cd /scratch/gem5-TAO/micro_exp/exp1

echo "Start: $(date)"
# Batch 1: 4 values
python train_exp1_lambda_sweep.py --gpu 0 --lambda_mono 0.0 &
python train_exp1_lambda_sweep.py --gpu 1 --lambda_mono 0.01 &
python train_exp1_lambda_sweep.py --gpu 2 --lambda_mono 0.1 &
python train_exp1_lambda_sweep.py --gpu 3 --lambda_mono 0.5 &
wait

# Batch 2: 2 values
python train_exp1_lambda_sweep.py --gpu 0 --lambda_mono 1.0 &
python train_exp1_lambda_sweep.py --gpu 1 --lambda_mono 5.0 &
wait

echo "End: $(date)"
echo "Summary:"
python -c "
import json, os
sweep_dir = 'results/exp1/lambda_sweep'
print(f\"{'λ_mono':>10} {'MAPE':>10}\")
print('-'*22)
for lam in [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]:
    p = os.path.join(sweep_dir, f'lambda_mono_{lam}.json')
    if os.path.exists(p):
        r = json.load(open(p))
        marker = ' ←' if lam == 0.0 else ''
        print(f\"{lam:>10} {r['overall_mape']:>9.2f}%{marker}\")
"
echo "Done!"
