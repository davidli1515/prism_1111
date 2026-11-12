#!/bin/bash
cd /scratch/gem5-TAO/micro_exp/exp4
mkdir -p logs

# (e) const_only — wave 1
python train_exp4_all.py --ablation const_only --ratio 0.05 --gpu 0 > logs/const_05.log 2>&1 &
python train_exp4_all.py --ablation const_only --ratio 0.10 --gpu 1 > logs/const_10.log 2>&1 &
python train_exp4_all.py --ablation const_only --ratio 0.15 --gpu 2 > logs/const_15.log 2>&1 &
wait

# (e) const_only — wave 2
python train_exp4_all.py --ablation const_only --ratio 0.20 --gpu 0 > logs/const_20.log 2>&1 &
python train_exp4_all.py --ablation const_only --ratio 0.25 --gpu 1 > logs/const_25.log 2>&1 &
python train_exp4_all.py --ablation const_only --ratio 0.30 --gpu 2 > logs/const_30.log 2>&1 &
wait

# (e) const_only — wave 3
python train_exp4_all.py --ablation const_only --ratio 0.40 --gpu 0 > logs/const_40.log 2>&1 &
python train_exp4_all.py --ablation const_only --ratio 0.50 --gpu 1 > logs/const_50.log 2>&1 &
# (f) mono_only — start packing
python train_exp4_all.py --ablation mono_only --ratio 0.05 --gpu 2 > logs/mono_05.log 2>&1 &
wait

# (f) mono_only — wave 4
python train_exp4_all.py --ablation mono_only --ratio 0.10 --gpu 0 > logs/mono_10.log 2>&1 &
python train_exp4_all.py --ablation mono_only --ratio 0.15 --gpu 1 > logs/mono_15.log 2>&1 &
python train_exp4_all.py --ablation mono_only --ratio 0.20 --gpu 2 > logs/mono_20.log 2>&1 &
wait

# (f) mono_only — wave 5
python train_exp4_all.py --ablation mono_only --ratio 0.25 --gpu 0 > logs/mono_25.log 2>&1 &
python train_exp4_all.py --ablation mono_only --ratio 0.30 --gpu 1 > logs/mono_30.log 2>&1 &
python train_exp4_all.py --ablation mono_only --ratio 0.40 --gpu 2 > logs/mono_40.log 2>&1 &
wait

# (f) mono_only — wave 6
python train_exp4_all.py --ablation mono_only --ratio 0.50 --gpu 0 > logs/mono_50.log 2>&1 &
wait

echo "Done. Run: python plot_exp4.py"
