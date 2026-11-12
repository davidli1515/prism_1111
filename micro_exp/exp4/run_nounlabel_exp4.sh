#!/bin/bash
cd /path/to/gem5/micro_exp/exp4

# wave 1
python train_exp4_all.py --ablation supervised_only --ratio 0.05 --gpu 0 > logs/sup_05.log 2>&1 &
python train_exp4_all.py --ablation supervised_only --ratio 0.10 --gpu 1 > logs/sup_10.log 2>&1 &
python train_exp4_all.py --ablation supervised_only --ratio 0.15 --gpu 2 > logs/sup_15.log 2>&1 &
wait

# wave 2
python train_exp4_all.py --ablation supervised_only --ratio 0.20 --gpu 0 > logs/sup_20.log 2>&1 &
python train_exp4_all.py --ablation supervised_only --ratio 0.25 --gpu 1 > logs/sup_25.log 2>&1 &
python train_exp4_all.py --ablation supervised_only --ratio 0.30 --gpu 2 > logs/sup_30.log 2>&1 &
wait

# wave 3
python train_exp4_all.py --ablation supervised_only --ratio 0.40 --gpu 0 > logs/sup_40.log 2>&1 &
python train_exp4_all.py --ablation supervised_only --ratio 0.50 --gpu 1 > logs/sup_50.log 2>&1 &
wait

echo "Done."