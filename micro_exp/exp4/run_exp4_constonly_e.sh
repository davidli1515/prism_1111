#!/bin/bash
python train_exp4_all.py --ablation const_only --ratio 0.05 --gpu 0 > logs_v2/const_05.log 2>&1 &
python train_exp4_all.py --ablation const_only --ratio 0.10 --gpu 1 > logs_v2/const_10.log 2>&1 &
python train_exp4_all.py --ablation const_only --ratio 0.15 --gpu 2 > logs_v2/const_15.log 2>&1 &
wait
python train_exp4_all.py --ablation const_only --ratio 0.20 --gpu 0 > logs_v2/const_20.log 2>&1 &
python train_exp4_all.py --ablation const_only --ratio 0.25 --gpu 1 > logs_v2/const_25.log 2>&1 &
python train_exp4_all.py --ablation const_only --ratio 0.30 --gpu 2 > logs_v2/const_30.log 2>&1 &
wait
python train_exp4_all.py --ablation const_only --ratio 0.40 --gpu 0 > logs_v2/const_40.log 2>&1 &
python train_exp4_all.py --ablation const_only --ratio 0.50 --gpu 1 > logs_v2/const_50.log 2>&1 &
wait
