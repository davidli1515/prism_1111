#!/bin/bash
python train_exp4_all.py --ablation full --ratio 0.05 --gpu 0 > logs_v2/full_05.log 2>&1 &
python train_exp4_all.py --ablation full --ratio 0.10 --gpu 1 > logs_v2/full_10.log 2>&1 &
python train_exp4_all.py --ablation full --ratio 0.15 --gpu 2 > logs_v2/full_15.log 2>&1 &
wait
python train_exp4_all.py --ablation full --ratio 0.20 --gpu 0 > logs_v2/full_20.log 2>&1 &
python train_exp4_all.py --ablation full --ratio 0.25 --gpu 1 > logs_v2/full_25.log 2>&1 &
python train_exp4_all.py --ablation full --ratio 0.30 --gpu 2 > logs_v2/full_30.log 2>&1 &
wait
python train_exp4_all.py --ablation full --ratio 0.40 --gpu 0 > logs_v2/full_40.log 2>&1 &
python train_exp4_all.py --ablation full --ratio 0.50 --gpu 1 > logs_v2/full_50.log 2>&1 &
wait
