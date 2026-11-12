#!/bin/bash
# Exp5: Pretrain+FT at different label ratios
# Usage: nohup bash run_exp5.sh &> exp5.log &
 
cd /scratch/gem5-TAO/micro_exp/exp2
 
CKPT="/scratch/gem5-TAO/micro_exp/exp2/results_pretrain/pretrained.pt"
OUT="/scratch/gem5-TAO/micro_exp/exp5"
 
echo "Start: $(date)"
 
# Round 1
python train_exp5_pretrain_ft.py --gpu 0 --ratio 0.05 --pretrain_ckpt $CKPT --output_dir $OUT &
python train_exp5_pretrain_ft.py --gpu 1 --ratio 0.10 --pretrain_ckpt $CKPT --output_dir $OUT &
python train_exp5_pretrain_ft.py --gpu 2 --ratio 0.15 --pretrain_ckpt $CKPT --output_dir $OUT &
wait
echo "Round 1 (5%, 10%, 15%) done: $(date)"
 
# Round 2
python train_exp5_pretrain_ft.py --gpu 0 --ratio 0.20 --pretrain_ckpt $CKPT --output_dir $OUT &
python train_exp5_pretrain_ft.py --gpu 1 --ratio 0.30 --pretrain_ckpt $CKPT --output_dir $OUT &
python train_exp5_pretrain_ft.py --gpu 2 --ratio 0.50 --pretrain_ckpt $CKPT --output_dir $OUT &
wait
echo "Round 2 (20%, 30%, 50%) done: $(date)"
 
echo "All done: $(date)"
