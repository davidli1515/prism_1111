#!/bin/bash
cd /scratch/gem5-TAO/micro_exp/exp2

CKPT="/scratch/gem5-TAO/micro_exp/exp2/results_pretrain/pretrained.pt"
OUT="/scratch/gem5-TAO/micro_exp/exp5"

python train_exp5_pretrain_ft.py --gpu 0 --ratio 0.20 --pretrain_ckpt $CKPT --output_dir $OUT &
python train_exp5_pretrain_ft.py --gpu 1 --ratio 0.30 --pretrain_ckpt $CKPT --output_dir $OUT &
python train_exp5_pretrain_ft.py --gpu 2 --ratio 0.40 --pretrain_ckpt $CKPT --output_dir $OUT &
wait
python train_exp5_pretrain_ft.py --gpu 0 --ratio 0.50 --pretrain_ckpt $CKPT --output_dir $OUT &
wait
echo "Done"