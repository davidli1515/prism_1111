#!/bin/bash
# Exp2-Mix: OOD semi-supervised, GEM-Mix only
# Usage: nohup bash run_exp2_mix_test.sh &> exp2_mix_test.log &

cd /scratch/gem5-TAO/micro_exp/exp2

BMS=(508.namd_r 526.blender_r 600.perlbench_s 602.gcc_s)

echo "Start: $(date)"

python train_exp2_mix.py --gpu 0 --ratio 0.05 --target "${BMS[0]}" &
python train_exp2_mix.py --gpu 1 --ratio 0.05 --target "${BMS[1]}" &
python train_exp2_mix.py --gpu 2 --ratio 0.05 --target "${BMS[2]}" &
python train_exp2_mix.py --gpu 0 --ratio 0.05 --target "${BMS[3]}" &

wait

echo "End: $(date)"
echo "All done!"
