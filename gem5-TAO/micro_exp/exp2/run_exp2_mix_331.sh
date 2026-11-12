cd /scratch/gem5-TAO/micro_exp/exp2

# 0% — 4 jobs
python train_exp2_mix_331.py --gpu 0 --ratio 0.0  --target 508.namd_r &
python train_exp2_mix_331.py --gpu 0 --ratio 0.0  --target 602.gcc_s &
python train_exp2_mix_331.py --gpu 1 --ratio 0.0  --target 526.blender_r &
python train_exp2_mix_331.py --gpu 1 --ratio 0.0  --target 600.perlbench_s &
# 5% — 4 jobs
python train_exp2_mix_331.py --gpu 2 --ratio 0.05 --target 508.namd_r &
python train_exp2_mix_331.py --gpu 2 --ratio 0.05 --target 602.gcc_s &
python train_exp2_mix_331.py --gpu 0 --ratio 0.05 --target 526.blender_r &
python train_exp2_mix_331.py --gpu 1 --ratio 0.05 --target 600.perlbench_s &
wait
