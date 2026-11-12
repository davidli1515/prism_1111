cd /scratch/gem5-TAO/micro_exp/exp1

# 183k + [256,128]
python train_exp1_256_128.py --gpu 0 --variant gem_full --small &
python train_exp1_256_128.py --gpu 1 --variant gem_const --small &
python train_exp1_256_128.py --gpu 2 --variant concorde --small &
wait