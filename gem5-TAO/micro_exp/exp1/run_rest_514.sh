cd /scratch/gem5-TAO/micro_exp/exp1

# [256,128] 183k
python train_exp1_256_128.py --gpu 0 --variant gem_full --small &
python train_exp1_256_128.py --gpu 0 --variant gem_const --small &
python train_exp1_256_128.py --gpu 1 --variant gem_vanilla --small &
python train_exp1_256_128.py --gpu 1 --variant gem_nod3 --small &
python train_exp1_256_128.py --gpu 2 --variant concorde --small &

# [512,256] 183k
python train_exp1_512_256.py --gpu 2 --variant gem_full --small &
python train_exp1_512_256.py --gpu 3 --variant gem_const --small &
python train_exp1_512_256.py --gpu 3 --variant gem_vanilla --small &
python train_exp1_512_256.py --gpu 3 --variant gem_nod3 --small &
python train_exp1_512_256.py --gpu 0 --variant concorde --small &
