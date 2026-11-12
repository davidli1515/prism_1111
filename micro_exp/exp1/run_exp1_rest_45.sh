cd /path/to/gem5/micro_exp/exp1

python train_exp1_512_256.py --gpu 0 --variant concorde --small &
python train_exp1_512_256.py --gpu 0 --variant gem_const --small &
python train_exp1_512_256.py --gpu 1 --variant gem_full --small &
python train_exp1_512_256.py --gpu 1 --variant gem_nod3 --small &
