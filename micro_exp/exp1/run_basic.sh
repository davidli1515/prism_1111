cd /path/to/gem5/micro_exp/exp1
python train_exp1.py --gpu 0 --variant gem_full &
python train_exp1.py --gpu 1 --variant gem_nod3 &
python train_exp1.py --gpu 2 --variant concorde &
wait
