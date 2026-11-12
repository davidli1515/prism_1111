
python train_exp1_512_256.py --gpu 0 --variant gem_full &
python train_exp1_512_256.py --gpu 1 --variant gem_const &
python train_exp1_512_256.py --gpu 2 --variant concorde &
wait