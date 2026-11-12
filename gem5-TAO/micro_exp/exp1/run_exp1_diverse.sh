cd /scratch/gem5-TAO/micro_exp/exp1

# 修改DEFAULT_DIVERSE_DIR或用--data_dir指定你的数据路径
DATA=/scratch2/wl730/micro_traindata_packed_diverse_all

# [512,256] 全部variant
python train_exp1_diverse.py --gpu 0 --variant gem_full    --data_dir $DATA &
python train_exp1_diverse.py --gpu 0 --variant gem_const   --data_dir $DATA &
python train_exp1_diverse.py --gpu 1 --variant gem_vanilla --data_dir $DATA &
python train_exp1_diverse.py --gpu 1 --variant concorde  --hidden 256,128  --data_dir $DATA &

# # [256,128]
# python train_exp1_diverse.py --gpu 2 --variant gem_full    --hidden 256,128 --data_dir $DATA &
# python train_exp1_diverse.py --gpu 2 --variant concorde    --hidden 256,128 --data_dir $DATA 
