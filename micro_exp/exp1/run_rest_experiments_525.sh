cd /path/to/gem5/micro_exp/exp1

# 确认OUTPUT_DIR
grep OUTPUT_DIR config.py
# 应该是 ./results/exp1_v2_run1

# [256,128] 408k
python train_exp1_256_128.py --gpu 0 --variant gem_const &
python train_exp1_256_128.py --gpu 0 --variant gem_vanilla &

# [512,256] 408k
python train_exp1_512_256.py --gpu 1 --variant gem_const &
python train_exp1_512_256.py --gpu 1 --variant gem_vanilla &

wait
# [256,128] 183k
python train_exp1_256_128.py --gpu 2 --variant gem_const --small &
python train_exp1_256_128.py --gpu 2 --variant gem_vanilla --small &

# [512,256] 183k
python train_exp1_512_256.py --gpu 3 --variant concorde --small &
python train_exp1_512_256.py --gpu 3 --variant gem_const --small &
python train_exp1_512_256.py --gpu 3 --variant gem_vanilla --small &
wait