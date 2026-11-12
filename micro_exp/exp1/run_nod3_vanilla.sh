#!/bin/bash
# 6 tasks: nod3 (183k) + vanilla (183k + 408k), [512,256] + [256,128]
# Usage: nohup bash run_nod3_vanilla.sh &> nod3_vanilla.log &

cd /path/to/gem5/micro_exp/exp1

echo "Starting 6 tasks: $(date)"
echo ""

# Task 1: 183k + [512,256] + gem_nod3
echo "[1/6] 183k [512,256] gem_nod3 -> GPU 0"
python train_exp1_512_256.py --gpu 0 --variant gem_nod3 --small &

# Task 2: 183k + [256,128] + gem_nod3
echo "[2/6] 183k [256,128] gem_nod3 -> GPU 0"
python train_exp1_256_128.py --gpu 0 --variant gem_nod3 --small &

# Task 3: 183k + [512,256] + gem_vanilla
echo "[3/6] 183k [512,256] gem_vanilla -> GPU 1"
python train_exp1_512_256.py --gpu 1 --variant gem_vanilla --small &

# Task 4: 183k + [256,128] + gem_vanilla
echo "[4/6] 183k [256,128] gem_vanilla -> GPU 1"
python train_exp1_256_128.py --gpu 1 --variant gem_vanilla --small &

# Task 5: 408k + [512,256] + gem_vanilla
echo "[5/6] 408k [512,256] gem_vanilla -> GPU 2"
python train_exp1_512_256.py --gpu 2 --variant gem_vanilla &

# Task 6: 408k + [256,128] + gem_vanilla
echo "[6/6] 408k [256,128] gem_vanilla -> GPU 2"
python train_exp1_256_128.py --gpu 2 --variant gem_vanilla &

wait
echo ""
echo "All done: $(date)"
