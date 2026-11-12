#!/bin/bash
# ============================================================
# Master script: Run all experiments
# Change RESULT_DIR below to control where results are saved
# ============================================================
RESULT_DIR="./results/exp1_v2_74"
# ============================================================

cd /scratch/gem5-TAO/micro_exp/exp1

# Update config.py OUTPUT_DIR
sed -i "s|^OUTPUT_DIR.*|OUTPUT_DIR = \"${RESULT_DIR}\"|" config.py
echo "Results will be saved to: ${RESULT_DIR}"
echo "Start: $(date)"

# ==================== Wave 1: [256,128] 408k + 183k ====================
echo ""
echo "===== Wave 1: [256,128] ====="
python train_exp1_256_128.py --gpu 0 --variant gem_full &
python train_exp1_256_128.py --gpu 0 --variant gem_const &
python train_exp1_256_128.py --gpu 1 --variant gem_vanilla &
python train_exp1_256_128.py --gpu 1 --variant gem_nod3 &
python train_exp1_256_128.py --gpu 2 --variant concorde &
python train_exp1_256_128.py --gpu 2 --variant gem_full --small &
python train_exp1_256_128.py --gpu 3 --variant gem_const --small &
python train_exp1_256_128.py --gpu 3 --variant gem_vanilla --small &
python train_exp1_256_128.py --gpu 0 --variant gem_nod3 --small &
python train_exp1_256_128.py --gpu 1 --variant concorde --small &

wait
echo "===== Wave 1 done: $(date) ====="

# ==================== Wave 2: [512,256] 408k + 183k ====================
echo ""
echo "===== Wave 2: [512,256] ====="
python train_exp1_512_256.py --gpu 0 --variant gem_full &
python train_exp1_512_256.py --gpu 0 --variant gem_const &
python train_exp1_512_256.py --gpu 1 --variant gem_vanilla &
python train_exp1_512_256.py --gpu 1 --variant gem_nod3 &
python train_exp1_512_256.py --gpu 2 --variant concorde &
python train_exp1_512_256.py --gpu 2 --variant gem_full --small &
python train_exp1_512_256.py --gpu 3 --variant gem_const --small &
python train_exp1_512_256.py --gpu 3 --variant gem_vanilla --small &
python train_exp1_512_256.py --gpu 0 --variant gem_nod3 --small &
python train_exp1_512_256.py --gpu 1 --variant concorde --small &

wait
echo "===== Wave 2 done: $(date) ====="

wait
echo ""
echo "===== Exp1 done: $(date) ====="

# ==================== Plot ====================
python plot_exp1_all.py --base ${RESULT_DIR} --output ${RESULT_DIR}/exp1_all.png
echo "Plots saved to ${RESULT_DIR}"