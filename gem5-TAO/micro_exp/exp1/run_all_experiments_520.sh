#!/bin/bash
# ============================================================
# Master script: Run all experiments
# Change RESULT_DIR below to control where results are saved
# ============================================================
RESULT_DIR="./results/exp1_v2_520"
# ============================================================

cd /scratch/gem5-TAO/micro_exp/exp1

# Update config.py OUTPUT_DIR
sed -i "s|^OUTPUT_DIR.*|OUTPUT_DIR = \"${RESULT_DIR}\"|" config.py
echo "Results will be saved to: ${RESULT_DIR}"
echo "Start: $(date)"

# ==================== Exp1: [256,128] 408k ====================
echo ""
echo "===== [256,128] 408k ====="
python train_exp1_256_128.py --gpu 0 --variant gem_full &
python train_exp1_256_128.py --gpu 1 --variant gem_const &
python train_exp1_256_128.py --gpu 2 --variant gem_vanilla &
python train_exp1_256_128.py --gpu 3 --variant gem_nod3 &
python train_exp1_256_128.py --gpu 0 --variant concorde &

wait
# ==================== Exp1: [512,256] 408k ====================
echo "===== [512,256] 408k ====="
python train_exp1_512_256.py --gpu 0 --variant gem_full &
python train_exp1_512_256.py --gpu 1 --variant gem_const &
python train_exp1_512_256.py --gpu 2 --variant gem_vanilla &
python train_exp1_512_256.py --gpu 3 --variant gem_nod3 &
python train_exp1_512_256.py --gpu 0 --variant concorde &

wait
# ==================== Exp1: [256,128] 183k ====================
echo "===== [256,128] 183k ====="
python train_exp1_256_128.py --gpu 0 --variant gem_full --small &
python train_exp1_256_128.py --gpu 1 --variant gem_const --small &
python train_exp1_256_128.py --gpu 2 --variant gem_vanilla --small &
python train_exp1_256_128.py --gpu 3 --variant gem_nod3 --small &
python train_exp1_256_128.py --gpu 0 --variant concorde --small &

# ==================== Exp1: [512,256] 183k ====================
echo "===== [512,256] 183k ====="
python train_exp1_512_256.py --gpu 0 --variant gem_full --small &
python train_exp1_512_256.py --gpu 1 --variant gem_const --small &
python train_exp1_512_256.py --gpu 2 --variant gem_vanilla --small &
python train_exp1_512_256.py --gpu 3 --variant gem_nod3 --small &
python train_exp1_512_256.py --gpu 1 --variant concorde --small &

wait
echo ""
echo "===== Exp1 done: $(date) ====="

# ==================== Exp2: OOD ====================
# echo ""
# echo "===== Exp2: OOD ====="
# cd /scratch/gem5-TAO/micro_exp/exp2

# # Foundation models
# nohup bash run_exp2_foundation.sh &> exp2_foundation.log &

# # OOD sweep (1%, 5%, 10%, 20%)
# nohup bash run_exp2_all_ratios.sh &> exp2_all_ratios.log &

# wait
# echo ""
# echo "===== All done: $(date) ====="

# ==================== Plot ====================
cd /scratch/gem5-TAO/micro_exp/exp1
python plot_exp1_all.py --base ${RESULT_DIR} --output ${RESULT_DIR}/exp1_all.png
echo "Plots saved to ${RESULT_DIR}"
