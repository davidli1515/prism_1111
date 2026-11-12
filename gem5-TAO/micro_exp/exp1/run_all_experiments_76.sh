#!/bin/bash
# ============================================================
# Exp1 with lambda_task=0.1, 183k only
# ============================================================
RESULT_DIR="./results/exp1_v3_76"
# ============================================================

cd /scratch/gem5-TAO/micro_exp/exp1
sed -i "s|^OUTPUT_DIR.*|OUTPUT_DIR = \"${RESULT_DIR}\"|" config.py
echo "Results: ${RESULT_DIR}"
echo "Start: $(date)"

# [512,256] 183k
python train_exp1_512_256.py --gpu 0 --variant gem_full --small &
python train_exp1_512_256.py --gpu 0 --variant gem_const --small &

# [256,128] 183k
python train_exp1_256_128.py --gpu 1 --variant gem_full --small &
python train_exp1_256_128.py --gpu 1 --variant gem_const --small &

wait
echo "Done: $(date)"

# Check
for DIR in exp1_512_256_183k exp1_256_128_183k; do
    echo "=== ${DIR} ==="
    for V in gem_full gem_const; do
        F="${RESULT_DIR}/${DIR}/${V}.json"
        if [ -f "$F" ]; then
            MAPE=$(python3 -c "import json; print(f'{json.load(open(\"$F\"))[\"overall_mape\"]:.2f}%')" 2>/dev/null)
            echo "  ✓ ${V}: ${MAPE}"
        else
            echo "  ✗ ${V}: MISSING"
        fi
    done
done
