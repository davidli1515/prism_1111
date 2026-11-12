#!/bin/bash
# ============================================================
# Exp1 with adaptive loss weights — PRISM variants only
# Results: exp1_v3_run1
# ============================================================
RESULT_DIR="./results/exp1_v3_run1"
# ============================================================

cd /path/to/gem5/micro_exp/exp1

# Update config.py OUTPUT_DIR
sed -i "s|^OUTPUT_DIR.*|OUTPUT_DIR = \"${RESULT_DIR}\"|" config.py
echo "Results: ${RESULT_DIR}"
echo "Start: $(date)"

# ==================== Wave 1: [256,128] 408k + 183k ====================
echo ""
echo "===== Wave 1: [256,128] ====="
python train_exp1_256_128.py --gpu 0 --variant gem_full &
python train_exp1_256_128.py --gpu 0 --variant gem_const &
python train_exp1_256_128.py --gpu 1 --variant gem_vanilla &
python train_exp1_256_128.py --gpu 1 --variant gem_full --small &
python train_exp1_256_128.py --gpu 2 --variant gem_const --small &
python train_exp1_256_128.py --gpu 2 --variant gem_vanilla --small &

wait
echo "===== Wave 1 done: $(date) ====="

# ==================== Wave 2: [512,256] 408k + 183k ====================
echo ""
echo "===== Wave 2: [512,256] ====="
python train_exp1_512_256.py --gpu 0 --variant gem_full &
python train_exp1_512_256.py --gpu 0 --variant gem_const &
python train_exp1_512_256.py --gpu 1 --variant gem_vanilla &
python train_exp1_512_256.py --gpu 1 --variant gem_full --small &
python train_exp1_512_256.py --gpu 2 --variant gem_const --small &
python train_exp1_512_256.py --gpu 2 --variant gem_vanilla --small &

wait
echo "===== Wave 2 done: $(date) ====="

echo ""
echo "===== All done: $(date) ====="

# Check results
echo ""
echo "--- Results ---"
VARIANTS="gem_full gem_const gem_vanilla"
for DIR in exp1_256_128 exp1_512_256 exp1_256_128_183k exp1_512_256_183k; do
    echo "=== ${DIR} ==="
    for V in $VARIANTS; do
        F="${RESULT_DIR}/${DIR}/${V}.json"
        if [ -f "$F" ]; then
            MAPE=$(python3 -c "import json; print(f'{json.load(open(\"$F\"))[\"overall_mape\"]:.2f}%')" 2>/dev/null)
            echo "  ✓ ${V}: ${MAPE}"
        else
            echo "  ✗ ${V}: MISSING"
        fi
    done
    echo ""
done
