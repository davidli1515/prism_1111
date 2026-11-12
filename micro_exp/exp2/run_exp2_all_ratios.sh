#!/bin/bash
# Run exp2 OOD: 408k (remaining 5% + 10% + 20%) then 183k (5% + 10% + 20%)
# 12 parallel (3 per GPU x 4 GPUs)
# Usage: nohup bash run_exp2_all_ratios.sh &> exp2_all_ratios.log &

cd /path/to/gem5/micro_exp/exp2

ALL_BMS=($(ls /data/micro_traindata_packed_all/gem/*.npy | xargs -n1 basename | sed 's/.npy//'))
SMALL_BMS=($(ls /data/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/.npy//'))

# Already done at 5% (408k)
DONE_5PCT="508.namd_r 526.blender_r 600.perlbench_s 602.gcc_s 605.mcf_s 607.cactuBSSN_s 619.lbm_s 620.omnetpp_s"

run_batch() {
    local RATIO=$1
    local SMALL_FLAG=$2
    shift 2
    local BMS=("$@")
    local TAG=$(python3 -c "print(int($RATIO*100))")
    local DATA_LABEL="408k"
    [ "$SMALL_FLAG" = "--small" ] && DATA_LABEL="183k"
    echo ""
    echo "========================================"
    echo "  [${DATA_LABEL}] Ratio=${TAG}%  Benchmarks=${#BMS[@]}"
    echo "  $(date)"
    echo "========================================"
    for ((i=0; i<${#BMS[@]}; i+=12)); do
        for j in 0 1 2 3 4 5 6 7 8 9 10 11; do
            idx=$((i+j))
            [ $idx -ge ${#BMS[@]} ] && continue
            GPU=$((j % 4))
            echo "  [${DATA_LABEL}|${TAG}%] ${BMS[$idx]} -> GPU $GPU"
            python train_exp2.py --gpu $GPU --ratio $RATIO --target "${BMS[$idx]}" $SMALL_FLAG &
        done
        wait
        echo "  [${DATA_LABEL}|${TAG}%] Batch $((i/12+1)) done: $(date)"
    done
    echo "  [${DATA_LABEL}|${TAG}%] All done: $(date)"
}

# ========== 408k ==========

# 1. Remaining 5% (408k)
REMAINING_5=()
for bm in "${ALL_BMS[@]}"; do
    if [[ ! " $DONE_5PCT " =~ " $bm " ]]; then
        REMAINING_5+=("$bm")
    fi
done
echo "408k remaining 5%: ${#REMAINING_5[@]} benchmarks"
run_batch 0.05 "" "${REMAINING_5[@]}"

# 2. All 10% (408k)
echo ""; echo "=== 408k 10% ==="
run_batch 0.10 "" "${ALL_BMS[@]}"

# 3. All 20% (408k)
echo ""; echo "=== 408k 20% ==="
run_batch 0.20 "" "${ALL_BMS[@]}"

# ========== 183k ==========

# 4. All 5% (183k)
echo ""; echo "=== 183k 5% ==="
run_batch 0.05 "--small" "${SMALL_BMS[@]}"

# 5. All 10% (183k)
echo ""; echo "=== 183k 10% ==="
run_batch 0.10 "--small" "${SMALL_BMS[@]}"

# 6. All 20% (183k)
echo ""; echo "=== 183k 20% ==="
run_batch 0.20 "--small" "${SMALL_BMS[@]}"

echo ""
echo "========== ALL DONE: $(date) =========="
echo ""

# Collect summary
collect() {
    local DIR=$1
    if [ -d "$DIR" ]; then
        echo "--- $DIR ---"
        python3 -c "
import json, glob, numpy as np
files = sorted(glob.glob('$DIR/exp2_ood_*.json'))
if not files: print('  No results'); exit()
results = [json.load(open(f)) for f in files]
g = np.mean([r['gem_full'] for r in results])
c = np.mean([r['concorde'] for r in results])
w = sum(1 for r in results if r['gem_full'] <= r['concorde'])
print(f'  {len(results)} benchmarks: GEM={g:.2f}% Concorde={c:.2f}% GEM_wins={w}/{len(results)}')
for r in sorted(results, key=lambda x:x['target']):
    flag = '<--' if r['gem_full'] <= r['concorde'] else ''
    print(f'    {r[\"target\"]:<20} GEM={r[\"gem_full\"]:>6.2f}% Con={r[\"concorde\"]:>6.2f}% {flag}')
"
    fi
}

echo "========== SUMMARY =========="
for DIR in results_5pct results_10pct results_20pct results_5pct_183k results_10pct_183k results_20pct_183k; do
    collect $DIR
done