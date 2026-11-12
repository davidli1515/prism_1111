#!/bin/bash
# Run DSE: joint optimization across 4 benchmarks
# Usage: nohup bash run_dse.sh &

cd /scratch/gem5-TAO/micro_exp/dse_exp

TIMESTAMP=$(date +"%m%d_%H%M")
RESULT_DIR="results_${TIMESTAMP}"
mkdir -p $RESULT_DIR

echo "=== Joint DSE Started: $(date) ===" | tee $RESULT_DIR/run.log
echo "=== Results: $RESULT_DIR ===" | tee -a $RESULT_DIR/run.log

# Joint optimization: geometric mean CPI across all 4 benchmarks
python run_gem_dse.py \
    --benchmark 605.mcf_s 623.xalancbmk_s 641.leela_s 600.perlbench_s \
    --methods gem_attribution gem_scalar gem_scalar_noreclaim gem_bo \
    --n-starts 50 --max-steps 30 \
    --reclaim-threshold 0.05 --bo-budget 500 --seed 42 \
    --output-dir $RESULT_DIR \
    2>&1 | tee $RESULT_DIR/dse_log.txt

echo "=== All Done: $(date) ===" | tee -a $RESULT_DIR/run.log
