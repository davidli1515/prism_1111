#!/bin/bash
# Run Shapley pruning experiments for all 4 benchmarks
# Usage: nohup bash run_shapley.sh &

# cd /scratch/gem5-TAO/micro_exp/dse_exp

# LOG_DIR="results/shapley_logs"
# mkdir -p $LOG_DIR

# echo "=== Shapley Pruning Started: $(date) ===" | tee $LOG_DIR/run.log

# for BM in 605.mcf_s 623.xalancbmk_s 641.leela_s 600.perlbench_s; do
#     echo "--- $BM started: $(date) ---" | tee -a $LOG_DIR/run.log
#     mkdir -p results/$BM
#     python shapley_pruning.py --benchmark $BM --n-permutations 200 \
#         2>&1 | tee results/$BM/shapley_log.txt
#     echo "--- $BM finished: $(date) ---" | tee -a $LOG_DIR/run.log
# done

# echo "=== All Done: $(date) ===" | tee -a $LOG_DIR/run.log


python shapley_pruning.py --benchmark 605.mcf_s --cos-threshold 0.3 --max-group-size 12 --n-permutations 2000 &
python shapley_pruning.py --benchmark 623.xalancbmk_s --cos-threshold 0.3 --max-group-size 12 --n-permutations 2000 &
python shapley_pruning.py --benchmark 641.leela_s --cos-threshold 0.3 --max-group-size 12 --n-permutations 2000 &
python shapley_pruning.py --benchmark 600.perlbench_s --cos-threshold 0.3 --max-group-size 12 --n-permutations 2000 &
wait