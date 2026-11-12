# #!/bin/bash
# # Run A+D3 OOD experiments across 4 GPUs in parallel
# # Usage: bash run_exp3_d3_parallel.sh

# cd /path/to/gem5/micro_exp/exp3

# # Get all benchmark names from data directory
# BENCHMARKS=($(ls /data/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/.npy//'))
# NUM_BM=${#BENCHMARKS[@]}
# NUM_GPUS=4

# echo "Found $NUM_BM benchmarks, distributing across $NUM_GPUS GPUs"

# # First run ID (fast, single GPU)
# echo "=== Running In-Distribution ==="
# python train_exp3_d3.py --gpu 0 --id-only
# echo "=== ID done ==="

# # Then run OOD in parallel
# echo "=== Running OOD across $NUM_GPUS GPUs ==="
# PIDS=()
# for i in "${!BENCHMARKS[@]}"; do
#     GPU=$((i % NUM_GPUS))
#     BM=${BENCHMARKS[$i]}
#     LOG="results/log_${BM}.txt"
    
#     echo "  Launching $BM on GPU $GPU"
#     python train_exp3_d3.py --gpu $GPU --ood-only --target "$BM" > "$LOG" 2>&1 &
#     PIDS+=($!)
    
#     # Wait every NUM_GPUS jobs
#     if (( (i + 1) % NUM_GPUS == 0 )); then
#         echo "  Waiting for batch $((i / NUM_GPUS + 1))..."
#         for pid in "${PIDS[@]}"; do
#             wait $pid
#         done
#         PIDS=()
#     fi
# done

# # Wait for remaining
# for pid in "${PIDS[@]}"; do
#     wait $pid
# done

# echo "=== All done ==="
# echo "Check results in results/exp3_d3_*.json and results/log_*.txt"


cp train_exp3_residual.py /path/to/gem5/micro_exp/exp3/
cd /path/to/gem5/micro_exp/exp3

# 先跑ID（快）
python train_exp3_residual.py --gpu 0 --id-only

# 4 GPU并行跑OOD
# Batch 1
python train_exp3_residual.py --gpu 0 --ood-only --target 508.namd_r &
python train_exp3_residual.py --gpu 1 --ood-only --target 526.blender_r &
python train_exp3_residual.py --gpu 2 --ood-only --target 600.perlbench_s &
python train_exp3_residual.py --gpu 3 --ood-only --target 602.gcc_s &
wait
# Batch 2
python train_exp3_residual.py --gpu 0 --ood-only --target 607.cactuBSSN_s &
python train_exp3_residual.py --gpu 1 --ood-only --target 619.lbm_s &
python train_exp3_residual.py --gpu 2 --ood-only --target 621.wrf_s &
python train_exp3_residual.py --gpu 3 --ood-only --target 638.imagick_s &
wait
# Batch 3
python train_exp3_residual.py --gpu 0 --ood-only --target 644.nab_s &
python train_exp3_residual.py --gpu 1 --ood-only --target 648.exchange2_s &
python train_exp3_residual.py --gpu 2 --ood-only --target 654.roms_s &
python train_exp3_residual.py --gpu 3 --ood-only --target 657.xz_s &
wait