#!/bin/bash
# Sweep cosine thresholds for all benchmarks.
# Requires full MC Shapley to be already cached (run run_all_shapley.sh first).
# Only recomputes pruned Shapley at each threshold.

cd /scratch/gem5-TAO/micro_exp/dse_exp

BENCHMARKS=(
    "508.namd_r"
    "526.blender_r"
    "600.perlbench_s"
    "602.gcc_s"
    "605.mcf_s"
    "607.cactuBSSN_s"
    "619.lbm_s"
    "620.omnetpp_s"
    "621.wrf_s"
    "623.xalancbmk_s"
    "631.deepsjeng_s"
    "638.imagick_s"
    "641.leela_s"
    "644.nab_s"
    "648.exchange2_s"
    "654.roms_s"
    "657.xz_s"
)

PARALLEL=4

echo "========================================"
echo "Threshold Sweep — ${#BENCHMARKS[@]} benchmarks"
echo "========================================"

for ((i=0; i<${#BENCHMARKS[@]}; i+=PARALLEL)); do
    batch=("${BENCHMARKS[@]:i:PARALLEL}")
    echo ""
    echo "--- Batch $((i/PARALLEL + 1)): ${batch[*]} ---"
    
    for bench in "${batch[@]}"; do
        mkdir -p "result_shapley/${bench}"
        echo "  Starting $bench ..."
        python sweep_threshold.py \
            --benchmark "$bench" \
            > "result_shapley/${bench}/sweep_log.txt" 2>&1 &
    done
    
    wait
    echo "  Batch done."
done

echo ""
echo "All sweeps done! Plots in result_shapley/*/"