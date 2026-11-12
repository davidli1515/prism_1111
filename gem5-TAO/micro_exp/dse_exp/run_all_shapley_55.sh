#!/bin/bash
# Run Shapley pruning for all 17 SPEC CPU 2017 benchmarks
# Full MC Shapley is cached after first run; subsequent runs only recompute pruned.
# Runs 4 benchmarks in parallel at a time.

cd /scratch/gem5-TAO/micro_exp/dse_exp

BENCHMARKS=(
    "508.namd_r"
    "526.blender_r"
    "607.cactuBSSN_s"
    "619.lbm_s"
    "620.omnetpp_s"
    "621.wrf_s"
    "631.deepsjeng_s"
    "638.imagick_s"
    "644.nab_s"
    "648.exchange2_s"
    "654.roms_s"
    "657.xz_s"
)

# Already completed — skip these:
# 600.perlbench_s, 602.gcc_s, 605.mcf_s, 623.xalancbmk_s, 641.leela_s

COS_THRESHOLD=0.3
MAX_GROUP_SIZE=12
N_PERMUTATIONS=2000
PARALLEL=4

echo "========================================"
echo "Shapley Pruning — ${#BENCHMARKS[@]} benchmarks"
echo "cos_threshold=$COS_THRESHOLD, max_group=$MAX_GROUP_SIZE, perms=$N_PERMUTATIONS"
echo "Parallel: $PARALLEL at a time"
echo "========================================"

# Run in batches
for ((i=0; i<${#BENCHMARKS[@]}; i+=PARALLEL)); do
    batch=("${BENCHMARKS[@]:i:PARALLEL}")
    echo ""
    echo "--- Batch $((i/PARALLEL + 1)): ${batch[*]} ---"
    
    for bench in "${batch[@]}"; do
        mkdir -p "result_shapley/${bench}"
        echo "  Starting $bench ..."
        python shapley_pruning.py \
            --benchmark "$bench" \
            --cos-threshold $COS_THRESHOLD \
            --max-group-size $MAX_GROUP_SIZE \
            --n-permutations $N_PERMUTATIONS \
            > "result_shapley/${bench}/shapley_log.txt" 2>&1 &
    done
    
    # Wait for this batch to finish
    wait
    echo "  Batch done."
done

echo ""
echo "========================================"
echo "All done! Results in result_shapley/*/shapley_results.json"
echo "========================================"

# Summary
echo ""
echo "Quick summary (all benchmarks):"
ALL_BENCHMARKS=(
    "508.namd_r" "526.blender_r" "600.perlbench_s" "602.gcc_s" "605.mcf_s"
    "607.cactuBSSN_s" "619.lbm_s" "620.omnetpp_s" "621.wrf_s" "623.xalancbmk_s"
    "631.deepsjeng_s" "638.imagick_s" "641.leela_s" "644.nab_s"
    "648.exchange2_s" "654.roms_s" "657.xz_s"
)
for bench in "${ALL_BENCHMARKS[@]}"; do
    bname=$(echo "$bench" | sed 's/.*\.\(.*\)_[sr]/\1/')
    json="result_shapley/${bench}/${bname}_shapley_results.json"
    if [ -f "$json" ]; then
        rho=$(python3 -c "import json; d=json.load(open('$json')); print(f\"{d['metrics']['pruned']['rho']:.3f}\")" 2>/dev/null)
        evals_full=$(python3 -c "import json; d=json.load(open('$json')); print(d['metrics']['full']['evals'])" 2>/dev/null)
        evals_pruned=$(python3 -c "import json; d=json.load(open('$json')); print(d['metrics']['pruned']['evals'])" 2>/dev/null)
        echo "  $bench: ρ=$rho, full=$evals_full evals, pruned=$evals_pruned evals"
    else
        echo "  $bench: FAILED (no JSON)"
    fi
done