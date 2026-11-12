#!/bin/bash
# Run only GEM+BO for both benchmarks
set -e
export PYTHONUNBUFFERED=1

DSE_DIR="/scratch/gem5-TAO/micro_exp/dse_exp"
cd "$DSE_DIR"

echo "Running GEM+BO only..."
echo "Start: $(date)"

python run_gem_dse.py \
    --benchmark 605.mcf_s \
    --methods gem_bo \
    --bo-budget 500 \
    --seed 42

python run_gem_dse.py \
    --benchmark 623.xalancbmk_s \
    --methods gem_bo \
    --bo-budget 500 \
    --seed 42

# Re-plot with all results
for BENCH in 605.mcf_s 623.xalancbmk_s; do
    python plot_dse_results.py --benchmark "$BENCH"
done

echo "Done! $(date)"