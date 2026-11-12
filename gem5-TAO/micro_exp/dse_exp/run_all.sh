#!/bin/bash
# Exp 5c: MSCS-Attributed Greedy DSE
# Methods: gem_attribution → gem_scalar → gem_bo
set -e
export PYTHONUNBUFFERED=1

DSE_DIR="/scratch/gem5-TAO/micro_exp/dse_exp"
cd "$DSE_DIR"

echo "============================================================"
echo "Exp 5c: MSCS-Attributed Greedy DSE"
echo "Start: $(date)"
echo "============================================================"

# --- 605.mcf_s ---
echo ""
echo "=========================================="
echo "DSE: 605.mcf_s — Start: $(date)"
echo "=========================================="
python run_gem_dse.py \
    --benchmark 605.mcf_s \
    --methods gem_attribution gem_scalar gem_bo \
    --n-starts 30 \
    --max-steps 30 \
    --bo-budget 500 \
    --threshold 0.001 \
    --reclaim-threshold 0.05 \
    --seed 42 \
    --heatmap
echo "DSE: 605.mcf_s — Done: $(date)"

# --- 623.xalancbmk_s ---
echo ""
echo "=========================================="
echo "DSE: 623.xalancbmk_s — Start: $(date)"
echo "=========================================="
python run_gem_dse.py \
    --benchmark 623.xalancbmk_s \
    --methods gem_attribution gem_scalar gem_bo \
    --n-starts 30 \
    --max-steps 30 \
    --bo-budget 500 \
    --threshold 0.0006 \
    --reclaim-threshold 0.05 \
    --seed 42 \
    --heatmap
echo "DSE: 623.xalancbmk_s — Done: $(date)"

# --- Plot ---
echo ""
echo "===== Plot results ====="
for BENCH in 605.mcf_s 623.xalancbmk_s; do
    echo "--- Plotting: $BENCH ---"
    python plot_dse_results.py --benchmark "$BENCH"
done

echo ""
echo "============================================================"
echo "All done! $(date)"
echo "============================================================"