#!/bin/bash
# Joint DSE: all methods × 17 SPEC17 benchmarks
# Usage: nohup bash run_dse.sh [result_dir] [bo_ckpt] &> log.txt &

cd /scratch/gem5-TAO/micro_exp/dse_exp

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

RESULT_DIR="${1:-results_17bm_530}"
BO_CKPT="${2:-}"
mkdir -p $RESULT_DIR

BMS="508.namd_r 526.blender_r 600.perlbench_s 602.gcc_s 605.mcf_s 607.cactuBSSN_s 619.lbm_s 620.omnetpp_s 621.wrf_s 623.xalancbmk_s 631.deepsjeng_s 638.imagick_s 641.leela_s 644.nab_s 648.exchange2_s 654.roms_s 657.xz_s"

echo "=== Joint DSE (17 benchmarks, all methods) Started: $(date) ==="
echo "=== Output: $RESULT_DIR ==="

BO_ARG=""
if [ -n "$BO_CKPT" ]; then
    BO_ARG="--bo-ckpt $BO_CKPT"
    echo "=== BO checkpoint: $BO_CKPT ==="
fi

python -u run_gem_dse.py \
    --benchmark $BMS \
    --methods gem_attribution gem_scalar gem_scalar_noreclaim gem_bo gem_bo_scalar gem_bo_mscs \
    --n-starts 50 --max-steps 30 \
    --reclaim-threshold 0.05 --bo-budget 500 --seed 42 \
    --refine-rounds 3 --refine-steps 15 \
    --output-dir $RESULT_DIR \
    $BO_ARG

python -u plot_dse_results.py --results-dir $RESULT_DIR

echo "=== Done: $(date) ==="
echo "=== Results: $RESULT_DIR ==="