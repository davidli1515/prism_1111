#!/bin/bash
# ============================================================
# Full gem5 validation for joint 17-benchmark DSE
#   - BO         : 10 Pareto configs
#   - BO + MSCS  : 19 Pareto configs
# Each config runs all 17 benchmarks (40M insts, 20M warmup).
# Fixes applied: absolute output dir + cwd for wrf/omnetpp/perlbench.
# ============================================================

cd /scratch/gem5-TAO/micro_exp/dse_exp

INSTS=40000000
WORKERS=16

# ------------------------------------------------------------
# BO+MSCS : 19 Pareto configs x 17 benchmarks = 323 runs
# ------------------------------------------------------------
echo "=== Starting BO+MSCS validation (19 configs x 17 bm) ==="
python validate_joint_dse.py \
    --json results_17bm_54/joint_17_gem_bo_mscs_results.json \
    --n-points 39 --insts $INSTS --workers $WORKERS \
    --output-dir validate_joint_mscs \
    2>&1 | tee validate_mscs_729.log
echo "=== BO+MSCS done ==="

# ------------------------------------------------------------
# BO : 10 Pareto configs x 17 benchmarks = 170 runs
# ------------------------------------------------------------
echo "=== Starting BO validation (10 configs x 17 bm) ==="
python validate_joint_dse.py \
    --json results_17bm_54/joint_17_gem_bo_results.json \
    --n-points 18 --insts $INSTS --workers $WORKERS \
    --output-dir validate_joint_bo \
    2>&1 | tee validate_bo_729.log
echo "=== BO done ==="

echo ""
echo "Analyze with:"
echo "  python analyze_joint.py validate_joint_mscs"
echo "  python analyze_joint.py validate_joint_bo"
