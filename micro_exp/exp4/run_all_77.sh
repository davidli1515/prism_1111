#/bin/bash
# run_all_77.sh — 183k Label Efficiency with recipe v77
#
# 5 variants × 12 ratios = 60 runs
# 10 parallel per wave, 6 waves
#
# Usage:
#   cd /path/to/gem5/micro_exp/exp4
#   bash run_all_77.sh

set -e

RESDIR="/path/to/gem5/micro_exp/exp4/results_run2"
LOGDIR="./logs_run2"

mkdir -p "$RESDIR" "$LOGDIR"

R() {
    local ABL=$1 RATIO=$2 GPU=$3
    python train_exp4_all.py --dataset 18k --ablation $ABL --ratio $RATIO --gpu $GPU \
        --results-dir "$RESDIR" > "${LOGDIR}/${ABL}_${RATIO}.log" 2>&1 &
}

RV() {
    local RATIO=$1 GPU=$2
    python train_exp4_vanilla.py --dataset 18k --ratio $RATIO --gpu $GPU \
        --results-dir "$RESDIR" > "${LOGDIR}/vanilla_${RATIO}.log" 2>&1 &
}

RX() {
    local RATIO=$1 GPU=$2
    python train_exp4_extra_unlabeled.py --ratio $RATIO --gpu $GPU \
        --results-dir "$RESDIR" > "${LOGDIR}/extra_${RATIO}.log" 2>&1 &
}

echo "============================================"
echo "Exp4 183k — Recipe v77 — 60 runs, 6 waves"
echo "  Results: ${RESDIR}/"
echo "  Logs:    ${LOGDIR}/"
echo "============================================"

# Wave 1: full 5-70% (10 runs)
echo ">>> Wave 1: full 5-70%"
R full 0.05 0;  R full 0.10 0;  R full 0.15 1;  R full 0.20 1;  R full 0.25 2
R full 0.30 2;  R full 0.40 3;  R full 0.50 3;  R full 0.60 0;  R full 0.70 1
wait; echo "  done."

# Wave 2: full 80-90 + sup 5-50% (2 + 8 = 10 runs)
echo ">>> Wave 2: full 80-90 + sup 5-50%"
R full 0.80 0;  R full 0.90 0
R supervised_only 0.05 1;  R supervised_only 0.10 1;  R supervised_only 0.15 2
R supervised_only 0.20 2;  R supervised_only 0.25 3;  R supervised_only 0.30 3
R supervised_only 0.40 0;  R supervised_only 0.50 1
wait; echo "  done."

# Wave 3: sup 60-90 + vanilla 5-40% (4 + 6 = 10 runs)
echo ">>> Wave 3: sup 60-90 + vanilla 5-40%"
R supervised_only 0.60 0;  R supervised_only 0.70 0
R supervised_only 0.80 1;  R supervised_only 0.90 1
RV 0.05 2;  RV 0.10 2;  RV 0.15 3;  RV 0.20 3;  RV 0.25 0;  RV 0.30 1
wait; echo "  done."

# Wave 4: vanilla 40-90 + concorde 5-20% (6 + 4 = 10 runs)
echo ">>> Wave 4: vanilla 40-90 + concorde 5-20%"
RV 0.40 0;  RV 0.50 0;  RV 0.60 1;  RV 0.70 1;  RV 0.80 2;  RV 0.90 2
R concorde 0.05 3;  R concorde 0.10 3;  R concorde 0.15 0;  R concorde 0.20 1
wait; echo "  done."

# Wave 5: concorde 25-90 + extra 5-10% (8 + 2 = 10 runs)
echo ">>> Wave 5: concorde 25-90 + extra 5-10%"
R concorde 0.25 0;  R concorde 0.30 0;  R concorde 0.40 1;  R concorde 0.50 1
R concorde 0.60 2;  R concorde 0.70 2;  R concorde 0.80 3;  R concorde 0.90 3
RX 0.05 0;  RX 0.10 1
wait; echo "  done."

# Wave 6: extra 15-90% (10 runs)
echo ">>> Wave 6: extra 15-90%"
RX 0.15 0;  RX 0.20 0;  RX 0.25 1;  RX 0.30 1;  RX 0.40 2
RX 0.50 2;  RX 0.60 3;  RX 0.70 3;  RX 0.80 0;  RX 0.90 1
wait; echo "  done."

echo ""
echo "============================================"
echo "All done! 60 runs → ${RESDIR}/"
echo ""
echo "  python summary_exp4.py --dataset 18k --results-dir ${RESDIR}"
echo "  python plot_concord_gem.py --dataset 18k --results-dir ${RESDIR}"
echo "============================================"
