#!/bin/bash
# run_exp4_final.sh — All experiments, results → results_525/
#
# 18k: 10 parallel | 40k: 6 parallel
# Fixed 1500 epochs, no early stopping
#
# Usage:
#   cd /scratch/gem5-TAO/micro_exp/exp4
#   bash run_exp4_final.sh

set -e

RESDIR="./results_525"
LOG18="./logs_18k_525"
LOG40="./logs_40k_525"

mkdir -p "$RESDIR" "$LOG18" "$LOG40"

R() {
    local DS=$1 ABL=$2 RATIO=$3 GPU=$4
    local LD="$LOG18"; [ "$DS" = "40k" ] && LD="$LOG40"
    python train_exp4_all.py --dataset $DS --ablation $ABL --ratio $RATIO --gpu $GPU \
        --results-dir "$RESDIR" > "${LD}/${ABL}_${RATIO}.log" 2>&1 &
}

RV() {
    local DS=$1 RATIO=$2 GPU=$3
    local LD="$LOG18"; [ "$DS" = "40k" ] && LD="$LOG40"
    python train_exp4_vanilla.py --dataset $DS --ratio $RATIO --gpu $GPU \
        --results-dir "$RESDIR" > "${LD}/vanilla_${RATIO}.log" 2>&1 &
}

echo "============================================"
echo "Exp4 FINAL → ${RESDIR}"
echo "  18k: 10 parallel | 40k: 6 parallel"
echo "  Logs: ${LOG18}/ ${LOG40}/"
echo "============================================"

# ##############################################
# 18k — 10 parallel, 48 runs, 5 waves
# ##############################################
echo ""
echo "==================== 18k ===================="

echo ">>> 18k Wave 1: full 5-70%"
R 18k full 0.05 0;  R 18k full 0.10 0;  R 18k full 0.15 1
R 18k full 0.20 1;  R 18k full 0.25 2;  R 18k full 0.30 2
R 18k full 0.40 3;  R 18k full 0.50 3;  R 18k full 0.60 0
R 18k full 0.70 1
wait; echo "  done."

echo ">>> 18k Wave 2: full 80-90 + sup 5-40%"
R 18k full 0.80 0;  R 18k full 0.90 0
R 18k supervised_only 0.05 1;  R 18k supervised_only 0.10 1
R 18k supervised_only 0.15 2;  R 18k supervised_only 0.20 2
R 18k supervised_only 0.25 3;  R 18k supervised_only 0.30 3
R 18k supervised_only 0.40 0;  R 18k supervised_only 0.50 1
wait; echo "  done."

echo ">>> 18k Wave 3: sup 60-90 + vanilla 5-30%"
R 18k supervised_only 0.60 0;  R 18k supervised_only 0.70 0
R 18k supervised_only 0.80 1;  R 18k supervised_only 0.90 1
RV 18k 0.05 2;  RV 18k 0.10 2;  RV 18k 0.15 3
RV 18k 0.20 3;  RV 18k 0.25 0;  RV 18k 0.30 1
wait; echo "  done."

echo ">>> 18k Wave 4: vanilla 40-90 + concorde 5-20%"
RV 18k 0.40 0;  RV 18k 0.50 0;  RV 18k 0.60 1
RV 18k 0.70 1;  RV 18k 0.80 2;  RV 18k 0.90 2
R 18k concorde 0.05 3;  R 18k concorde 0.10 3
R 18k concorde 0.15 0;  R 18k concorde 0.20 1
wait; echo "  done."

echo ">>> 18k Wave 5: concorde 25-90%"
R 18k concorde 0.25 0;  R 18k concorde 0.30 0;  R 18k concorde 0.40 1
R 18k concorde 0.50 1;  R 18k concorde 0.60 2;  R 18k concorde 0.70 2
R 18k concorde 0.80 3;  R 18k concorde 0.90 3
wait; echo "  done."

echo "18k complete."

# ##############################################
# 40k — 6 parallel, 48 runs, 8 waves
# ##############################################
echo ""
echo "==================== 40k ===================="

echo ">>> 40k Wave 1: full 5-30%"
R 40k full 0.05 0;  R 40k full 0.10 1;  R 40k full 0.15 2
R 40k full 0.20 0;  R 40k full 0.25 1;  R 40k full 0.30 2
wait; echo "  done."

echo ">>> 40k Wave 2: full 40-90%"
R 40k full 0.40 0;  R 40k full 0.50 1;  R 40k full 0.60 2
R 40k full 0.70 0;  R 40k full 0.80 1;  R 40k full 0.90 2
wait; echo "  done."

echo ">>> 40k Wave 3: sup 5-30%"
R 40k supervised_only 0.05 0;  R 40k supervised_only 0.10 1;  R 40k supervised_only 0.15 2
R 40k supervised_only 0.20 0;  R 40k supervised_only 0.25 1;  R 40k supervised_only 0.30 2
wait; echo "  done."

echo ">>> 40k Wave 4: sup 40-90%"
R 40k supervised_only 0.40 0;  R 40k supervised_only 0.50 1;  R 40k supervised_only 0.60 2
R 40k supervised_only 0.70 0;  R 40k supervised_only 0.80 1;  R 40k supervised_only 0.90 2
wait; echo "  done."

echo ">>> 40k Wave 5: vanilla 5-30%"
RV 40k 0.05 0;  RV 40k 0.10 1;  RV 40k 0.15 2
RV 40k 0.20 0;  RV 40k 0.25 1;  RV 40k 0.30 2
wait; echo "  done."

echo ">>> 40k Wave 6: vanilla 40-90%"
RV 40k 0.40 0;  RV 40k 0.50 1;  RV 40k 0.60 2
RV 40k 0.70 0;  RV 40k 0.80 1;  RV 40k 0.90 2
wait; echo "  done."

echo ">>> 40k Wave 7: concorde 5-30%"
R 40k concorde 0.05 0;  R 40k concorde 0.10 1;  R 40k concorde 0.15 2
R 40k concorde 0.20 0;  R 40k concorde 0.25 1;  R 40k concorde 0.30 2
wait; echo "  done."

echo ">>> 40k Wave 8: concorde 40-90%"
R 40k concorde 0.40 0;  R 40k concorde 0.50 1;  R 40k concorde 0.60 2
R 40k concorde 0.70 0;  R 40k concorde 0.80 1;  R 40k concorde 0.90 2
wait; echo "  done."

echo "40k complete."

echo ""
echo "============================================"
echo "All done! 96 runs → ${RESDIR}/"
echo ""
echo "  python summary_exp4.py --dataset 18k --results-dir ${RESDIR}"
echo "  python summary_exp4.py --dataset 40k --results-dir ${RESDIR}"
echo "  python plot_concord_gem.py --dataset 18k --outdir ${RESDIR}"
echo "  python plot_concord_gem.py --dataset 40k --outdir ${RESDIR}"
echo "============================================"