
#!/bin/bash
# run_exp4_extra_unlabeled.sh — 18k + extra unlabeled from 40k
#
# 8 ratios, 8 parallel (1 wave)
#
# Usage:
#   cd /path/to/gem5/micro_exp/exp4
#   bash run_exp4_extra_unlabeled.sh

set -e

RESDIR="./results_run1"
LOGDIR="./logs_18k_run2"
mkdir -p "$RESDIR" "$LOGDIR"

echo "============================================"
echo "Exp4 Extra Unlabeled: 18k + 40k extras"
echo "  Results: ${RESDIR}/exp4_18k_extra_unlabeled/"
echo "============================================"

python train_exp4_extra_unlabeled.py --ratio 0.05 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/extra_unlabeled_0.05.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.10 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/extra_unlabeled_0.10.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.15 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/extra_unlabeled_0.15.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.20 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/extra_unlabeled_0.20.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.25 --gpu 2 --results-dir $RESDIR > ${LOGDIR}/extra_unlabeled_0.25.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.30 --gpu 2 --results-dir $RESDIR > ${LOGDIR}/extra_unlabeled_0.30.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.40 --gpu 3 --results-dir $RESDIR > ${LOGDIR}/extra_unlabeled_0.40.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.50 --gpu 3 --results-dir $RESDIR > ${LOGDIR}/extra_unlabeled_0.50.log 2>&1 &
wait

echo "Done!"
