cd /scratch/gem5-TAO/micro_exp/exp4

RESDIR="./results_525"
LOGDIR="./logs_18k_525"

python train_exp4_extra_unlabeled.py --ratio 0.60 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/extra_unlabeled_0.60.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.70 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/extra_unlabeled_0.70.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.80 --gpu 2 --results-dir $RESDIR > ${LOGDIR}/extra_unlabeled_0.80.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.90 --gpu 3 --results-dir $RESDIR > ${LOGDIR}/extra_unlabeled_0.90.log 2>&1 &
wait
