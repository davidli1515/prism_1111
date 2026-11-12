cd /path/to/gem5/micro_exp/exp4
RESDIR="/path/to/gem5/micro_exp/exp4/results_run2"
LOGDIR="./logs_run2"

# Wave 1: full 5-70% (10 runs)
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.05 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/full_0.05.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.10 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/full_0.10.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.15 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/full_0.15.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.20 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/full_0.20.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.25 --gpu 2 --results-dir $RESDIR > ${LOGDIR}/full_0.25.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.30 --gpu 2 --results-dir $RESDIR > ${LOGDIR}/full_0.30.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.40 --gpu 3 --results-dir $RESDIR > ${LOGDIR}/full_0.40.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.50 --gpu 3 --results-dir $RESDIR > ${LOGDIR}/full_0.50.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.60 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/full_0.60.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.70 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/full_0.70.log 2>&1 &
wait; echo "Wave 1 done."

# Wave 2: full 80-90 + sup 5-40% (2+8=10 runs)
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.80 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/full_0.80.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.90 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/full_0.90.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.05 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/supervised_only_0.05.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.10 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/supervised_only_0.10.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.15 --gpu 2 --results-dir $RESDIR > ${LOGDIR}/supervised_only_0.15.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.20 --gpu 2 --results-dir $RESDIR > ${LOGDIR}/supervised_only_0.20.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.25 --gpu 3 --results-dir $RESDIR > ${LOGDIR}/supervised_only_0.25.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.30 --gpu 3 --results-dir $RESDIR > ${LOGDIR}/supervised_only_0.30.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.40 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/supervised_only_0.40.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.50 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/supervised_only_0.50.log 2>&1 &
wait; echo "Wave 2 done."

# Wave 3: sup 60-90 + extra 5-30% (4+6=10 runs)
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.60 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/supervised_only_0.60.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.70 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/supervised_only_0.70.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.80 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/supervised_only_0.80.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.90 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/supervised_only_0.90.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.05 --gpu 2 --results-dir $RESDIR > ${LOGDIR}/extra_0.05.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.10 --gpu 2 --results-dir $RESDIR > ${LOGDIR}/extra_0.10.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.15 --gpu 3 --results-dir $RESDIR > ${LOGDIR}/extra_0.15.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.20 --gpu 3 --results-dir $RESDIR > ${LOGDIR}/extra_0.20.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.25 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/extra_0.25.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.30 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/extra_0.30.log 2>&1 &
wait; echo "Wave 3 done."

# Wave 4: extra 40-90% (6 runs)
python train_exp4_extra_unlabeled.py --ratio 0.40 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/extra_0.40.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.50 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/extra_0.50.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.60 --gpu 2 --results-dir $RESDIR > ${LOGDIR}/extra_0.60.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.70 --gpu 3 --results-dir $RESDIR > ${LOGDIR}/extra_0.70.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.80 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/extra_0.80.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.90 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/extra_0.90.log 2>&1 &
wait; echo "Wave 4 done."

echo "All done. 36 runs."
