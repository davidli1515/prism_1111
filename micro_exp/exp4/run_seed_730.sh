cd /path/to/gem5/micro_exp/exp4
RESDIR_ABL="/path/to/gem5/micro_exp/exp4/results_ablation_seeds"
LOGDIR="./logs_run2"
mkdir -p $RESDIR_ABL

# V3 full (baseline)
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.05 --gpu 0 --seed 42 --results-dir $RESDIR_ABL > ${LOGDIR}/v3_s42_5.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.05 --gpu 0 --seed 123 --results-dir $RESDIR_ABL > ${LOGDIR}/v3_s123_5.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.05 --gpu 1 --seed 456 --results-dir $RESDIR_ABL > ${LOGDIR}/v3_s456_5.log 2>&1 &

# V3 remove L_mono from L (labeled)
python train_exp4_all.py --dataset 18k --ablation full --no-mono-labeled --ratio 0.05 --gpu 1 --seed 42 --results-dir ${RESDIR_ABL}_nomono_L > ${LOGDIR}/v3nomonoL_s42_5.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --no-mono-labeled --ratio 0.05 --gpu 2 --seed 123 --results-dir ${RESDIR_ABL}_nomono_L > ${LOGDIR}/v3nomonoL_s123_5.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --no-mono-labeled --ratio 0.05 --gpu 2 --seed 456 --results-dir ${RESDIR_ABL}_nomono_L > ${LOGDIR}/v3nomonoL_s456_5.log 2>&1 &

# V3 remove L_mono from U (unlabeled) = const_only
python train_exp4_all.py --dataset 18k --ablation const_only --ratio 0.05 --gpu 3 --seed 42 --results-dir ${RESDIR_ABL}_nomono_U > ${LOGDIR}/v3nomonoU_s42_5.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation const_only --ratio 0.05 --gpu 3 --seed 123 --results-dir ${RESDIR_ABL}_nomono_U > ${LOGDIR}/v3nomonoU_s123_5.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation const_only --ratio 0.05 --gpu 0 --seed 456 --results-dir ${RESDIR_ABL}_nomono_U > ${LOGDIR}/v3nomonoU_s456_5.log 2>&1 &

wait

echo "=== V3 full ==="
grep overall_mape ${RESDIR_ABL}/exp4_18k/exp4_5pct_s*.json
echo "=== V3 no L_mono from L ==="
grep overall_mape ${RESDIR_ABL}_nomono_L/exp4_18k/exp4_5pct_s*.json
echo "=== V3 no L_mono from U ==="
grep overall_mape ${RESDIR_ABL}_nomono_U/exp4_18k/exp4_5pct_s*.json
