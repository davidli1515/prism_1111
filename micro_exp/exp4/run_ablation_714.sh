# cd /path/to/gem5/micro_exp/exp4
# RESDIR_ABL="/path/to/gem5/micro_exp/exp4/results_ablation"
# LOGDIR="./logs_run2"
# mkdir -p $RESDIR_ABL

# # === Ablation 1: labeled no L_mono, 10 ratios ===
# python train_exp4_all.py --dataset 18k --ablation full --no-mono-labeled --ratio 0.05 --gpu 0 --results-dir $RESDIR_ABL > ${LOGDIR}/no_mono_lab_0.05.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation full --no-mono-labeled --ratio 0.15 --gpu 0 --results-dir $RESDIR_ABL > ${LOGDIR}/no_mono_lab_0.15.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation full --no-mono-labeled --ratio 0.20 --gpu 1 --results-dir $RESDIR_ABL > ${LOGDIR}/no_mono_lab_0.20.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation full --no-mono-labeled --ratio 0.30 --gpu 1 --results-dir $RESDIR_ABL > ${LOGDIR}/no_mono_lab_0.30.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation full --no-mono-labeled --ratio 0.40 --gpu 2 --results-dir $RESDIR_ABL > ${LOGDIR}/no_mono_lab_0.40.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation full --no-mono-labeled --ratio 0.50 --gpu 2 --results-dir $RESDIR_ABL > ${LOGDIR}/no_mono_lab_0.50.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation full --no-mono-labeled --ratio 0.60 --gpu 3 --results-dir $RESDIR_ABL > ${LOGDIR}/no_mono_lab_0.60.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation full --no-mono-labeled --ratio 0.70 --gpu 3 --results-dir $RESDIR_ABL > ${LOGDIR}/no_mono_lab_0.70.log 2>&1 &
# wait
# python train_exp4_all.py --dataset 18k --ablation full --no-mono-labeled --ratio 0.80 --gpu 0 --results-dir $RESDIR_ABL > ${LOGDIR}/no_mono_lab_0.80.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation full --no-mono-labeled --ratio 0.90 --gpu 1 --results-dir $RESDIR_ABL > ${LOGDIR}/no_mono_lab_0.90.log 2>&1 &

# # === Ablation 2: unlabeled const_only, 10 ratios ===
# python train_exp4_all.py --dataset 18k --ablation const_only --ratio 0.05 --gpu 2 --results-dir $RESDIR_ABL > ${LOGDIR}/const_only_0.05.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation const_only --ratio 0.15 --gpu 3 --results-dir $RESDIR_ABL > ${LOGDIR}/const_only_0.15.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation const_only --ratio 0.20 --gpu 0 --results-dir $RESDIR_ABL > ${LOGDIR}/const_only_0.20.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation const_only --ratio 0.30 --gpu 1 --results-dir $RESDIR_ABL > ${LOGDIR}/const_only_0.30.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation const_only --ratio 0.40 --gpu 2 --results-dir $RESDIR_ABL > ${LOGDIR}/const_only_0.40.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation const_only --ratio 0.50 --gpu 3 --results-dir $RESDIR_ABL > ${LOGDIR}/const_only_0.50.log 2>&1 &
# wait
# python train_exp4_all.py --dataset 18k --ablation const_only --ratio 0.60 --gpu 0 --results-dir $RESDIR_ABL > ${LOGDIR}/const_only_0.60.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation const_only --ratio 0.70 --gpu 1 --results-dir $RESDIR_ABL > ${LOGDIR}/const_only_0.70.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation const_only --ratio 0.80 --gpu 2 --results-dir $RESDIR_ABL > ${LOGDIR}/const_only_0.80.log 2>&1 &
# python train_exp4_all.py --dataset 18k --ablation const_only --ratio 0.90 --gpu 3 --results-dir $RESDIR_ABL > ${LOGDIR}/const_only_0.90.log 2>&1 &
# # Extra 100% with [256, 128] — exp1 model size ablation
# python train_exp4_extra_unlabeled.py --ratio 1.0 --gpu 0 --hidden 256 128 --results-dir $RESDIR_ABL > ${LOGDIR}/extra_256_128_1.00.log 2>&1 &
# wait

# echo "=== Ablation 1: No L_mono on labeled ==="
# for r in 5 15 20 30 40 50 60 70 80 90; do
#   f="${RESDIR_ABL}/exp4_18k/exp4_${r}pct.json"
#   [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); print(f'  {r}%: {d[\"overall_mape\"]:.2f}%')"
# done

# echo "=== Ablation 2: Unlabeled const_only ==="
# for r in 5 15 20 30 40 50 60 70 80 90; do
#   f="${RESDIR_ABL}/exp4_18k_const_only/exp4_${r}pct.json"
#   [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); print(f'  {r}%: {d[\"overall_mape\"]:.2f}%')"
# done



cd /path/to/gem5/micro_exp/exp4
RESDIR_ABL="/path/to/gem5/micro_exp/exp4/results_ablation"
LOGDIR="./logs_run2"

# Extra + no L_mono, 全 ratio
python train_exp4_extra_unlabeled.py --ratio 0.05 --gpu 0 --no-mono --results-dir $RESDIR_ABL > ${LOGDIR}/extra_nomono_0.05.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.15 --gpu 0 --no-mono --results-dir $RESDIR_ABL > ${LOGDIR}/extra_nomono_0.15.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.20 --gpu 1 --no-mono --results-dir $RESDIR_ABL > ${LOGDIR}/extra_nomono_0.20.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.30 --gpu 1 --no-mono --results-dir $RESDIR_ABL > ${LOGDIR}/extra_nomono_0.30.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.40 --gpu 2 --no-mono --results-dir $RESDIR_ABL > ${LOGDIR}/extra_nomono_0.40.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.50 --gpu 2 --no-mono --results-dir $RESDIR_ABL > ${LOGDIR}/extra_nomono_0.50.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.60 --gpu 3 --no-mono --results-dir $RESDIR_ABL > ${LOGDIR}/extra_nomono_0.60.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.70 --gpu 3 --no-mono --results-dir $RESDIR_ABL > ${LOGDIR}/extra_nomono_0.70.log 2>&1 &
wait
python train_exp4_extra_unlabeled.py --ratio 0.80 --gpu 0 --no-mono --results-dir $RESDIR_ABL > ${LOGDIR}/extra_nomono_0.80.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.90 --gpu 1 --no-mono --results-dir $RESDIR_ABL > ${LOGDIR}/extra_nomono_0.90.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 1.0 --gpu 2 --no-mono --results-dir $RESDIR_ABL > ${LOGDIR}/extra_nomono_1.00.log 2>&1 &

# 顺便跑 256_128 ablation
python train_exp4_extra_unlabeled.py --ratio 1.0 --gpu 3 --hidden 256 128 --results-dir $RESDIR_ABL > ${LOGDIR}/extra_256_128_1.00.log 2>&1 &
wait

echo "=== Extra + no L_mono ==="
for r in 5 15 20 30 40 50 60 70 80 90 100; do
  f="${RESDIR_ABL}/exp4_18k_extra_unlabeled/exp4_${r}pct.json"
  [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); print(f'  {r}%: {d[\"overall_mape\"]:.2f}%')"
done