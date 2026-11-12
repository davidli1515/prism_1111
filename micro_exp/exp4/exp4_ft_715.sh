cd /path/to/gem5/micro_exp/exp4
RESDIR="/path/to/gem5/micro_exp/exp4/results_run2"
LOGDIR="./logs_run2"

python train_exp4_finetune_unlabeled.py --ratio 0.05 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/ft1_0.05.log 2>&1 &
python train_exp4_finetune_unlabeled.py --ratio 0.15 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/ft1_0.15.log 2>&1 &
python train_exp4_finetune_unlabeled.py --ratio 0.20 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/ft1_0.20.log 2>&1 &
python train_exp4_finetune_unlabeled.py --ratio 0.30 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/ft1_0.30.log 2>&1 &
python train_exp4_finetune_unlabeled.py --ratio 0.40 --gpu 2 --results-dir $RESDIR > ${LOGDIR}/ft1_0.40.log 2>&1 &
python train_exp4_finetune_unlabeled.py --ratio 0.50 --gpu 2 --results-dir $RESDIR > ${LOGDIR}/ft1_0.50.log 2>&1 &
python train_exp4_finetune_unlabeled.py --ratio 0.60 --gpu 3 --results-dir $RESDIR > ${LOGDIR}/ft1_0.60.log 2>&1 &
python train_exp4_finetune_unlabeled.py --ratio 0.70 --gpu 3 --results-dir $RESDIR > ${LOGDIR}/ft1_0.70.log 2>&1 &
wait
python train_exp4_finetune_unlabeled.py --ratio 0.80 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/ft1_0.80.log 2>&1 &
python train_exp4_finetune_unlabeled.py --ratio 0.90 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/ft1_0.90.log 2>&1 &
wait

echo "=== Type 1: Supervised → Unlabeled Finetune ==="
for r in 5 15 20 30 40 50 60 70 80 90; do
  f="${RESDIR}/exp4_18k_pretrain_ft_type1/exp4_${r}pct.json"
  [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); print(f'  {r}%: {d[\"pre_finetune_mape\"]:.2f}% → {d[\"overall_mape\"]:.2f}%')"
done
