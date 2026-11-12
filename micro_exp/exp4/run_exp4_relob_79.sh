cd /path/to/gem5/micro_exp/exp4
RESDIR="/path/to/gem5/micro_exp/exp4/results_run2"
LOGDIR="./logs_run2"
RESDIR2="/path/to/gem5/micro_exp/exp4/results_relobralo"

# 纯 extra
python train_exp4_extra_unlabeled.py --ratio 0.65 --gpu 0 \
  --results-dir $RESDIR > ${LOGDIR}/extra_0.65.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.68 --gpu 1 \
  --results-dir $RESDIR > ${LOGDIR}/extra_0.68.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.70 --gpu 2 \
  --results-dir $RESDIR > ${LOGDIR}/extra_0.70.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 1.0 --gpu 3 \
  --results-dir $RESDIR > ${LOGDIR}/extra_1.00.log 2>&1 &

# extra + bw10_renorm + minlr
python train_exp4_extra_unlabeled.py --ratio 0.65 --gpu 0 \
  --base-weight 10 --renorm-total --min-lr 1.25e-4 \
  --results-dir $RESDIR2 > ${LOGDIR}/extra_bw10r_minlr_0.65.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.68 --gpu 1 \
  --base-weight 10 --renorm-total --min-lr 1.25e-4 \
  --results-dir $RESDIR2 > ${LOGDIR}/extra_bw10r_minlr_0.68.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 0.70 --gpu 2 \
  --base-weight 10 --renorm-total --min-lr 1.25e-4 \
  --results-dir $RESDIR2 > ${LOGDIR}/extra_bw10r_minlr_0.70.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 1.0 --gpu 3 \
  --base-weight 10 --renorm-total --min-lr 1.25e-4 \
  --results-dir $RESDIR2 > ${LOGDIR}/extra_bw10r_minlr_1.00.log 2>&1 &

# 同时跑 PRISM full 100%（无 extra）和 Concorde 100% 做对照
python train_exp4_all.py --dataset 18k --ablation full --ratio 1.0 --gpu 0 \
  --results-dir $RESDIR > ${LOGDIR}/full_1.00.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation concorde --ratio 1.0 --gpu 1 \
  --results-dir $RESDIR > ${LOGDIR}/concorde_1.00.log 2>&1 &
wait

echo "=== Plain Extra ==="
for r in 65 68 70 100; do
  f="${RESDIR}/exp4_18k_extra_unlabeled/exp4_${r}pct.json"
  [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); print(f'  Extra {r}%: {d[\"overall_mape\"]:.2f}%')"
done

echo "=== Extra + bw10r + minlr ==="
for r in 65 68 70 100; do
  f="${RESDIR2}/exp4_18k_extra_unlabeled/exp4_${r}pct.json"
  [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); print(f'  Extra+bw10r {r}%: {d[\"overall_mape\"]:.2f}%')"
done

echo "=== Baselines (exp4 100%) ==="
for v in exp4_18k exp4_18k_concorde; do
  f="${RESDIR}/${v}/exp4_100pct.json"
  [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); print(f'  {\"$v\"}: {d[\"overall_mape\"]:.2f}%')"
done