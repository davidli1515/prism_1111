cd /scratch/gem5-TAO/micro_exp/exp4
RESDIR="/scratch/gem5-TAO/micro_exp/exp4/results_77_p90"
LOGDIR="./logs_77"
mkdir -p $RESDIR

python train_exp4_extra_unlabeled.py --ratio 0.70 --gpu 0 --results-dir $RESDIR > ${LOGDIR}/extra_p90_0.70.log 2>&1 &
python train_exp4_extra_unlabeled.py --ratio 1.00 --gpu 1 --results-dir $RESDIR > ${LOGDIR}/extra_p90_1.00.log 2>&1 &
wait

echo "=== Results with P90 ==="
for r in 70 100; do
  f="${RESDIR}/exp4_18k_extra_unlabeled/exp4_${r}pct.json"
  [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); print(f'  {r}%: MAPE={d[\"overall_mape\"]:.2f}%, P90={d.get(\"p90\",\"N/A\")}')"
done