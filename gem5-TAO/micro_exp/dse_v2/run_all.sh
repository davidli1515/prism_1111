#!/bin/bash
# DSE v2 sweep: METHODS x SEEDS, CPU jobs (surrogate on CPU, McPAT on CPU).
# Skips finished runs.  Usage:
#   OUT=results_v2_1005 JOBS=12 nohup bash run_all.sh > run_all_1005.log 2>&1 &
cd /scratch/gem5-TAO/micro_exp/dse_v2
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
OUT=${OUT:-results_v2}
JOBS=${JOBS:-12}
METHODS=(${METHODS:-random bo greedy_scalar greedy_mscs bo_scalar bo_mscs})   # ablations: greedy_mscs_decouple greedy_mscs_protect
SEEDS=(${SEEDS:-42})
EXTRA=${EXTRA:-}            # e.g. EXTRA="--total-queries 3000 --bo-queries 500"
mkdir -p $OUT/logs

echo "Start $(date) | ${#METHODS[@]} methods x ${#SEEDS[@]} seeds | jobs=$JOBS"
for s in "${SEEDS[@]}"; do
  for m in "${METHODS[@]}"; do
    [ -f $OUT/${m}_s${s}.json ] && continue
    while [ $(jobs -rp | wc -l) -ge $JOBS ]; do sleep 5; done
    python run_dse_v2.py --method $m --seed $s --out $OUT $EXTRA > $OUT/logs/${m}_s${s}.log 2>&1 &
    echo "[$(date +%H:%M)] $m s$s"
  done
done
wait
python summarize_dse_v2.py $OUT
echo "ALL DONE $(date)"
