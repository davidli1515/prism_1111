#!/bin/bash
# Coordinated-swap 2x2 ablation, seed 42, CPU only. Never overwrites (run_coord.py refuses).
#   verify_fast (bit-exact features) -> selfcheck -> shared BO base -> 4 arms (2 at a time, 17 workers each) -> summary
# Usage: setsid nohup bash run_all.sh > run_all.log 2>&1 < /dev/null &
cd /scratch/gem5-TAO/micro_exp/dse_v2/coord_ablation
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1  # no .pyc into dse_exp/ or exp1/
mkdir -p results/logs
SEED=${SEED:-42}
echo "start $(date)"
python verify_fast.py > results/logs/verify_fast.log 2>&1
grep -q "VERIFY_FAST PASS" results/logs/verify_fast.log || { echo "fast features not bit-exact"; exit 1; }
python run_coord.py --selfcheck > results/logs/selfcheck.log 2>&1
grep -q "SELFCHECK PASS" results/logs/selfcheck.log || { echo "selfcheck failed"; exit 1; }
[ -f results/bo_base_s$SEED.json ] || python run_coord.py --arm bo --seed $SEED > results/logs/bo_s$SEED.log 2>&1
[ -f results/bo_base_s$SEED.json ] || { echo "BO failed"; exit 1; }
echo "BO done $(date)"
for pair in "mscs_protect scalar_noprotect" "mscs_noprotect scalar_protect"; do
  for arm in $pair; do
    [ -f results/${arm}_s$SEED.json ] || python run_coord.py --arm $arm --seed $SEED > results/logs/${arm}_s$SEED.log 2>&1 &
  done
  wait
  echo "$pair done $(date)"
done
python summarize.py --seed $SEED | tee results/summary_s$SEED.txt
echo "ALL DONE $(date)"
