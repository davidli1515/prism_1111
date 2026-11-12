#!/bin/bash
# dse17 pipeline (CPU). Never overwrites; writes only under dse17/.
#   selfcheck17 -> fit fast area model (iso mode only) -> BO base -> arms x modes -> summary
# Env: SEED (42), ARMS ("mscs_protect scalar_protect"), MODES ("orig iso"), DELTA (0.03)
# Usage: setsid nohup bash run_all17.sh > run_all17.log 2>&1 < /dev/null &
cd /scratch/gem5-TAO/micro_exp/dse_v2/dse17
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
SEED=${SEED:-42}; ARMS=${ARMS:-"mscs_protect scalar_protect"}; MODES=${MODES:-"orig iso"}; DELTA=${DELTA:-0.03}
mkdir -p results/logs
echo "start $(date)  seed=$SEED arms=[$ARMS] modes=[$MODES] delta=$DELTA"
python selfcheck17.py > results/logs/selfcheck17.log 2>&1
grep -q "SELFCHECK17 PASS" results/logs/selfcheck17.log || { echo "selfcheck failed"; exit 1; }
if [[ "$MODES" == *iso* ]] && [ ! -f area_model.pkl ]; then
  python fit_area_model.py > results/logs/fit_area_model.log 2>&1 || { echo "area model fit failed"; exit 1; }
fi
[ -f results/bo_base_s$SEED.json ] || python run_dse17.py --arm bo --seed $SEED > results/logs/bo_s$SEED.log 2>&1
[ -f results/bo_base_s$SEED.json ] || { echo "BO failed"; exit 1; }
echo "BO done $(date)"
for mode in $MODES; do
  for arm in $ARMS; do
    python run_dse17.py --arm $arm --mode $mode --delta $DELTA --seed $SEED > results/logs/${arm}_${mode}_s$SEED.log 2>&1 &
  done
  wait
  echo "$mode done $(date)"
done
python summarize17.py --seed $SEED | tee results/summary17_s$SEED.txt
echo "ALL DONE $(date)"
