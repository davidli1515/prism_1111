#!/bin/bash
# waits for dse17/area_model.pkl; if the fit process disappears without producing it, refits (detached)
cd /scratch/gem5-TAO/micro_exp/dse_v2/traj
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
until [ -f ../dse17/area_model.pkl ]; do
  if ! pgrep -f fit_area_model.py >/dev/null; then
    echo "fit not running and no model -> refit $(date)"
    (cd ../dse17 && python -B fit_area_model.py --workers 24 > fit_area_model_retry.log 2>&1)
  fi
  sleep 30
done
while pgrep -f fit_area_model.py >/dev/null; do sleep 10; done
echo "area model ready $(date)"; cat ../dse17/area_model_report.json
bash run_all_traj.sh
