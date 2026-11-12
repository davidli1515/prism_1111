#!/bin/bash
# Wait until any GPU has no compute processes, then run the headroom check on it.
# Launch: setsid nohup bash wait_gpu_then_headroom.sh > wait_headroom_1005.log 2>&1 < /dev/null &
cd /scratch/gem5-TAO/micro_exp/dse_v2
PY=/scratch2/wl730/conda_envs/hybrid/bin/python
echo "[$(date)] waiting for an idle GPU ..."
while :; do
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  for g in 0 1 2 3; do
    u=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
    echo "$busy" | grep -q "$u" || { GPU=$g; break 2; }
  done
  sleep 300
done
echo "[$(date)] GPU $GPU idle -> headroom check"
[ -f results_v2_1005/headroom.json ] && { echo "headroom.json exists, skip"; exit 0; }
OMP_NUM_THREADS=4 $PY headroom_check.py results_v2_1005 --rounds 3 --device cuda:$GPU > headroom_1005.log 2>&1
echo "[$(date)] done (see headroom_1005.log, results_v2_1005/headroom.json)"
