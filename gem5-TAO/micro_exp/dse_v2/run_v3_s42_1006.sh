#!/bin/bash
# DSE v3 joint, seed 42 only. Waits until GPU 1 AND GPU 3 have no compute processes,
# then launches 4 methods (GPU 1: bo, bo_v3_mscs; GPU 3: bo_v3_scalar, bo_v3_mscsavg).
# Launch: setsid nohup bash run_v3_s42_1006.sh > run_v3_s42_1006.log 2>&1 < /dev/null &
cd /scratch/gem5-TAO/micro_exp/dse_v2
PY=/scratch2/wl730/conda_envs/hybrid/bin/python
export OMP_NUM_THREADS=2
OUT=results_v3_1006; mkdir -p $OUT/logs
busy() {  # 0 = busy
  u=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader)
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"
}
echo "[$(date)] waiting for GPU 1 and GPU 3 to be idle ..."
while busy 1 || busy 3; do sleep 60; done
echo "[$(date)] GPU 1 and 3 idle -> launch"
for spec in "bo 1" "bo_v3_mscs 1" "bo_v3_scalar 3" "bo_v3_mscsavg 3"; do
  set -- $spec
  [ -f $OUT/$1_s42.json ] && continue
  $PY run_dse_v3.py --method $1 --seed 42 --device cuda:$2 --total-queries 5000 --bo-queries 500 \
      --out $OUT > $OUT/logs/$1_s42.log 2>&1 &
  echo "[$(date +%H:%M)] started $1 on GPU $2"
done
wait
$PY summarize_dse_v2.py $OUT --checkpoints 500 1000 2000 3000 5000 > $OUT/summary.txt 2>&1
echo "ALL DONE $(date)"
