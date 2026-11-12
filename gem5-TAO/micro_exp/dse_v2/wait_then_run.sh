#!/bin/bash
# Wait until the user's own jobs are done, then launch the DSE v2 sweep.
#   - waits while any train_exp4_lambda_sens.py process (user's GPU 0/3 jobs) is alive
#   - then waits until GPU 0 and GPU 3 have no compute processes left
#   - DSE itself runs on CPU only (surrogate forward + McPAT), no GPU is used
# Launch (survives closing VS Code):
#   setsid nohup bash wait_then_run.sh > wait_then_run_1005.log 2>&1 < /dev/null &
cd /scratch/gem5-TAO/micro_exp/dse_v2
echo "[$(date)] waiting for train_exp4_lambda_sens.py jobs ..."
while pgrep -u "$USER" -f train_exp4_lambda_sens.py > /dev/null; do sleep 120; done
echo "[$(date)] lambda_sens jobs finished; waiting for GPU 0 and 3 to be idle ..."
gpu_busy() {
  for g in 0 3; do
    u=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && return 0
  done
  return 1
}
while gpu_busy; do sleep 120; done
echo "[$(date)] GPUs idle -> starting DSE v2"
export CUDA_VISIBLE_DEVICES=""            # make sure nothing lands on a GPU
OUT=results_v2_1005 JOBS=8 SEEDS="42" \
METHODS="random bo greedy_scalar greedy_mscs greedy_mscs_jac bo_scalar bo_mscs bo_mscs_jac" \
  bash run_all.sh
echo "[$(date)] wait_then_run finished"
