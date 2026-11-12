#!/bin/bash
# v4 queue (session-independent): single-bench bo_v4_mscs for every bench with complete v3,
# then the two joint runs. Skips finished. JOBS concurrent, mcpat-workers 2 each.
cd /scratch/gem5-TAO/micro_exp/dse_v2
PY=/scratch2/wl730/conda_envs/hybrid/bin/python
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
JOBS=${JOBS:-4}
wait_slot() { while [ $(jobs -rp | wc -l) -ge $JOBS ]; do sleep 30; done; }
echo "Start $(date)"
for b in 623.xalancbmk_s 619.lbm_s 620.omnetpp_s 621.wrf_s 631.deepsjeng_s 638.imagick_s 641.leela_s 644.nab_s 648.exchange2_s; do
  [ -f results_v4_1007/$b/bo_v4_mscs_s42.json ] && continue
  wait_slot
  $PY run_dse_v4.py --bench $b --method bo_v4_mscs --seed 42 --mcpat-workers 2 --out results_v4_1007 \
      > results_v4_1007/logs/${b}_bo_v4_mscs_s42.log 2>&1 &
  echo "[$(date +%H:%M)] $b"
done
wait_slot; [ -f results_v4_1007/joint_17/bo_v4_mscs_s42.json ] || { $PY run_dse_v4.py --joint --method bo_v4_mscs --seed 42 --tau-m 0.05 --mcpat-workers 4 --out results_v4_1007 > results_v4_1007/logs/joint_bo_v4_mscs_taum0p05_s42.log 2>&1 & echo "[$(date +%H:%M)] joint tau_m=0.05"; }
wait_slot; [ -f results_v4_1007_noprotect/joint_17/bo_v4_mscs_s42.json ] || { mkdir -p results_v4_1007_noprotect/logs; $PY run_dse_v4.py --joint --method bo_v4_mscs --seed 42 --tau-m 1.0 --mcpat-workers 4 --out results_v4_1007_noprotect > results_v4_1007_noprotect/logs/joint_bo_v4_mscs_noprotect_s42.log 2>&1 & echo "[$(date +%H:%M)] joint no-protect"; }
wait
$PY summarize_v4_single.py > /dev/null 2>&1
echo "ALL DONE $(date)"
