#!/bin/bash
# (1) small-start trajectories (17 benchmarks x 1000 starts, results_small/, never overwrites)
# (2) gem5 check of 16 random-start walks (gem5_val/, never overwrites)  -- run concurrently
# Usage: setsid nohup bash run_night2.sh > run_night2.log 2>&1 < /dev/null &
cd /scratch/gem5-TAO/micro_exp/dse_v2/traj
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p results_small/logs gem5_val
echo "start $(date)"
GEM5_WORKERS=12 python gem5_traj_val.py > gem5_val/run.log 2>&1 &
GPID=$!
BENCHES="508.namd_r 526.blender_r 600.perlbench_s 602.gcc_s 605.mcf_s 607.cactuBSSN_s 619.lbm_s 620.omnetpp_s 621.wrf_s 623.xalancbmk_s 631.deepsjeng_s 638.imagick_s 641.leela_s 644.nab_s 648.exchange2_s 654.roms_s 657.xz_s"
for b in $BENCHES; do
  python run_traj.py --bench $b --n-starts 1000 --start-mode small --out results_small > results_small/logs/$b.log 2>&1 &
done
wait $GPID; echo "gem5 check done $(date)"
wait
echo "small-start trajectories done $(date)"
echo "ALL DONE $(date)"
