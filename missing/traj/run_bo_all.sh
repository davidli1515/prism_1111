#!/bin/bash
# BO-as-optimiser trajectories: 17 benchmarks x 10 seeds x budget 500 -> results_bo/ (never overwrites)
cd /scratch/gem5-TAO/micro_exp/dse_v2/traj
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p results_bo/logs
echo "start $(date)"
for b in 508.namd_r 526.blender_r 600.perlbench_s 602.gcc_s 605.mcf_s 607.cactuBSSN_s 619.lbm_s 620.omnetpp_s 621.wrf_s 623.xalancbmk_s 631.deepsjeng_s 638.imagick_s 641.leela_s 644.nab_s 648.exchange2_s 654.roms_s 657.xz_s; do
  python run_bo_traj.py --bench $b --seeds 0 1 2 3 4 5 6 7 8 9 > results_bo/logs/$b.log 2>&1 &
done
wait
echo "ALL DONE $(date)"
