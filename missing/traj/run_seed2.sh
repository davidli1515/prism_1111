#!/bin/bash
# 100 fresh random starts (seed 2026+1000) per benchmark -> results_seed2/ (never overwrites), then robustness analysis
cd /scratch/gem5-TAO/micro_exp/dse_v2/traj
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p results_seed2/logs
echo "start $(date)"
for b in 508.namd_r 526.blender_r 600.perlbench_s 602.gcc_s 605.mcf_s 607.cactuBSSN_s 619.lbm_s 620.omnetpp_s 621.wrf_s 623.xalancbmk_s 631.deepsjeng_s 638.imagick_s 641.leela_s 644.nab_s 648.exchange2_s 654.roms_s 657.xz_s; do
  python run_traj.py --bench $b --n-starts 100 --seed-offset 1000 --out results_seed2 > results_seed2/logs/$b.log 2>&1 &
done
wait
echo "fresh starts done $(date)"
OMP_NUM_THREADS=8 python robust_clusters.py > results_seed2/robust_clusters.log 2>&1
echo "ALL DONE $(date)"
