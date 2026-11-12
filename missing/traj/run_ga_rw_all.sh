#!/bin/bash
# GA (ArchGym sko.GA, 10 seeds x 512 evals) then random walker (100k designs) for 17 benchmarks
# -> results_ga/, results_rw/ (never overwrites; finished seeds/parts are skipped)
cd /scratch/gem5-TAO/micro_exp/dse_v2/traj
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
B="508.namd_r 526.blender_r 600.perlbench_s 602.gcc_s 605.mcf_s 607.cactuBSSN_s 619.lbm_s 620.omnetpp_s 621.wrf_s 623.xalancbmk_s 631.deepsjeng_s 638.imagick_s 641.leela_s 644.nab_s 648.exchange2_s 654.roms_s 657.xz_s"
mkdir -p results_ga/logs results_rw/logs
echo "GA start $(date)"
for b in $B; do python run_ga_traj.py --bench $b --seeds 0 1 2 3 4 5 6 7 8 9 2>&1 | grep -v "^iter:" > results_ga/logs/$b.log & done
wait
echo "GA done $(date); RW start"
for b in $B; do python run_rw_traj.py --bench $b --n 100000 > results_rw/logs/$b.log 2>&1 & done
wait
echo "ALL DONE $(date)"
