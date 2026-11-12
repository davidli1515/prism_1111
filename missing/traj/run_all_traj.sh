#!/bin/bash
# Greedy-trajectory collection, CPU only, detached. Never overwrites (parts are skipped if present).
#   selfcheck -> 17 single-benchmark runs (1000 starts each, in parallel) -> joint-17 run (2 shards x 17
#   workload workers) -> McPAT check of the fast area model on a state sample
# Usage: setsid nohup bash run_all_traj.sh > run_all_traj.log 2>&1 < /dev/null &
cd /scratch/gem5-TAO/micro_exp/dse_v2/traj
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
N=${N_STARTS:-1000}
mkdir -p results/logs
echo "start $(date)  n_starts=$N"
python selfcheck_traj.py 602.gcc_s > results/logs/selfcheck.log 2>&1
grep -q "SELFCHECK_TRAJ PASS" results/logs/selfcheck.log || { echo "selfcheck failed"; exit 1; }
BENCHES="508.namd_r 526.blender_r 600.perlbench_s 602.gcc_s 605.mcf_s 607.cactuBSSN_s 619.lbm_s 620.omnetpp_s 621.wrf_s 623.xalancbmk_s 631.deepsjeng_s 638.imagick_s 641.leela_s 644.nab_s 648.exchange2_s 654.roms_s 657.xz_s"
for b in $BENCHES; do
  python run_traj.py --bench $b --n-starts $N > results/logs/$b.log 2>&1 &
done
wait
echo "singles done $(date)"
for s in 0 1; do
  python run_traj.py --joint --n-starts $N --shard $s --n-shards 2 > results/logs/joint_s$s.log 2>&1 &
done
wait
echo "joint done $(date)"
python check_area.py > results/logs/area_check.log 2>&1
echo "ALL DONE $(date)"
