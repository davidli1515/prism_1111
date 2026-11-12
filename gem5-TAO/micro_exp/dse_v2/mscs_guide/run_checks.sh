#!/bin/bash
# MSCS-guidance feasibility checks (CPU only, ~1-1.5 h on 32 cores). Never overwrites.
#   1. build_map.py      relief map on 40 random configs, 17 workloads   (~20 min, 1 proc)
#   2. check_prune.py    A1/A2 on 6 benchmarks in parallel              (~35 min)
#   3. check_pairs.py    B on the 10 lowest-gain states per benchmark    (~20 min)
#   4. summarize.py
# Usage: nohup bash mscs_guide/run_checks.sh > mscs_guide/run_checks.log 2>&1 &
cd /scratch/gem5-TAO/micro_exp/dse_v2
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=mscs_guide
MAP=$G/maps/map_n40_s0.npz
BENCHES=(${BENCHES:-638.imagick_s 623.xalancbmk_s 641.leela_s 648.exchange2_s 602.gcc_s 605.mcf_s})
mkdir -p $G/results/logs

echo "Start $(date)"
[ -f $MAP ] || python $G/build_map.py --n-states 40 --seed 0 > $G/results/logs/build_map.log 2>&1
[ -f $MAP ] || { echo "map failed"; exit 1; }
echo "map done $(date)"

for b in "${BENCHES[@]}"; do
  [ -f $G/results/prune_$b.json ] || python $G/check_prune.py --bench $b --map $MAP > $G/results/logs/prune_$b.log 2>&1 &
done
wait
echo "prune done $(date)"

for b in "${BENCHES[@]}"; do
  [ -f $G/results/pairs_$b.json ] || python $G/check_pairs.py --bench $b --map $MAP > $G/results/logs/pairs_$b.log 2>&1 &
done
wait
echo "pairs done $(date)"
python $G/summarize.py | tee $G/results/summary.txt
echo "ALL DONE $(date)"
