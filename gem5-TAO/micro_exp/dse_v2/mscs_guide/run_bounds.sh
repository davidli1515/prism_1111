#!/bin/bash
# Check C (stack bounds -> branch-and-bound). CPU only, ~10-20 min. Never overwrites.
# Usage: nohup bash mscs_guide/run_bounds.sh > mscs_guide/run_bounds.log 2>&1 &
cd /scratch/gem5-TAO/micro_exp/dse_v2
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=mscs_guide
BENCHES=(${BENCHES:-638.imagick_s 623.xalancbmk_s 641.leela_s 648.exchange2_s 602.gcc_s 605.mcf_s})
mkdir -p $G/results/logs
echo "Start $(date)"
for b in "${BENCHES[@]}"; do
  [ -f $G/results/bounds_$b.json ] || python $G/check_bounds.py --bench $b > $G/results/logs/bounds_$b.log 2>&1 &
done
wait
python $G/summarize_bounds.py | tee $G/results/bounds_summary.txt
echo "ALL DONE $(date)"
