#!/bin/bash
# Overnight queue (2026-10-10). Never overwrites: check_bounds refuses existing files,
# run_gem5_bounds skips runs that already have result.json.
#   1. wait for the running 6-bench x 5-point batch
#   2. surrogate-side check_bounds for the other 6 benchmarks (point selection + PRISM stacks)
#   3. gem5: 12 benchmarks x 10 points (bo500 front), 16 workers
#   4. analysis -> analysis.txt
cd /scratch/gem5-TAO/micro_exp/dse_v2
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=gem5_bounds
echo "queue start $(date)"
while pgrep -f "python run_gem5_bounds.py --workers 16" > /dev/null; do sleep 60; done
echo "batch 1 done $(date)"

NEW="600.perlbench_s 619.lbm_s 620.omnetpp_s 621.wrf_s 631.deepsjeng_s 644.nab_s"
for b in $NEW; do
  [ -f mscs_guide/results/bounds_$b.json ] || python mscs_guide/check_bounds.py --bench $b > mscs_guide/results/logs/bounds_$b.log 2>&1 &
done
wait
echo "check_bounds done $(date)"

ALL="638.imagick_s 623.xalancbmk_s 641.leela_s 648.exchange2_s 602.gcc_s 605.mcf_s $NEW"
cd $G
python run_gem5_bounds.py --benches $ALL --points 10 --workers 16 > full_12x10.log 2>&1
echo "gem5 12x10 done $(date)"
python analyze_gem5_bounds.py > analysis.txt 2>&1
echo "ALL DONE $(date)"
