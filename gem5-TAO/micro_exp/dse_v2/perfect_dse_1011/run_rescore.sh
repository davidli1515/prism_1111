#!/bin/bash
# Rescore all archived configs with stage-resolved stacks (12 single benchmarks + joint_17 in 8 chunks).
cd /scratch/gem5-TAO/micro_exp/dse_v2/perfect_dse_1011
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p logs
{ for b in 600.perlbench_s 602.gcc_s 605.mcf_s 619.lbm_s 620.omnetpp_s 621.wrf_s 623.xalancbmk_s 631.deepsjeng_s 641.leela_s 644.nab_s 648.exchange2_s; do
    [ -f stacks/$b.npz ] || echo "--bench $b"; done
  for k in 0 1 2 3 4 5 6 7; do [ -f stacks/joint_17_c$k-of-8.npz ] || echo "--joint --chunk $k --n-chunks 8"; done
} | xargs -P 12 -I{} sh -c 'python -B rescore.py {} > "logs/$(echo {} | tr -d " -")".log 2>&1'
echo "RESCORE DONE $(date)"
