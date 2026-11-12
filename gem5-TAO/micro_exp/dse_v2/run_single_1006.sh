#!/bin/bash
# Single-benchmark DSE v3: 17 SPEC x {bo, bo_v3_scalar, bo_v3_mscs} x seed 42, GPU 1 (surrogate), McPAT on CPU.
# Skips finished runs (and run_dse_v3_single.py refuses to overwrite).
cd /scratch/gem5-TAO/micro_exp/dse_v2
PY=/scratch2/wl730/conda_envs/hybrid/bin/python
export OMP_NUM_THREADS=2
OUT=results_v3_single_1006; mkdir -p $OUT/logs
JOBS=${JOBS:-6}
BENCHES="605.mcf_s 623.xalancbmk_s 600.perlbench_s 602.gcc_s 619.lbm_s 620.omnetpp_s 621.wrf_s \
 631.deepsjeng_s 638.imagick_s 641.leela_s 644.nab_s 648.exchange2_s 654.roms_s 657.xz_s \
 508.namd_r 526.blender_r 607.cactuBSSN_s"
echo "Start $(date) jobs=$JOBS"
for b in $BENCHES; do
  for m in bo bo_v3_scalar bo_v3_mscs; do
    [ -f $OUT/$b/${m}_s42.json ] && continue
    while [ $(jobs -rp | wc -l) -ge $JOBS ]; do sleep 30; done
    $PY run_dse_v3_single.py --bench $b --method $m --seed 42 --device cuda:1 \
        --total-queries 5000 --bo-queries 500 --out $OUT > $OUT/logs/${b}_${m}_s42.log 2>&1 &
    echo "[$(date +%H:%M)] $b $m"
  done
done
wait
$PY summarize_single.py $OUT > /dev/null 2>&1
echo "ALL DONE $(date)"
