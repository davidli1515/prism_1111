#!/bin/bash
# Single-core timing of the offline pipeline on copies (originals untouched): ext sim warmup + sim, then dse_precompute.
SIM=/scratch/gem5-TAO/external_simulator/simple_trace_analysis
TOOL=/scratch/gem5-TAO/micro_exp/dse_exp/precompute_tool/dse_precompute
SRC=/scratch/gem5-TAO/micro_exp/dse_exp/traces/644.nab_s
ST=/scratch2/wl730/micro_ckpt/m5out_cpts/644.nab_s/217200000000/simpletrace.txt
W=$PWD/traces/644.nab_s; O=$PWD/pre; mkdir -p $W $O
export OMP_NUM_THREADS=1
for sid in "$@"; do
  d=$W/slice_$sid; mkdir -p $d; cp $SRC/slice_$sid/o3trace.txt $d/
  n=$((20000000 + sid*100000))
  t0=$(date +%s.%N); head -n $n $ST > $d/warm.tmp; t1=$(date +%s.%N)
  taskset -c 5 $SIM warmup $d/warm.tmp $d/ck 64 1024 tage > /dev/null 2>&1; t2=$(date +%s.%N)
  taskset -c 5 $SIM sim $d/o3trace.txt $d/ck 64 1024 tage > /dev/null 2>&1; t3=$(date +%s.%N)
  taskset -c 5 $TOOL $W $O --verify 1 --slice-range $sid $sid --l1i 64 --l2 1024 > pre_$sid.log 2>&1; t4=$(date +%s.%N)
  rm -f $d/warm.tmp
  echo "slice $sid head $(echo "$t1-$t0"|bc) warmup $(echo "$t2-$t1"|bc) sim $(echo "$t3-$t2"|bc) precompute $(echo "$t4-$t3"|bc)"
done
