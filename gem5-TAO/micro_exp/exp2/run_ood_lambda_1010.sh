#!/bin/bash
# Cross-program OOD lambda sweep (low-label V3, sources 20% labeled, BN fix, --norm subset):
#   lam_task {0.1,1,10} x lam_mono {0,0.01}; N {256,1024}; 8 targets; seed 42.
# One launcher per combo, placed by free GPU memory. Skips finished runs.
cd /scratch/gem5-TAO/micro_exp/exp2
OUT=/scratch/gem5-TAO/micro_exp/exp2/results_ood_lambda_1010
mkdir -p $OUT
go() {  # $1 lam_task  $2 lam_mono  $3 gpu  $4 max_jobs
  tag="t$(printf %g $1 | sed 's/\./p/')_m$(printf %g $2 | sed 's/\./p/')"
  RESULTS_DIR=$OUT/$tag/src20 SHOTS="256 1024" SEEDS="42" VARIANTS="prism" GPUS="$3" MAX_JOBS=$4 \
  LAM_ARGS="--lam-task $1 --lam-const 1 --lam-mono $2 --lam-unlab 1 --norm subset --src-ratio 0.20" \
  bash run_ood_lowlabel.sh > $OUT/run_$tag.log 2>&1
}
go 0.1 0    1 3 &
go 0.1 0.01 1 3 &
go 1   0    1 3 &
go 1   0.01 0 3 &
go 10  0    3 3 &
go 10  0.01 2 2 &
wait
echo "ALL DONE $(date)"
