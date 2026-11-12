#!/bin/bash
# lam_task {30,100} x lam_mono {0,0.01} x seeds {123,456} at 100% labels on one GPU,
# then evaluate all lambda_probe checkpoints on unseen simpoints (inference only).
cd /path/to/gem5/micro_exp/exp1
export PATH=/data/conda_envs/hybrid/bin:$PATH
GPU=${GPU:-3}; OUT=results/lambda_probe
for t in 30 100; do for m in 0 0.01; do for s in 123 456; do
  tag="t${t}_c1_m$(printf %g $m | sed 's/\./p/')_s$s"
  [ -f $OUT/$tag.json ] && { echo "skip $tag"; continue; }
  python train_exp1_lambda_sens.py --lam-task $t --lam-const 1 --lam-mono $m --seed $s --gpu $GPU \
      --out $OUT > $OUT/logs/$tag.log 2>&1 &
  echo "[$(date +%H:%M)] $tag"
done; done; done
wait
echo "TRAIN DONE $(date)"
cd ../window_ids
python eval_unseen_simpoint.py --gpu $GPU --out unseen_simpoint/eval_lambda_probe_seeds_1005.json \
    --ckpts "../exp1/results/lambda_probe/*.pt" > unseen_simpoint/eval_lambda_probe_seeds_1005.log 2>&1
echo "ALL DONE $(date)"
