#!/bin/bash
# (1) step-matched labeled-only: concorde, hard @ 1521 epochs (6084 steps), 5/10/20/50% x 3 seeds   (soft = v2_stepmatch, done)
# (2) saturation: concorde, hard, soft @ 3042 epochs (12168 steps), 5/20% x 3 seeds
# (3) PRISM (40k extras) with lam_unlab 0, 5/10/20/50% x 3 seeds (3 concurrent: ~41 GB host RAM each)
# Output only in scratchpad.
SP=/tmp/claude-1017/-scratch-gem5-TAO/90b1d870-5e44-4246-a121-ddc22036bb0a/scratchpad
PY=/scratch2/wl730/conda_envs/hybrid/bin/python; EXP4=/scratch/gem5-TAO/micro_exp/exp4
L=$SP/truth_logs; mkdir -p $L; cd $EXP4
QS=$L/q_small.txt; QP=$L/q_prism.txt; LOCK=$L/.lock; : > $QS; : > $QP
for s in 42 123 456; do
  for r in 0.05 0.10 0.20 0.50; do for k in concorde hard; do
    echo "s1_${k}_r${r}_s$s $SP/train_budget.py --kind $k --ratio $r --epochs 1521 --seed $s --out $SP/s1_$k" >> $QS; done; done
  for r in 0.05 0.20; do for k in concorde hard soft; do
    echo "s2_${k}_r${r}_s$s $SP/train_budget.py --kind $k --ratio $r --epochs 3042 --seed $s --out $SP/s2_$k" >> $QS; done; done
  for r in 0.05 0.10 0.20 0.50; do
    echo "s3_prism_u0_r${r}_s$s $EXP4/train_exp4_prism_lambda.py --ratio $r --lam-task 10 --lam-const 1 --lam-mono 0 --lam-unlab 0 --input concorde --seed $s --out $SP/s3_prism_u0" >> $QP; done
done
mkdir -p $SP/s1_concorde $SP/s1_hard $SP/s2_concorde $SP/s2_hard $SP/s2_soft $SP/s3_prism_u0
pop() { flock $LOCK bash -c "l=\$(head -n1 $1); [ -n \"\$l\" ] && sed -i 1d $1; echo \"\$l\""; }
worker() { local g=$1 q=$2 line t
  while line=$(pop $q); [ -n "$line" ]; do t=${line%% *}
    $PY ${line#* } --gpu $g > $L/$t.log 2>&1; echo "$(date '+%T') gpu$g $t rc=$?"; done; }
for g in 0 1 2; do worker $g $QP & done
for g in 0 1 2 3; do worker $g $QS & worker $g $QS & done
wait; echo ALL DONE
