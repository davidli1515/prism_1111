#!/bin/bash
# exp1 loss-weight sweep (100% labels) WITHOUT the 808 D3 features (--input concorde, 3873 dims),
# same 29 configs x 3 seeds as results/lambda_sweep_1004 + results/lambda_probe_1005.
# Memory-aware: start a job on a GPU only if it has >= MIN_FREE_MB free and < PER_GPU of our jobs;
# wait SETTLE s after each launch. Finished runs are skipped; the trainer refuses to overwrite.
# DRY=1 only prints the queue.
# Usage: mkdir -p results/lambda_sweep_nod3_1026; nohup bash run_exp1_lambda_nod3_1026.sh > results/lambda_sweep_nod3_1026/launcher.log 2>&1 &
cd "$(dirname "$0")"
PY=/scratch2/wl730/conda_envs/hybrid/bin/python
OUT=results/lambda_sweep_nod3_1026; LOG=$OUT/logs
MIN_FREE_MB=${MIN_FREE_MB:-12000}; PER_GPU=${PER_GPU:-2}; SETTLE=${SETTLE:-90}; GPUS=(0 1 2 3)
CFGS=""
for t in 0.1 0.3 1 3 10; do for m in 0 0.01 0.1 1; do CFGS+="$t,1,$m "; done; done
for c in 0 0.1 0.3 3 10; do CFGS+="1,$c,0.01 "; done
for t in 30 100; do for m in 0 0.01; do CFGS+="$t,1,$m "; done; done
f() { python3 -c "print(('%g' % $1).replace('.', 'p'))"; }
JOBS=()
for s in 42 123 456; do for cfg in $CFGS; do
  IFS=, read t c m <<< "$cfg"
  tag=t$(f $t)_c$(f $c)_m$(f $m)_s${s}_nod3
  [ -e $OUT/$tag.json ] || JOBS+=("$tag --lam-task $t --lam-const $c --lam-mono $m --seed $s")
done; done
echo "$(date '+%F %T') queued ${#JOBS[@]} jobs (MIN_FREE_MB=$MIN_FREE_MB PER_GPU=$PER_GPU)"
if [ -n "$DRY" ]; then printf '%s\n' "${JOBS[@]}" | cut -d' ' -f1; exit 0; fi
mkdir -p $LOG
declare -A PIDGPU
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  launched=0
  for g in ${GPUS[@]}; do
    [ $i -ge ${#JOBS[@]} ] && break
    n=0; for p in "${!PIDGPU[@]}"; do
      if kill -0 $p 2>/dev/null; then [ "${PIDGPU[$p]}" = $g ] && n=$((n+1)); else unset PIDGPU[$p]; fi
    done
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
    if [ $n -lt $PER_GPU ] && [ $free -ge $MIN_FREE_MB ]; then
      job=${JOBS[$i]}; tag=${job%% *}
      $PY train_exp1_lambda_sens.py ${job#* } --input concorde --gpu $g --out $OUT > $LOG/$tag.log 2>&1 &
      PIDGPU[$!]=$g; echo "$(date '+%F %T') gpu$g (free ${free}MB) start $tag"
      i=$((i+1)); launched=1; sleep $SETTLE
    fi
  done
  [ $launched = 0 ] && sleep 60
done
wait
echo "$(date '+%F %T') all done; $(ls $OUT/*.json | wc -l) results"
