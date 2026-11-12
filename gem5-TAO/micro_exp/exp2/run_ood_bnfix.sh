#!/bin/bash
# ============================================================
# Exp2 v2 OOD sweep: 23 targets x N_SHOTS x VARIANTS x SEEDS
# Skips runs whose JSON already exists (safe to re-launch).
#
# Usage (override any list via env):
#   nohup bash run_ood.sh &> run_ood.log &
#   GPUS="1 3" MAX_JOBS=8 SHOTS="0 256" SEEDS="42" bash run_ood.sh
#
# 18k dataset only. Data is GPU-resident: ~5 GB GPU mem/job;
# host RAM peaks at ~2x the train split (~9 GB) during setup only.
# ============================================================

cd /scratch/gem5-TAO/micro_exp/exp2
export PATH=/scratch2/wl730/conda_envs/hybrid/bin:$PATH

DATA_DIR="/scratch2/wl730/micro_traindata_packed/gem"
RESULTS_DIR="${RESULTS_DIR:-/scratch/gem5-TAO/micro_exp/exp2/results_ood_bnfix_1006}"
LOG_DIR="${RESULTS_DIR}/logs"
BMS=(${TARGETS:-602.gcc_s 605.mcf_s 607.cactuBSSN_s 638.imagick_s 508.namd_r snappy re2 657.xz_s})
SHOTS=(${SHOTS:-32 256 1024 4096})  # max 7476: smallest bm has 7975 rows x 15/16
VARIANTS=(${VARIANTS:-prism})
LAM_ARGS="${LAM_ARGS:---lam-task 10 --lam-const 1 --lam-mono 0.01 --lam-unlab 1 --norm batch}"
SEEDS=(${SEEDS:-42})
GPUS=(${GPUS:-0 1 2 3})
MAX_JOBS=${MAX_JOBS:-16}   # ~4 per GPU

mkdir -p "${LOG_DIR}"

declare -A PID_GPU

reap() {
    for pid in "${!PID_GPU[@]}"; do
        kill -0 "$pid" 2>/dev/null || unset PID_GPU[$pid]
    done
}

wait_for_slot() {
    while [ ${#PID_GPU[@]} -ge $MAX_JOBS ]; do
        wait -n 2>/dev/null || sleep 5
        reap
    done
}

pick_gpu() {
    local best=${GPUS[0]} min=999
    for g in "${GPUS[@]}"; do
        local c=0
        for pid in "${!PID_GPU[@]}"; do [ "${PID_GPU[$pid]}" = "$g" ] && ((c++)); done
        if [ $c -lt $min ]; then min=$c; best=$g; fi
    done
    echo $best
}

echo "Start: $(date)"
echo "Targets(${#BMS[@]}): ${BMS[*]}"
echo "Shots: ${SHOTS[*]} | Variants: ${VARIANTS[*]} | Seeds: ${SEEDS[*]}"
echo "GPUs: ${GPUS[*]} | Max jobs: ${MAX_JOBS}"

for S in "${SEEDS[@]}"; do
for N in "${SHOTS[@]}"; do
for V in "${VARIANTS[@]}"; do
for BM in "${BMS[@]}"; do
    OUT="${RESULTS_DIR}/${V}/${BM}/n${N}_s${S}.json"
    if [ -f "$OUT" ]; then echo "[skip] $OUT"; continue; fi
    wait_for_slot
    G=$(pick_gpu)
    LOG="${LOG_DIR}/${V}_${BM}_n${N}_s${S}.log"
    echo "[$(date +%H:%M)] GPU $G: ${V} ${BM} N=${N} seed=${S}"
    python train_ood_bnfix.py ${LAM_ARGS} --target "$BM" --n-shot "$N" --variant "$V" --seed "$S" \
        --gpu "$G" --results-dir "$RESULTS_DIR" > "$LOG" 2>&1 &
    PID_GPU[$!]=$G
done
done
done
done

wait
echo "End: $(date)"
python summarize_ood.py --results-dir "$RESULTS_DIR"
