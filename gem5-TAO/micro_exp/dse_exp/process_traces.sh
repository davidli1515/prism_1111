#!/bin/bash
# Process o3trace.txt through external simulator (cache + branch prediction)
#
# Usage:
#   bash process_traces.sh <bench> [--slice-range S E] [--parallel N]
#
# Examples:
#   # Verify on 3 slices
#   bash process_traces.sh 605.mcf_s --slice-range 0 2
#
#   # Full run, 16 parallel
#   bash process_traces.sh 605.mcf_s --parallel 16
#   bash process_traces.sh 623.xalancbmk_s --parallel 16

# set -e  # disabled: ((n++)) returns 1 when n=0, killing script

# ═══ Paths ═══
SIM_PATH="/scratch/gem5-TAO/external_simulator/simple_trace_analysis"
TRACE_BASE="/scratch/gem5-TAO/micro_exp/dse_exp/traces"

# Checkpoint dirs (simpletrace.txt locations)
declare -A CKPT_DIRS
CKPT_DIRS["605.mcf_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/605.mcf_s/917300000000"
CKPT_DIRS["623.xalancbmk_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/623.xalancbmk_s/537100000000"
CKPT_DIRS["641.leela_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/641.leela_s/1510100000000"
CKPT_DIRS["600.perlbench_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/600.perlbench_s/240100000000"
CKPT_DIRS["508.namd_r"]="/scratch2/wl730/micro_ckpt/m5out_cpts/508.namd_r/1200100000000"
CKPT_DIRS["526.blender_r"]="/scratch2/wl730/micro_ckpt/m5out_cpts/526.blender_r/43600000000"
CKPT_DIRS["602.gcc_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/602.gcc_s/430300000000"
CKPT_DIRS["607.cactuBSSN_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/607.cactuBSSN_s/295000000000"
CKPT_DIRS["619.lbm_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/619.lbm_s/1900300000000"
CKPT_DIRS["620.omnetpp_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/620.omnetpp_s/85300000000"
CKPT_DIRS["621.wrf_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/621.wrf_s/162800000000"
CKPT_DIRS["631.deepsjeng_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/631.deepsjeng_s/1154700000000"
CKPT_DIRS["638.imagick_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/638.imagick_s/41800000000"
CKPT_DIRS["644.nab_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/644.nab_s/217200000000"
CKPT_DIRS["648.exchange2_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/648.exchange2_s/444700000000"
CKPT_DIRS["654.roms_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/654.roms_s/306300000000"
CKPT_DIRS["657.xz_s"]="/scratch2/wl730/micro_ckpt/m5out_cpts/657.xz_s/108200000000"

# Cache config (defaults, can override with --l1i / --l2)
L1_KB=64
L2_KB=1024
BP_TYPE="tage"

# Warmup base: slice_0 = 20,000,000, each subsequent slice += 100,000
WARMUP_BASE=20000000
WARMUP_STEP=100000

# ═══ Parse args ═══
BENCH=${1:?"Usage: $0 <bench> [--slice-range S E] [--parallel N] [--l1i KB] [--l2 KB]"}
shift

SLICE_START=0
SLICE_END=399
N_PARALLEL=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --slice-range) SLICE_START=$2; SLICE_END=$3; shift 3 ;;
        --parallel)    N_PARALLEL=$2; shift 2 ;;
        --l1i)         L1_KB=$2; shift 2 ;;
        --l2)          L2_KB=$2; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TRACE_DIR="${TRACE_BASE}/${BENCH}"
CKPT_DIR="${CKPT_DIRS[$BENCH]}"
SIMPLETRACE="${CKPT_DIR}/simpletrace.txt"

# ═══ Validation ═══
[ -f "$SIM_PATH" ] || { echo "Error: ext sim not found: $SIM_PATH"; exit 1; }
[ -d "$TRACE_DIR" ] || { echo "Error: trace dir not found: $TRACE_DIR"; exit 1; }
[ -f "$SIMPLETRACE" ] || { echo "Error: simpletrace not found: $SIMPLETRACE"; exit 1; }

echo "=== Process Traces: ${BENCH} ==="
echo "  Trace dir:    ${TRACE_DIR}"
echo "  Simpletrace:  ${SIMPLETRACE}"
echo "  Cache:        L1=${L1_KB}KB L2=${L2_KB}KB BP=${BP_TYPE}"
echo "  Slices:       ${SLICE_START}-${SLICE_END}"
echo "  Parallel:     ${N_PARALLEL}"
echo ""

# ═══ Process single slice ═══
process_one() {
    local sid=$1
    local slice_dir="${TRACE_DIR}/slice_${sid}"
    local o3trace="${slice_dir}/o3trace.txt"
    local warmup_n=$((WARMUP_BASE + sid * WARMUP_STEP))

    # Check trace exists
    [ -f "$o3trace" ] || { echo "[slice_${sid}] SKIP: o3trace.txt not found"; return 1; }

    # Check if already processed (marker file)
    local marker="${slice_dir}/.ext_sim_done_${L1_KB}_${L2_KB}_${BP_TYPE}"
    if [ -f "$marker" ]; then
        echo "[slice_${sid}] already processed, skip"
        return 0
    fi

    # 1. Extract warmup from simpletrace
    local warmup_tmp="${slice_dir}/warmup_trace.tmp"
    head -n ${warmup_n} "$SIMPLETRACE" > "$warmup_tmp"

    # 2. Run ext sim warmup (creates cache + bp checkpoint)
    local ckpt="${slice_dir}/extsim_ckpt"
    $SIM_PATH warmup "$warmup_tmp" "$ckpt" $L1_KB $L2_KB $BP_TYPE > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "[slice_${sid}] ERROR: warmup failed"
        rm -f "$warmup_tmp"
        return 1
    fi

    # 3. Run ext sim on o3trace (appends Cache_Latency + BPred_Dir)
    $SIM_PATH sim "$o3trace" "$ckpt" $L1_KB $L2_KB $BP_TYPE > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "[slice_${sid}] ERROR: sim failed"
        rm -f "$warmup_tmp" "${ckpt}_cache.bin" "${ckpt}_bp.bin"
        return 1
    fi

    # 4. Cleanup temp files
    rm -f "$warmup_tmp" "${ckpt}_cache.bin" "${ckpt}_bp.bin"

    # 5. Mark as done
    touch "$marker"

    # 6. Verify output
    local nfields_after=$(head -1 "$o3trace" | awk -F',' '{print NF}')
    echo "[slice_${sid}] done (warmup=${warmup_n}, fields=${nfields_after})"
    return 0
}

export -f process_one
export SIM_PATH TRACE_DIR SIMPLETRACE L1_KB L2_KB BP_TYPE WARMUP_BASE WARMUP_STEP

# ═══ Run ═══
n_ok=0
n_fail=0
t_start=$(date +%s)

if [ $N_PARALLEL -le 1 ]; then
    # Sequential
    for ((s=SLICE_START; s<=SLICE_END; s++)); do
        process_one $s
        [ $? -eq 0 ] && ((n_ok++)) || ((n_fail++))
    done
else
    # Parallel batches
    for ((s=SLICE_START; s<=SLICE_END; s+=N_PARALLEL)); do
        e=$((s + N_PARALLEL - 1))
        [ $e -gt $SLICE_END ] && e=$SLICE_END

        pids=()
        for ((i=s; i<=e; i++)); do
            process_one $i &
            pids+=($!)
        done
        for p in "${pids[@]}"; do
            wait $p
            [ $? -eq 0 ] && ((n_ok++)) || ((n_fail++))
        done
    done
fi

t_end=$(date +%s)
echo ""
echo "=== Done: ${n_ok} OK, ${n_fail} failed, $((t_end - t_start))s ==="