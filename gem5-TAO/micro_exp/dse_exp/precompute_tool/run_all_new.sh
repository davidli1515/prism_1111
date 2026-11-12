#!/bin/bash
# End-to-end pipeline: gem5 trace → ext sim → precompute → cleanup
# For the 13 new benchmarks (mcf/xalan/leela/perlbench already done)
#
# Usage: nohup bash run_all_new.sh &> run_all_new.log &

TOOL="/scratch/gem5-TAO/micro_exp/dse_exp/precompute_tool/dse_precompute"
TRACE_BASE="/scratch/gem5-TAO/micro_exp/dse_exp/traces"
PRE_BASE="/scratch/gem5-TAO/micro_exp/dse_exp/precompute"
DSE_DIR="/scratch/gem5-TAO/micro_exp/dse_exp"

NEW_BENCHMARKS=(
    "508.namd_r" "526.blender_r" "602.gcc_s" "607.cactuBSSN_s"
    "619.lbm_s" "620.omnetpp_s" "621.wrf_s" "631.deepsjeng_s"
    "638.imagick_s" "644.nab_s" "648.exchange2_s" "654.roms_s" "657.xz_s"
)

echo "=== Processing ${#NEW_BENCHMARKS[@]} benchmarks ==="
echo "Started: $(date)"
echo ""

for bench in "${NEW_BENCHMARKS[@]}"; do
    echo "============================================================"
    echo "=== ${bench} === $(date)"
    echo "============================================================"

    # Step 1: Generate o3trace via gem5
    echo "[${bench}] Step 1: gem5 trace generation..."
    python "${DSE_DIR}/prepare_base_features.py" --benchmark "$bench" --workers 24
    n_traces=$(find "${TRACE_BASE}/${bench}" -name "o3trace.txt" 2>/dev/null | wc -l)
    echo "[${bench}] Traces: ${n_traces}/400"

    if [ "$n_traces" -lt 100 ]; then
        echo "[${bench}] WARNING: too few traces (${n_traces}), skipping"
        continue
    fi

    # Step 2: External simulator (cache + branch prediction)
    echo "[${bench}] Step 2: ext sim processing..."
    bash "${DSE_DIR}/process_traces.sh" "$bench" --parallel 28

    # Step 3: Precompute
    echo "[${bench}] Step 3: precompute..."
    for start in 0 100 200 300; do
        $TOOL \
            "${TRACE_BASE}/${bench}" \
            "${PRE_BASE}/${bench}" \
            --verify 3 --slice-range $start $((start+99)) &
    done
    wait

    n_pre=$(find "${PRE_BASE}/${bench}" -name "dse_precomputed.npy" 2>/dev/null | wc -l)
    echo "[${bench}] Precomputed: ${n_pre}/400"

    # Step 4: Cleanup traces (keep only precomputed)
    echo "[${bench}] Step 4: cleanup traces..."
    find "${TRACE_BASE}/${bench}" -name "o3trace.txt" -delete
    find "${TRACE_BASE}/${bench}" -name ".ext_sim_done_*" -delete
    echo "[${bench}] Done"
    echo ""
done

echo "============================================================"
echo "=== Summary ==="
echo "============================================================"
for bench in "${NEW_BENCHMARKS[@]}"; do
    dir="${PRE_BASE}/${bench}"
    count=$(find "$dir" -name "dse_precomputed.npy" 2>/dev/null | wc -l)
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    echo "  ${bench}: ${count}/400 precomputed, ${size}"
done

# Also show existing
for bench in 605.mcf_s 623.xalancbmk_s 641.leela_s 600.perlbench_s; do
    dir="${PRE_BASE}/${bench}"
    count=$(find "$dir" -name "dse_precomputed.npy" 2>/dev/null | wc -l)
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    echo "  ${bench}: ${count} precomputed, ${size} (existing)"
done

total=$(du -sh "$PRE_BASE" | cut -f1)
echo ""
echo "Total storage: ${total}"
echo "Finished: $(date)"