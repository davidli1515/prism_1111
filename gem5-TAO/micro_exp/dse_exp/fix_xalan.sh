#!/bin/bash
# Re-run xalancbmk with CORRECT checkpoint (537100000000)
# Fixes the mismatch where trace was from 537100000000 but ext sim
# warmup used 377600000000.
#
# Usage: bash fix_xalan.sh

DSE_DIR="/scratch/gem5-TAO/micro_exp/dse_exp"
TOOL="${DSE_DIR}/precompute_tool/dse_precompute"
BENCH="623.xalancbmk_s"

cd "$DSE_DIR"

echo "=== Fixing ${BENCH} ==="
echo "Started: $(date)"

# Verify correct simpletrace exists
ST="/scratch2/wl730/micro_ckpt/m5out_cpts/623.xalancbmk_s/537100000000/simpletrace.txt"
if [ ! -f "$ST" ]; then
    echo "ERROR: $ST not found"
    exit 1
fi
echo "Simpletrace: $(du -h $ST | cut -f1)"

# Check current field count
nf=$(head -1 traces/${BENCH}/slice_0/o3trace.txt | awk -F',' '{print NF}')
echo "Current o3trace fields: ${nf}"

# Remove ext_sim markers to force re-processing with correct checkpoint
find traces/${BENCH} -name ".ext_sim_done_*" -delete
echo "Cleared ext_sim markers"
echo ""

# Re-run for all 5 L1 configs
for l1 in 16 32 64 128 256; do
    echo "=== L1=${l1} === $(date)"

    # ext sim (appends new correct Cache_Latency + BPred_Dir at end;
    #          parser reads only last 2 fields, so this is correct)
    bash process_traces.sh ${BENCH} --parallel 28 --l1i $l1

    # precompute
    for start in 0 100 200 300; do
        $TOOL traces/${BENCH} precompute/l1_${l1}/${BENCH} \
            --verify 3 --slice-range $start $((start+99)) --l1i $l1 &
    done
    wait

    n=$(find precompute/l1_${l1}/${BENCH} -name "dse_precomputed.npy" | wc -l)
    exact=$(grep -h "EXACT" precompute/l1_${l1}/${BENCH}/slice_*.log 2>/dev/null | wc -l)
    echo "  L1=${l1}: ${n} slices, ${exact} EXACT"
    echo ""
done

echo "=== Done: $(date) ==="
echo "Verify field count after:"
head -1 traces/${BENCH}/slice_0/o3trace.txt | awk -F',' '{print "  fields:", NF}'
