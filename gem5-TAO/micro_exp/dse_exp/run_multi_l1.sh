#!/bin/bash
# Generate precompute for 4 L1 sizes: 32, 64, 128, 256
# L2 fixed at 1024KB
#
# Usage: nohup bash run_multi_l1.sh &> run_multi_l1.log &

# TOOL="/scratch/gem5-TAO/micro_exp/dse_exp/precompute_tool/dse_precompute"
# DSE_DIR="/scratch/gem5-TAO/micro_exp/dse_exp"
# TRACE_BASE="${DSE_DIR}/traces"
# PRE_BASE="${DSE_DIR}/precompute"

# L1_SIZES=(32 64 128 256)
# L2_SIZE=1024

# BENCHMARKS=$(ls "$TRACE_BASE")

# echo "=== Multi-L1 Precompute ==="
# echo "  L1 sizes: ${L1_SIZES[*]}"
# echo "  L2 size: ${L2_SIZE}"
# echo "  Benchmarks: $(echo $BENCHMARKS | wc -w)"
# echo "  Started: $(date)"
# echo ""

# for L1 in "${L1_SIZES[@]}"; do
#     echo "============================================================"
#     echo "=== L1 = ${L1}KB === $(date)"
#     echo "============================================================"

#     OUT_DIR="${PRE_BASE}/l1_${L1}"

#     # L1=64 already done — just move existing precompute
#     if [ "$L1" -eq 64 ]; then
#         echo "  L1=64: moving existing precompute to ${OUT_DIR}"
#         for bench in $BENCHMARKS; do
#             src="${PRE_BASE}/${bench}"
#             dst="${OUT_DIR}/${bench}"
#             if [ -d "$src" ] && [ ! -d "$dst" ]; then
#                 mv "$src" "$dst"
#                 echo "    moved ${bench}"
#             elif [ -d "$dst" ]; then
#                 echo "    ${bench} already in place"
#             fi
#         done
#         # Create parent dir structure
#         mkdir -p "$OUT_DIR"
#         continue
#     fi

#     # Step 1: Run ext sim with this L1 config
#     echo "  Step 1: ext sim (L1=${L1}, L2=${L2_SIZE})..."
#     for bench in $BENCHMARKS; do
#         trace_dir="${TRACE_BASE}/${bench}"
#         [ -d "$trace_dir" ] || continue
#         bash "${DSE_DIR}/process_traces.sh" "$bench" --parallel 28 \
#             --l1i "$L1" --l2 "$L2_SIZE" 2>&1 | tail -1
#     done

#     # Step 2: Precompute
#     echo "  Step 2: precompute..."
#     for bench in $BENCHMARKS; do
#         trace_dir="${TRACE_BASE}/${bench}"
#         [ -d "$trace_dir" ] || continue
#         for start in 0 100 200 300; do
#             $TOOL \
#                 "$trace_dir" \
#                 "${OUT_DIR}/${bench}" \
#                 --verify 1 --slice-range $start $((start+99)) \
#                 --l1i "$L1" --l2 "$L2_SIZE" &
#         done
#         wait
#         n=$(find "${OUT_DIR}/${bench}" -name "dse_precomputed.npy" 2>/dev/null | wc -l)
#         echo "    ${bench}: ${n} slices"
#     done
#     echo ""
# done

# # Summary
# echo "============================================================"
# echo "=== Summary ==="
# echo "============================================================"
# for L1 in "${L1_SIZES[@]}"; do
#     dir="${PRE_BASE}/l1_${L1}"
#     total=$(find "$dir" -name "dse_precomputed.npy" 2>/dev/null | wc -l)
#     size=$(du -sh "$dir" 2>/dev/null | cut -f1)
#     echo "  L1=${L1}KB: ${total} slices, ${size}"
# done
# echo ""
# echo "Total: $(du -sh ${PRE_BASE} | cut -f1)"
# echo "Finished: $(date)"

cd /scratch/gem5-TAO/micro_exp/dse_exp

# ═══ Step 1: L1=64 copy到新结构 ═══
# mkdir -p precompute/l1_64
# for bench in $(ls precompute/ | grep -v "^l1_"); do
#     [ -d "precompute/${bench}" ] && cp -r "precompute/${bench}" "precompute/l1_64/${bench}"
# done
# echo "L1=64: $(find precompute/l1_64 -name 'dse_precomputed.npy' | wc -l) slices"

# ═══ Step 2: 10个新benchmark的trace ═══
# for bench in 508.namd_r 526.blender_r 602.gcc_s 607.cactuBSSN_s 619.lbm_s 621.wrf_s 631.deepsjeng_s 638.imagick_s 644.nab_s 648.exchange2_s; do
#     echo "=== ${bench} === $(date)"
#     python prepare_base_features.py --benchmark $bench --workers 24
# done

# # ═══ Step 3: ext sim + precompute（4个L1） ═══
# TOOL="/scratch/gem5-TAO/micro_exp/dse_exp/precompute_tool/dse_precompute"
# for bench in 508.namd_r 526.blender_r 602.gcc_s 607.cactuBSSN_s 619.lbm_s 621.wrf_s 631.deepsjeng_s 638.imagick_s 644.nab_s 648.exchange2_s; do
#     for l1 in 32 64 128 256; do
#         bash process_traces.sh $bench --parallel 28 --l1i $l1
#         for start in 0 100 200 300; do
#             $TOOL traces/${bench} precompute/l1_${l1}/${bench} \
#                 --verify 1 --slice-range $start $((start+99)) --l1i $l1 &
#         done
#         wait
#     done
# done

# # ═══ Step 4: 检查 ═══
# for l1 in 32 64 128 256; do
#     total=$(find precompute/l1_${l1} -name "dse_precomputed.npy" 2>/dev/null | wc -l)
#     echo "L1=${l1}: ${total} slices"
# done

cd /scratch/gem5-TAO/micro_exp/dse_exp
TOOL="/scratch/gem5-TAO/micro_exp/dse_exp/precompute_tool/dse_precompute"

MISSING="508.namd_r 526.blender_r 602.gcc_s 607.cactuBSSN_s 619.lbm_s 621.wrf_s 631.deepsjeng_s 638.imagick_s 644.nab_s 648.exchange2_s"

# for l1 in 32 128 256; do
#     echo "=== L1=${l1} === $(date)"
    
#     # ext sim
#     for bench in $MISSING; do
#         bash process_traces.sh $bench --parallel 28 --l1i $l1
#     done
    
#     # precompute
#     for bench in $MISSING; do
#         for start in 0 100 200 300; do
#             $TOOL traces/${bench} precompute/l1_${l1}/${bench} \
#                 --verify 1 --slice-range $start $((start+99)) --l1i $l1 &
#         done
#         wait
#         n=$(find precompute/l1_${l1}/${bench} -name "dse_precomputed.npy" 2>/dev/null | wc -l)
#         echo "  ${bench}: ${n}"
#     done
# done

cd /scratch/gem5-TAO/micro_exp/dse_exp
TOOL="/scratch/gem5-TAO/micro_exp/dse_exp/precompute_tool/dse_precompute"
ALL_BENCH=$(ls traces/)

for bench in $ALL_BENCH; do
    bash process_traces.sh $bench --parallel 28 --l1i 16
done

for bench in $ALL_BENCH; do
    for start in 0 100 200 300; do
        $TOOL traces/${bench} precompute/l1_16/${bench} \
            --verify 1 --slice-range $start $((start+99)) --l1i 16 &
    done
    wait
    n=$(find precompute/l1_16/${bench} -name "dse_precomputed.npy" 2>/dev/null | wc -l)
    echo "${bench}: ${n}"
done
