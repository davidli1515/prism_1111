#!/bin/bash
RESULT_DIR="result_shapley_520"
PERMS=200
THRESHOLD=0.3
PARALLEL=4

cd /scratch/gem5-TAO/micro_exp/dse_exp

BENCHMARKS=(
    "508.namd_r" "526.blender_r" "600.perlbench_s" "602.gcc_s" "605.mcf_s"
    "607.cactuBSSN_s" "619.lbm_s" "620.omnetpp_s" "621.wrf_s" "623.xalancbmk_s"
    "631.deepsjeng_s" "638.imagick_s" "641.leela_s" "644.nab_s"
    "648.exchange2_s" "654.roms_s" "657.xz_s"
)

# for bench in "${BENCHMARKS[@]}"; do mkdir -p "${RESULT_DIR}/${bench}"; done

# echo "========================================"
# echo "Full MC + Pruned + Ablation — ${#BENCHMARKS[@]} benchmarks"
# echo "Result dir: $RESULT_DIR"
# echo "========================================"

# # Step 1: Full MC + Pruned
# for ((i=0; i<${#BENCHMARKS[@]}; i+=PARALLEL)); do
#     batch=("${BENCHMARKS[@]:i:PARALLEL}")
#     echo "--- Step1 Batch $((i/PARALLEL + 1)): ${batch[*]} ---"
#     for bench in "${batch[@]}"; do
#         python shapley_pruning.py \
#             --benchmark "$bench" \
#             --cos-threshold $THRESHOLD \
#             --n-permutations $PERMS \
#             --output-dir "$RESULT_DIR" \
#             > "${RESULT_DIR}/${bench}/shapley_log.txt" 2>&1 &
#     done
#     wait
# done

# echo ""
# echo "Step 1 Summary:"
# for bench in "${BENCHMARKS[@]}"; do
#     bname=$(echo "$bench" | sed 's/.*\.\(.*\)_[sr]/\1/')
#     json="${RESULT_DIR}/${bench}/${bname}_shapley_results.json"
#     if [ -f "$json" ]; then
#         rho=$(python3 -c "import json; d=json.load(open('$json')); print(f\"{d['metrics']['pruned']['rho']:.3f}\")" 2>/dev/null)
#         ef=$(python3 -c "import json; d=json.load(open('$json')); print(d['metrics']['full']['evals'])" 2>/dev/null)
#         ep=$(python3 -c "import json; d=json.load(open('$json')); print(d['metrics']['pruned']['evals'])" 2>/dev/null)
#         echo "  $bench: ρ=$rho, full=$ef, pruned=$ep"
#     else
#         echo "  $bench: FAILED"
#     fi
# done

# Step 2: Ablation (matched-budget MC)
echo ""
echo "Step 2: Matched-budget Ablation"
for ((i=0; i<${#BENCHMARKS[@]}; i+=PARALLEL)); do
    batch=("${BENCHMARKS[@]:i:PARALLEL}")
    echo "--- Step2 Batch $((i/PARALLEL + 1)): ${batch[*]} ---"
    for bench in "${batch[@]}"; do
        python ablation_mc_vs_pruned.py \
            --benchmark "$bench" \
            > "${RESULT_DIR}/${bench}/ablation_log.txt" 2>&1 &
    done
    wait
done

# Final Summary
echo ""
echo "================================================================================"
echo "  Final Summary: Pruned vs Matched-budget MC"
echo "================================================================================"
printf "  %-24s %8s %8s %8s %8s %8s %8s\n" \
    "Benchmark" "Prun ρ" "Prun E" "MC ρ" "MC E" "Full ρ" "Full E"
echo "  ─────────────────────────────────────────────────────────────────────────────────"

WINS=0
TOTAL=0
for bench in "${BENCHMARKS[@]}"; do
    bname=$(echo "$bench" | sed 's/.*\.\(.*\)_[sr]/\1/')
    json="${RESULT_DIR}/${bench}/${bname}_shapley_results.json"
    ablation="${RESULT_DIR}/${bench}/${bname}_ablation_matched.json"
    
    if [ -f "$json" ] && [ -f "$ablation" ]; then
        row=$(python3 -c "
import json
d = json.load(open('$json'))
a = json.load(open('$ablation'))
rp = d['metrics']['pruned']['rho']
ep = d['metrics']['pruned']['evals']
rf = d['metrics']['full']['rho']
ef = d['metrics']['full']['evals']
rm = a['matched_mc']['rho']
em = a['matched_mc']['evals']
win = 'W' if rp > rm else 'L'
print(f\"  {\"$bench\":<24s} {rp:8.3f} {ep:8d} {rm:8.3f} {em:8d} {rf:8.3f} {ef:8d}  {win}\")
" 2>/dev/null)
        echo "$row"
        
        is_win=$(python3 -c "
import json
a = json.load(open('$ablation'))
print('1' if a['pruned_wins'] else '0')
" 2>/dev/null)
        WINS=$((WINS + is_win))
        TOTAL=$((TOTAL + 1))
    else
        echo "  $bench: MISSING"
    fi
done

echo "  ─────────────────────────────────────────────────────────────────────────────────"
echo "  Pruned wins: ${WINS}/${TOTAL} benchmarks"
echo ""
echo "All done! Results in ${RESULT_DIR}/"