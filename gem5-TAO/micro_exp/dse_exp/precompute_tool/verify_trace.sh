# # 两个workload同时跑，共8路
# for bench in 605.mcf_s 623.xalancbmk_s; do
#     for start in 0 100 200 300; do
#         ./dse_precompute \
#             /scratch/gem5-TAO/micro_exp/dse_exp/traces/${bench} \
#             /scratch/gem5-TAO/micro_exp/dse_exp/precompute/${bench} \
#             --verify 5 --slice-range $start $((start+99)) &
#     done
# done
# wait
# echo "All done"


# for bench in 605.mcf_s 623.xalancbmk_s; do
#     dir="/scratch/gem5-TAO/micro_exp/dse_exp/precompute/${bench}"
#     count=$(find "$dir" -name "dse_precomputed.npy" | wc -l)
#     exact=$(grep -rh "EXACT" ${dir}/slice_*.log 2>/dev/null | wc -l)
#     check=$(grep -rh "CHECK" ${dir}/slice_*.log 2>/dev/null | wc -l)
#     size=$(du -sh "$dir" | cut -f1)
#     echo "${bench}: ${count}/400 precomputed, EXACT=${exact} CHECK=${check}, size=${size}"
# done

for bench in 605.mcf_s 623.xalancbmk_s; do
    dir="/scratch/gem5-TAO/micro_exp/dse_exp/precompute/${bench}"
    echo "${bench}:"
    for s in $(seq 0 399); do
        [ -f "${dir}/slice_${s}/dse_precomputed.npy" ] && printf "%d " $s
    done
    echo ""
done
```