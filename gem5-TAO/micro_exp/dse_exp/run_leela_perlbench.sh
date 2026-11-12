# # 生成o3trace（每个约2-4小时，24 workers）
# python prepare_base_features.py --benchmark 641.leela_s --workers 24
# python prepare_base_features.py --benchmark 600.perlbench_s --workers 24

# # ext sim处理
# bash process_traces.sh 641.leela_s --parallel 28
# bash process_traces.sh 600.perlbench_s --parallel 28

# # precompute
# TOOL="/scratch/gem5-TAO/micro_exp/dse_exp/precompute_tool/dse_precompute"
# for bench in 641.leela_s 600.perlbench_s; do
#     for start in 0 100 200 300; do
#         $TOOL \
#             /scratch/gem5-TAO/micro_exp/dse_exp/traces/${bench} \
#             /scratch/gem5-TAO/micro_exp/dse_exp/precompute/${bench} \
#             --verify 5 --slice-range $start $((start+99)) &
#     done
# done
# wait

# # 检查
# for bench in 641.leela_s 600.perlbench_s; do
#     dir="/scratch/gem5-TAO/micro_exp/dse_exp/precompute/${bench}"
#     count=$(find "$dir" -name "dse_precomputed.npy" | wc -l)
#     echo "${bench}: ${count}/400"
# done

# ext sim处理
bash process_traces.sh 600.perlbench_s --parallel 28

# precompute
TOOL="/scratch/gem5-TAO/micro_exp/dse_exp/precompute_tool/dse_precompute"
for bench in 600.perlbench_s; do
    for start in 0 100 200 300; do
        $TOOL \
            /scratch/gem5-TAO/micro_exp/dse_exp/traces/${bench} \
            /scratch/gem5-TAO/micro_exp/dse_exp/precompute/${bench} \
            --verify 5 --slice-range $start $((start+99)) &
    done
done
wait

# 检查
for bench in 600.perlbench_s; do
    dir="/scratch/gem5-TAO/micro_exp/dse_exp/precompute/${bench}"
    count=$(find "$dir" -name "dse_precomputed.npy" | wc -l)
    echo "${bench}: ${count}/400"
done