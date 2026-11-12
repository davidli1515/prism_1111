# 1. 对已经处理过的slice（字段≥72的），手动创建marker文件，避免重复处理
for bench in 605.mcf_s 623.xalancbmk_s; do
    dir="/scratch/gem5-TAO/micro_exp/dse_exp/traces/${bench}"
    for s in $(seq 0 399); do
        f="${dir}/slice_${s}/o3trace.txt"
        [ -f "$f" ] || continue
        nf=$(head -1 "$f" | awk -F',' '{print NF}')
        if [ "$nf" -ge 72 ]; then
            touch "${dir}/slice_${s}/.ext_sim_done_64_1024_tage"
        fi
    done
    # 统计
    done_count=$(find "$dir" -name ".ext_sim_done_64_1024_tage" | wc -l)
    echo "${bench}: ${done_count}/400 already done"
done

# 2. 处理剩余未处理的
bash process_traces.sh 605.mcf_s --parallel 28
bash process_traces.sh 623.xalancbmk_s --parallel 28

# 确认全部完成
for bench in 605.mcf_s 623.xalancbmk_s; do
    dir="/scratch/gem5-TAO/micro_exp/dse_exp/traces/${bench}"
    count=$(find "$dir" -name ".ext_sim_done_64_1024_tage" | wc -l)
    echo "${bench}: ${count}/400 traces processed"
done

# 第二步：等全部400/400后，再跑precompute
bash run_precompute.sh 28 5