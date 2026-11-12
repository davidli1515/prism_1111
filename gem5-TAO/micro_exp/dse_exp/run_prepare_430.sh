cd /scratch/gem5-TAO/micro_exp/dse_exp

# 注意：xz_s需要32GB/进程，24 workers × 32GB = 768GB > 你的350GB
# xz_s用8 workers跑
python prepare_base_features.py --benchmark 657.xz_s --workers 8
python prepare_base_features.py --benchmark 654.roms_s --workers 16
python prepare_base_features.py --benchmark 620.omnetpp_s --workers 24

# ext sim
bash process_traces.sh 657.xz_s --parallel 28
bash process_traces.sh 654.roms_s --parallel 28
bash process_traces.sh 620.omnetpp_s --parallel 28

# precompute
TOOL="/scratch/gem5-TAO/micro_exp/dse_exp/precompute_tool/dse_precompute"
for bench in 657.xz_s 654.roms_s 620.omnetpp_s; do
    for start in 0 100 200 300; do
        $TOOL \
            /scratch/gem5-TAO/micro_exp/dse_exp/traces/${bench} \
            /scratch/gem5-TAO/micro_exp/dse_exp/precompute/${bench} \
            --verify 3 --slice-range $start $((start+99)) &
    done
    wait
done

# 检查
for bench in 657.xz_s 654.roms_s 620.omnetpp_s; do
    dir="/scratch/gem5-TAO/micro_exp/dse_exp/precompute/${bench}"
    count=$(find "$dir" -name "dse_precomputed.npy" | wc -l)
    echo "${bench}: ${count}/400"
done
