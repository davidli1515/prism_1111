cd /scratch/gem5-TAO/micro_exp/exp2
mkdir -p results

# 4 GPU并行（假设17个benchmark，分5批）
# 自动获取benchmark列表
BMS=($(ls /scratch2/wl730/micro_traindata_packed/gem/*.npy | xargs -n1 basename | sed 's/.npy//'))

# 分批跑
for ((i=0; i<${#BMS[@]}; i+=4)); do
    for j in 0 1 2 3; do
        idx=$((i+j))
        [ $idx -ge ${#BMS[@]} ] && continue
        echo "Launching ${BMS[$idx]} on GPU $j"
        python train_exp2.py --gpu $j --target ${BMS[$idx]} &
    done
    wait
done

# 跑完后收集结果+画图
python plot_exp2.py --collect --dir results/
