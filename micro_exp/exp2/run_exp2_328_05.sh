#!/bin/bash
cd /path/to/gem5/micro_exp/exp2
BMS=($(ls /data/micro_traindata_packed_all/gem/*.npy | xargs -n1 basename | sed 's/.npy//'))

for ((i=0; i<${#BMS[@]}; i+=6)); do
    for j in 0 1 2 3 4 5; do
        idx=$((i+j))
        [ $idx -ge ${#BMS[@]} ] && continue
        GPU=$((j % 3))
        echo "Launching ${BMS[$idx]} on GPU $GPU"
        python train_exp2.py --gpu $GPU --target "${BMS[$idx]}" &
    done
    wait
    echo "Batch $((i/6+1)) done"
done

python plot_exp2.py --collect --dir results_v3/