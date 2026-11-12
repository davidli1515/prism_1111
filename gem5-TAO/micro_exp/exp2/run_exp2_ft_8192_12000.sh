cd /scratch/gem5-TAO/micro_exp/exp2

for N in 8192 12000; do
    for bm in $(ls /scratch2/wl730/micro_traindata_packed_all/gem/*.npy | xargs -n1 basename | sed 's/.npy//'); do
        GPU=$(( RANDOM % 3 ))
        python train_exp2_finetune_serverB.py --gpu $GPU --target $bm --n_samples $N &
        python train_exp2_finetune_mix_serverB.py --gpu $GPU --target $bm --n_samples $N &
    done
    wait
    echo "N=$N done: $(date)"
done
echo "All done"
