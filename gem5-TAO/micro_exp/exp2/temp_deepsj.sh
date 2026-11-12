cd /scratch/gem5-TAO/micro_exp/exp2

for N in 32 256 512 1024 2048 4096; do
    python train_exp2_finetune_serverB.py --gpu 0 --target 631.deepsjeng_s --n_samples $N &
    python train_exp2_finetune_mix_serverB.py --gpu 1 --target 631.deepsjeng_s --n_samples $N &
done
wait
echo "Done"
