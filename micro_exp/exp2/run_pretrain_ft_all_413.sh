cd /path/to/gem5/micro_exp/exp2
CKPT="/path/to/gem5/micro_exp/exp2/results_pretrain/pretrained.pt"
NORM="/path/to/gem5/micro_exp/exp2/results_pretrain/gem_normalizer.npz"
OUT="/path/to/gem5/micro_exp/exp5"

# 关掉early stopping
sed -i 's/FT_PATIENCE = 200/FT_PATIENCE = 99999/' train_exp5_pretrain_ft.py

for R in 0.50 0.60 0.70 0.80 0.90 1.00; do
    GPU=$(( RANDOM % 3 ))
    python train_exp5_pretrain_ft.py --gpu $GPU --ratio $R \
        --pretrain_ckpt $CKPT --pretrain_norm $NORM --output_dir $OUT &
done
wait

# 改回来
sed -i 's/FT_PATIENCE = 99999/FT_PATIENCE = 200/'train_exp5_pretrain_ft.py
echo "Done"
