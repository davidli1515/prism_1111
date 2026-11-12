# ============================================================
# Server B (408k) - run on server B
# ============================================================
 
# cd /scratch/gem5-TAO/micro_exp/exp2
 
# # Step 1: Pretrain (single job, ~30min)
# python train_exp2_pretrain.py --gpu 0
# echo "ServerB pretrain done: $(date)"
 
# # Step 2: FT 12 ratios, 6 parallel per round (~2h)
# CKPT="/scratch/gem5-TAO/micro_exp/exp2/results_pretrain/pretrained.pt"
# NORM="/scratch/gem5-TAO/micro_exp/exp2/results_pretrain/gem_normalizer.npz"
# OUT="/scratch/gem5-TAO/micro_exp/exp5"
 
# for R in 0.05 0.15 0.20 0.25 0.30 0.40; do
#     GPU=$(( RANDOM % 3 ))
#     python train_exp5_pretrain_ft.py --gpu $GPU --ratio $R \
#         --pretrain_ckpt $CKPT --pretrain_norm $NORM --output_dir $OUT &
# done
# wait
# echo "ServerB round 1 done: $(date)"
 
# for R in 0.50 0.60 0.70 0.80 0.90 1.00; do
#     GPU=$(( RANDOM % 3 ))
#     python train_exp5_pretrain_ft.py --gpu $GPU --ratio $R \
#         --pretrain_ckpt $CKPT --pretrain_norm $NORM --output_dir $OUT &
# done
# wait
# echo "ServerB all done: $(date)"


cd /scratch/gem5-TAO/micro_exp/exp5

CKPT="/scratch/gem5-TAO/micro_exp/exp2/results_pretrain/pretrained.pt"
NORM="/scratch/gem5-TAO/micro_exp/exp2/results_pretrain/gem_normalizer.npz"

for R in 0.05 0.15 0.20 0.25 0.30 0.40; do
    GPU=$(( RANDOM % 3 ))
    python /scratch/gem5-TAO/micro_exp/exp2/train_exp5_pretrain_ft.py \
        --gpu $GPU --ratio $R --pretrain_ckpt $CKPT --pretrain_norm $NORM --output_dir . &
done
wait
for R in 0.50 0.60 0.70 0.80 0.90 1.00; do
    GPU=$(( RANDOM % 3 ))
    python /scratch/gem5-TAO/micro_exp/exp2/train_exp5_pretrain_ft.py \
        --gpu $GPU --ratio $R --pretrain_ckpt $CKPT --pretrain_norm $NORM --output_dir . &
done
wait
echo "Done"