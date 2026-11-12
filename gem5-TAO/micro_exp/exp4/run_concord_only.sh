cd /scratch/gem5-TAO/micro_exp/exp4

python train_exp4_all.py --ablation concorde --ratio 0.05 --gpu 0 > logs_v2/con_05.log 2>&1 &
python train_exp4_all.py --ablation concorde --ratio 0.10 --gpu 1 > logs_v2/con_10.log 2>&1 &
python train_exp4_all.py --ablation concorde --ratio 0.15 --gpu 2 > logs_v2/con_15.log 2>&1 &
wait

python train_exp4_all.py --ablation concorde --ratio 0.20 --gpu 0 > logs_v2/con_20.log 2>&1 &
python train_exp4_all.py --ablation concorde --ratio 0.25 --gpu 1 > logs_v2/con_25.log 2>&1 &
python train_exp4_all.py --ablation concorde --ratio 0.30 --gpu 2 > logs_v2/con_30.log 2>&1 &
wait

python train_exp4_all.py --ablation concorde --ratio 0.40 --gpu 0 > logs_v2/con_40.log 2>&1 &
python train_exp4_all.py --ablation concorde --ratio 0.50 --gpu 1 > logs_v2/con_50.log 2>&1 &
wait

# python train_exp4_all.py --ablation concorde --ratio 0.60 --gpu 0 > logs_v2/con_60.log 2>&1 &
# python train_exp4_all.py --ablation concorde --ratio 0.70 --gpu 1 > logs_v2/con_70.log 2>&1 &
# python train_exp4_all.py --ablation concorde --ratio 0.80 --gpu 2 > logs_v2/con_80.log 2>&1 &
# wait

# python train_exp4_all.py --ablation concorde --ratio 0.90 --gpu 0 > logs_v2/con_90.log 2>&1 &
# wait

echo "Done."
