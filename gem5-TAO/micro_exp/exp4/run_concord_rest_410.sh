cd /scratch/gem5-TAO/micro_exp/exp4

python train_exp4_all.py --dataset 40k --ablation concorde --ratio 0.60 --gpu 0 > logs_40k/concorde_0.60.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation concorde --ratio 0.70 --gpu 1 > logs_40k/concorde_0.70.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation concorde --ratio 0.80 --gpu 2 > logs_40k/concorde_0.80.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation concorde --ratio 0.90 --gpu 0 > logs_40k/concorde_0.90.log 2>&1 &
wait
