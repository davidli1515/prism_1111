cd /path/to/gem5/micro_exp/exp4

python train_exp4_all.py --dataset 40k --ablation supervised_only --ratio 0.25 --gpu 0 > logs_40k/supervised_only_0.25.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation supervised_only --ratio 0.30 --gpu 1 > logs_40k/supervised_only_0.30.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation supervised_only --ratio 0.40 --gpu 2 > logs_40k/supervised_only_0.40.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation supervised_only --ratio 0.50 --gpu 0 > logs_40k/supervised_only_0.50.log 2>&1 &
wait
