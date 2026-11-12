cd /path/to/gem5/micro_exp/exp4

# Wave 1: full 60-90% (8 parallel)
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.60 --gpu 0 > logs_18k/full_0.60.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.70 --gpu 0 > logs_18k/full_0.70.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.80 --gpu 1 > logs_18k/full_0.80.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation full --ratio 0.90 --gpu 1 > logs_18k/full_0.90.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation full --ratio 0.60 --gpu 2 > logs_40k/full_0.60.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation full --ratio 0.70 --gpu 2 > logs_40k/full_0.70.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation full --ratio 0.80 --gpu 3 > logs_40k/full_0.80.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation full --ratio 0.90 --gpu 3 > logs_40k/full_0.90.log 2>&1 &
wait; echo "Wave 1 done."

# Wave 2: supervised_only 60-90% (8 parallel)
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.60 --gpu 0 > logs_18k/supervised_only_0.60.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.70 --gpu 0 > logs_18k/supervised_only_0.70.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.80 --gpu 1 > logs_18k/supervised_only_0.80.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation supervised_only --ratio 0.90 --gpu 1 > logs_18k/supervised_only_0.90.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation supervised_only --ratio 0.60 --gpu 2 > logs_40k/supervised_only_0.60.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation supervised_only --ratio 0.70 --gpu 2 > logs_40k/supervised_only_0.70.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation supervised_only --ratio 0.80 --gpu 3 > logs_40k/supervised_only_0.80.log 2>&1 &
python train_exp4_all.py --dataset 40k --ablation supervised_only --ratio 0.90 --gpu 3 > logs_40k/supervised_only_0.90.log 2>&1 &
wait; echo "Wave 2 done."

# Wave 3: vanilla 60-90% (8 parallel)
python train_exp4_vanilla.py --dataset 18k --ratio 0.60 --gpu 0 > logs_18k/vanilla_0.60.log 2>&1 &
python train_exp4_vanilla.py --dataset 18k --ratio 0.70 --gpu 0 > logs_18k/vanilla_0.70.log 2>&1 &
python train_exp4_vanilla.py --dataset 18k --ratio 0.80 --gpu 1 > logs_18k/vanilla_0.80.log 2>&1 &
python train_exp4_vanilla.py --dataset 18k --ratio 0.90 --gpu 1 > logs_18k/vanilla_0.90.log 2>&1 &
python train_exp4_vanilla.py --dataset 40k --ratio 0.60 --gpu 2 > logs_40k/vanilla_0.60.log 2>&1 &
python train_exp4_vanilla.py --dataset 40k --ratio 0.70 --gpu 2 > logs_40k/vanilla_0.70.log 2>&1 &
python train_exp4_vanilla.py --dataset 40k --ratio 0.80 --gpu 3 > logs_40k/vanilla_0.80.log 2>&1 &
python train_exp4_vanilla.py --dataset 40k --ratio 0.90 --gpu 3 > logs_40k/vanilla_0.90.log 2>&1 &
wait; echo "Wave 3 done."

echo "All done."
