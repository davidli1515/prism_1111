cd /scratch/gem5-TAO/micro_exp/exp4

# Wave 1: 5-50% + 60-70% (8 parallel)
python train_exp4_all.py --dataset 18k --ablation concorde --ratio 0.05 --gpu 0 > logs_18k/concorde_0.05.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation concorde --ratio 0.15 --gpu 0 > logs_18k/concorde_0.15.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation concorde --ratio 0.20 --gpu 1 > logs_18k/concorde_0.20.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation concorde --ratio 0.25 --gpu 1 > logs_18k/concorde_0.25.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation concorde --ratio 0.30 --gpu 2 > logs_18k/concorde_0.30.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation concorde --ratio 0.40 --gpu 2 > logs_18k/concorde_0.40.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation concorde --ratio 0.50 --gpu 3 > logs_18k/concorde_0.50.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation concorde --ratio 0.60 --gpu 3 > logs_18k/concorde_0.60.log 2>&1 &
wait

# Wave 2: 70-90% (3 runs)
python train_exp4_all.py --dataset 18k --ablation concorde --ratio 0.70 --gpu 0 > logs_18k/concorde_0.70.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation concorde --ratio 0.80 --gpu 1 > logs_18k/concorde_0.80.log 2>&1 &
python train_exp4_all.py --dataset 18k --ablation concorde --ratio 0.90 --gpu 2 > logs_18k/concorde_0.90.log 2>&1 &
wait

echo "Done. Concorde arch: [256, 128]"
