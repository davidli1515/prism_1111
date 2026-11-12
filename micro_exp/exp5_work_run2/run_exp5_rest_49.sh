cd /path/to/gem5/micro_exp/exp2

python train_exp5_pretrain_ft.py --gpu 0 --ratio 0.60 &
python train_exp5_pretrain_ft.py --gpu 1 --ratio 0.70 &
python train_exp5_pretrain_ft.py --gpu 2 --ratio 0.80 &
python train_exp5_pretrain_ft.py --gpu 0 --ratio 0.90 &
wait
echo "408k done"
