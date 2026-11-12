cd /scratch/gem5-TAO/micro_exp/exp4

python grad_projection_semi.py --mode full --gpu 0 --epochs 50 > logs/proj_full.log 2>&1 &
python grad_projection_semi.py --mode semi --gpu 1 --ratio 0.05 --epochs 50 > logs/proj_semi.log 2>&1 &
wait

# 画图
python plot_grad_projection.py --full grad_projection_full_100pct.json --semi grad_projection_semi_5pct.json
