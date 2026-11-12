#!/bin/bash
# Run all 6 ablation variants in parallel on 4 GPUs
# Usage: nohup bash run_exp1_ablation.sh &> ablation.log &

# cd /scratch/gem5-TAO/micro_exp/exp1

# echo "Start: $(date)"
# echo "Running 3 noD3 variants in parallel"
# python train_exp1_ablation.py --gpu 0 --variant nod3_vanilla &
# python train_exp1_ablation.py --gpu 1 --variant nod3_mono &
# python train_exp1_ablation.py --gpu 2 --variant nod3_const &
# wait

# echo "End: $(date)"
# echo "Summary:"
# python -c "
# import json, os
# ablation_dir = 'results/exp1/ablation'
# order = ['d3_const','d3_mono','d3_vanilla','nod3_const','nod3_mono','nod3_vanilla']
# labels = {
#     'd3_const':'D3 + L_task+L_const', 'd3_mono':'D3 + L_task+L_mono', 'd3_vanilla':'D3 + L_task',
#     'nod3_const':'noD3 + L_task+L_const', 'nod3_mono':'noD3 + L_task+L_mono', 'nod3_vanilla':'noD3 + L_task',
# }
# print(f\"{'Variant':<28} {'MAPE':>8}\")
# print('-'*38)
# for n in order:
#     p = os.path.join(ablation_dir, f'{n}.json')
#     if os.path.exists(p):
#         r = json.load(open(p))
#         print(f\"  {labels[n]:<26} {r['overall_mape']:>7.2f}%\")
#     else:
#         print(f\"  {labels[n]:<26}     ...\")
# "
# echo "All done!"

#!/bin/bash
# Run all 6 ablation variants in parallel on 4 GPUs
# Usage: nohup bash run_exp1_ablation.sh &> ablation.log &

cd /scratch/gem5-TAO/micro_exp/exp1

# echo "Start: $(date)"
echo "Batch 1: 3 D3 variants"
# python train_exp1_ablation.py --gpu 0 --variant d3_vanilla &
# python train_exp1_ablation.py --gpu 1 --variant d3_mono &
python train_exp1_ablation.py --gpu 1 --variant d3_const &
wait

# echo "Batch 2: 3 noD3 variants"
# python train_exp1_ablation.py --gpu 0 --variant nod3_vanilla &
# python train_exp1_ablation.py --gpu 1 --variant nod3_mono &
# python train_exp1_ablation.py --gpu 2 --variant nod3_const &
# wait

echo "End: $(date)"
echo "Summary:"
python -c "
import json, os
ablation_dir = 'results/exp1_v2/ablation'
order = ['d3_const','d3_mono','d3_vanilla','nod3_const','nod3_mono','nod3_vanilla']
labels = {
    'd3_const':'D3 + L_task+L_const', 'd3_mono':'D3 + L_task+L_mono', 'd3_vanilla':'D3 + L_task',
    'nod3_const':'noD3 + L_task+L_const', 'nod3_mono':'noD3 + L_task+L_mono', 'nod3_vanilla':'noD3 + L_task',
}
print(f\"{'Variant':<28} {'MAPE':>8}\")
print('-'*38)
for n in order:
    p = os.path.join(ablation_dir, f'{n}.json')
    if os.path.exists(p):
        r = json.load(open(p))
        print(f\"  {labels[n]:<26} {r['overall_mape']:>7.2f}%\")
    else:
        print(f\"  {labels[n]:<26}     ...\")
"
echo "All done!"