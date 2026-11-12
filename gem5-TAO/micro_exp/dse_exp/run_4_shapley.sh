python shapley_pruning.py --benchmark 605.mcf_s --cos-threshold 0.3 --max-group-size 12 --n-permutations 2000 &
python shapley_pruning.py --benchmark 623.xalancbmk_s --cos-threshold 0.3 --max-group-size 12 --n-permutations 2000 &
python shapley_pruning.py --benchmark 641.leela_s --cos-threshold 0.3 --max-group-size 12 --n-permutations 2000 &
python shapley_pruning.py --benchmark 600.perlbench_s --cos-threshold 0.3 --max-group-size 12 --n-permutations 2000 &
wait
