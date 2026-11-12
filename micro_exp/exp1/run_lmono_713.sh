cd /path/to/gem5/micro_exp/exp1
sed -i 's|^OUTPUT_DIR.*|OUTPUT_DIR = "./results/exp1_v3_run2"|' config.py
python train_exp1_512_256.py --gpu 0 --variant gem_nod3 --small &
python train_exp1_256_128.py --gpu 1 --variant gem_nod3 --small &