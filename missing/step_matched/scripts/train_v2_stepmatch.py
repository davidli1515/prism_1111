"""
Step-matched labeled-only control: train_exp4_lambda_sens.py unchanged (original GEMModel, V2),
except batch = 50000 * ratio. With labeled rows only this gives 4 steps/epoch = 6084 steps,
the same LR milestones, the same number of labeled rows per step and the same 1521 passes over
the labeled rows as V3; the only difference from V3 is that no unlabeled rows are in the batch.

Usage: python train_v2_stepmatch.py --variant v2 --ratio 0.05 --lam-task 10 --lam-const 1 \
           --lam-mono 0 --input concorde --seed 42 --gpu 0 --out DIR
"""
import sys
sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp4")
sys.path.insert(1, "/scratch/gem5-TAO/micro_exp/exp1")
import train_exp4_lambda_sens as T

if __name__ == "__main__":
    a = sys.argv
    assert "--out" in a and a[a.index("--variant") + 1] == "v2"
    T.BATCH_SIZE = int(round(50000 * float(a[a.index("--ratio") + 1])))
    T.main()
