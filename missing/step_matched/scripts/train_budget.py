"""Labeled-only training with a matched step budget.
--kind concorde|soft|hard ; batch = round(50000*ratio) -> 4 steps/epoch at 5-50% (6084 steps at 1521 epochs);
--epochs E: LR halved at FIXED_MILESTONES_18K * E/1521 (1521 -> [2555,3528,4563,5597]).
soft = V2 (train_exp4_lambda_sens --variant v2), hard = shared-Base MSCS, concorde = GPU Concorde k1.
Usage: python train_budget.py --kind hard --ratio 0.05 --epochs 1521 --seed 42 --gpu 0 --out DIR
"""
import sys
sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp4"); sys.path.insert(1, "/scratch/gem5-TAO/micro_exp/exp1")
SP = "/tmp/claude-1017/-scratch-gem5-TAO/90b1d870-5e44-4246-a121-ddc22036bb0a/scratchpad"; sys.path.insert(2, SP)

def pop(flag):
    i = sys.argv.index(flag); v = sys.argv[i + 1]; del sys.argv[i:i + 2]; return v

kind = pop("--kind")
a = sys.argv
assert "--out" in a
ratio = float(a[a.index("--ratio") + 1]); E = int(a[a.index("--epochs") + 1])
bs = int(round(50000 * ratio))
ms = [int(round(m * E / 1521)) for m in [2555, 3528, 4563, 5597]]
if kind == "concorde":
    import train_exp4_concorde_multihead as M
    M.BATCH_SIZE = bs; M.FIXED_MILESTONES_18K = ms
    sys.argv += ["--heads", "1", "--unlab", "none"]
    M.main()
else:
    import train_exp4_lambda_sens as T
    T.BATCH_SIZE = bs; T.FIXED_MILESTONES_18K = ms
    if kind == "hard":
        from train_hard_mscs import HardMSCSModel
        T.GEMModel = HardMSCSModel
    sys.argv += ["--variant", "v2", "--lam-task", "10", "--lam-const", "1", "--lam-mono", "0", "--input", "concorde"]
    T.main()
