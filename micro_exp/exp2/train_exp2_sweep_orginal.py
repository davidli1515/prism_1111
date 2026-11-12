"""
Experiment 2 Sweep: MAPE vs. Target Train Ratio
  For each target benchmark, vary the train ratio: 5%, 10%, 15%, 20%, 50%
  Compare GEM+D3, GEM-noD3, Concorde

Usage:
    python train_exp2_sweep.py --gpu 0                       # all benchmarks
    python train_exp2_sweep.py --gpu 0 --target 508.namd_r   # single benchmark
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")
from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM,
    CONCORDE_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE,
    CONCORDE_LR, CONCORDE_WEIGHT_DECAY,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    SEED,
)
from model import (
    GEMModel, ConcordeModel,
    compute_cpi_mean, loss_task, loss_const, loss_mono,
)
from dataset import (
    load_all_benchmarks, FeatureNormalizer, GEMDataset,
)

EXP2_SWEEP_DIR = "/path/to/gem5/micro_exp/exp2/results_sweep"
TRAIN_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

# Use same epochs/patience as main experiments for fairness
SWEEP_GEM_EPOCHS = NUM_EPOCHS       # 300
SWEEP_CON_EPOCHS = NUM_EPOCHS       # 300 (not 1521 — use same small batch as GEM)
SWEEP_PATIENCE = PATIENCE           # 30


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


# ============================================================
# Data Split
# ============================================================

def create_ood_split(all_data, target_name, train_ratio, batch_size=BATCH_SIZE, seed=SEED):
    rng = np.random.RandomState(seed)
    train_gem, train_con, train_mscs, train_cpi, train_wmin = [], [], [], [], []
    test_gem, test_con, test_mscs, test_cpi, test_wmin = [], [], [], [], []

    for name, data in sorted(all_data.items()):
        if name == target_name:
            n = data["n_samples"]
            idx = rng.permutation(n)
            split = max(1, int(n * train_ratio))
            tr_idx, te_idx = idx[:split], idx[split:]

            train_gem.append(data["gem_features"][tr_idx])
            train_con.append(data["concord_features"][tr_idx])
            train_mscs.append(data["mscs_target"][tr_idx])
            train_cpi.append(data["cpi_golden"][tr_idx])
            train_wmin.append(data["w_min"][tr_idx])

            test_gem.append(data["gem_features"][te_idx])
            test_con.append(data["concord_features"][te_idx])
            test_mscs.append(data["mscs_target"][te_idx])
            test_cpi.append(data["cpi_golden"][te_idx])
            test_wmin.append(data["w_min"][te_idx])
        else:
            train_gem.append(data["gem_features"])
            train_con.append(data["concord_features"])
            train_mscs.append(data["mscs_target"])
            train_cpi.append(data["cpi_golden"])
            train_wmin.append(data["w_min"])

    train_gem  = np.concatenate(train_gem)
    train_con  = np.concatenate(train_con)
    train_mscs = np.concatenate(train_mscs)
    train_cpi  = np.concatenate(train_cpi)
    train_wmin = np.concatenate(train_wmin)
    test_gem   = np.concatenate(test_gem)
    test_con   = np.concatenate(test_con)
    test_mscs  = np.concatenate(test_mscs)
    test_cpi   = np.concatenate(test_cpi)
    test_wmin  = np.concatenate(test_wmin)

    gem_norm = FeatureNormalizer()
    train_gem = gem_norm.fit_transform(train_gem)
    test_gem  = gem_norm.transform(test_gem)
    con_norm = FeatureNormalizer()
    train_con = con_norm.fit_transform(train_con)
    test_con  = con_norm.transform(test_con)

    train_ds = GEMDataset(train_gem, train_con, train_mscs, train_cpi, train_wmin)
    test_ds  = GEMDataset(test_gem, test_con, test_mscs, test_cpi, test_wmin)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    n_target_train = max(1, int(all_data[target_name]["n_samples"] * train_ratio))
    return train_loader, test_loader, n_target_train


# ============================================================
# Evaluate & Train (compact)
# ============================================================

@torch.no_grad()
def eval_gem(model, loader, device, use_gem=True):
    model.eval()
    P, G = [], []
    for b in loader:
        f = b["gem_features" if use_gem else "concord_features"].to(device)
        m = model(f)
        P.append(compute_cpi_mean(m, b["w_min"].to(device)).cpu())
        G.append(b["cpi_golden"])
    return compute_mape(torch.cat(P), torch.cat(G))


@torch.no_grad()
def eval_con(model, loader, device):
    model.eval()
    P, G = [], []
    for b in loader:
        p = model(b["concord_features"].to(device))
        P.append(p.cpu())
        G.append(b["cpi_golden"])
    return compute_mape(torch.cat(P), torch.cat(G))


def train_gem_q(model, tl, vl, device, use_gem=True, epochs=SWEEP_GEM_EPOCHS):
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    fk = "gem_features" if use_gem else "concord_features"
    best, state, pc = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for b in tl:
            f = b[fk].to(device)
            mt = b["mscs_target"].to(device)
            cg = b["cpi_golden"].to(device)
            wm = b["w_min"].to(device)
            mp = model(f)
            l = loss_task(mp, mt) + LAMBDA_CONST * loss_const(mp, wm, cg) + LAMBDA_MONO * loss_mono(mp)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        m = eval_gem(model, vl, device, use_gem)
        if m < best: best, state, pc = m, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= SWEEP_PATIENCE: break
    model.load_state_dict(state)
    return best


def train_con_q(model, tl, vl, device, epochs=SWEEP_CON_EPOCHS):
    opt = optim.AdamW(model.parameters(), lr=CONCORDE_LR, weight_decay=CONCORDE_WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rl = lambda p,t,e=1e-8: (torch.abs(p-t)/(torch.abs(t)+e)).mean()
    best, state, pc = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for b in tl:
            p = model(b["concord_features"].to(device))
            l = rl(p, b["cpi_golden"].to(device))
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        m = eval_con(model, vl, device)
        if m < best: best, state, pc = m, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= SWEEP_PATIENCE: break
    model.load_state_dict(state)
    return best


# ============================================================
# Run One Target × All Ratios (inclusive splits)
# ============================================================

def run_target_sweep(all_data, target_name, device):
    """
    For each ratio, the training samples of the target benchmark
    are inclusive: 5% ⊂ 10% ⊂ 15% ⊂ 20% ⊂ 30% ⊂ 40%.
    This ensures monotonic improvement.
    """
    n_total = all_data[target_name]["n_samples"]
    print(f"\n{'='*60}")
    print(f"  Sweep: {target_name} ({n_total} total samples)")
    print(f"{'='*60}")

    # Fixed permutation — same for all ratios
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_total)

    sweep_results = {"target": target_name, "ratios": {}}

    for ratio in TRAIN_RATIOS:
        n_train = max(1, int(n_total * ratio))
        tr_idx = perm[:n_train]
        te_idx = perm[n_train:]

        print(f"\n  --- ratio={ratio:.0%} (target train={n_train}, test={len(te_idx)}) ---")

        # Build data: other benchmarks (all) + target (partial)
        train_gem, train_con, train_mscs, train_cpi, train_wmin = [], [], [], [], []
        for name, data in sorted(all_data.items()):
            if name == target_name:
                train_gem.append(data["gem_features"][tr_idx])
                train_con.append(data["concord_features"][tr_idx])
                train_mscs.append(data["mscs_target"][tr_idx])
                train_cpi.append(data["cpi_golden"][tr_idx])
                train_wmin.append(data["w_min"][tr_idx])
            else:
                train_gem.append(data["gem_features"])
                train_con.append(data["concord_features"])
                train_mscs.append(data["mscs_target"])
                train_cpi.append(data["cpi_golden"])
                train_wmin.append(data["w_min"])

        train_gem  = np.concatenate(train_gem)
        train_con  = np.concatenate(train_con)
        train_mscs = np.concatenate(train_mscs)
        train_cpi  = np.concatenate(train_cpi)
        train_wmin = np.concatenate(train_wmin)

        td = all_data[target_name]
        test_gem  = td["gem_features"][te_idx]
        test_con  = td["concord_features"][te_idx]
        test_mscs = td["mscs_target"][te_idx]
        test_cpi  = td["cpi_golden"][te_idx]
        test_wmin = td["w_min"][te_idx]

        # Normalize
        gem_norm = FeatureNormalizer()
        train_gem = gem_norm.fit_transform(train_gem)
        test_gem  = gem_norm.transform(test_gem)
        con_norm = FeatureNormalizer()
        train_con = con_norm.fit_transform(train_con)
        test_con  = con_norm.transform(test_con)

        train_ds = GEMDataset(train_gem, train_con, train_mscs, train_cpi, train_wmin)
        test_ds  = GEMDataset(test_gem, test_con, test_mscs, test_cpi, test_wmin)
        tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
        vl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

        # GEM+D3
        set_seed(SEED)
        m = GEMModel(GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT).to(device)
        gf = train_gem_q(m, tl, vl, device, True)
        print(f"    GEM+D3:   {gf:.2f}%")

        # GEM-noD3
        set_seed(SEED)
        m = GEMModel(CONCORDE_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT).to(device)
        gn = train_gem_q(m, tl, vl, device, False)
        print(f"    GEM-noD3: {gn:.2f}%")

        # Concorde
        set_seed(SEED)
        m = ConcordeModel(CONCORDE_INPUT_DIM, CONCORDE_HIDDEN_DIMS, 0.0).to(device)
        cn = train_con_q(m, tl, vl, device)
        print(f"    Concorde: {cn:.2f}%")

        sweep_results["ratios"][f"{ratio:.2f}"] = {
            "gem_full": gf, "gem_nod3": gn, "concorde": cn,
            "n_target_train": n_train,
        }

    # Save
    path = os.path.join(EXP2_SWEEP_DIR, f"sweep_{target_name}.json")
    with open(path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n  Saved to {path}")
    return sweep_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Exp2 Sweep")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(EXP2_SWEEP_DIR, exist_ok=True)

    print("\nLoading benchmarks...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    bms = sorted(all_data.keys())
    targets = [args.target] if args.target else bms
    if args.target:
        assert args.target in bms

    for i, t in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}]")
        run_target_sweep(all_data, t, device)

    print("\nAll done!")


if __name__ == "__main__":
    main()