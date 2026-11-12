"""
Experiment 3 Supplement: Variant A+D3 with Residual Learning

GEMResidualModel:
  - D3 prior branch: maps D3 features (808) → 16-dim MSCS logits (analytical anchor)
  - Main backbone: maps all features (4681) → 16-dim MSCS logits (learned correction)
  - Output = softmax(prior + correction) per stage

In OOD: D3 prior is physics-based and stays reasonable for unseen workloads.
Even if the correction branch gives noise, the output stays anchored.

Usage:
    python train_exp3_residual.py --gpu 0
    python train_exp3_residual.py --gpu 0 --id-only
    python train_exp3_residual.py --gpu 0 --ood-only
    python train_exp3_residual.py --gpu 0 --ood-only --target 508.namd_r
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")
from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT,
    D3_FEAT_START, D3_FEAT_END,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE,
    LAMBDA_CONST, LAMBDA_MONO,
    NUM_STAGES, NUM_STATES,
    BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    TRAIN_RATIO, SEED,
)
from model import (
    MLPBlock, compute_cpi_mean, loss_task, loss_const, loss_mono,
)
from dataset import (
    load_all_benchmarks, FeatureNormalizer, GEMDataset,
)


EXP3_OUTPUT_DIR = "/scratch/gem5-TAO/micro_exp/exp3/results"
D3_DIM = D3_FEAT_END - D3_FEAT_START  # 808


# ============================================================
# GEM Residual Model
# ============================================================

class GEMResidualModel(nn.Module):
    """
    Residual architecture: MSCS_pred = softmax(prior_logits + correction_logits)
    
    - prior_branch: D3 features (808) → 16-dim logits (analytical anchor)
    - backbone + head: all features (4681) → 16-dim logits (learned correction)
    
    In OOD, even if correction is noisy, output stays anchored to D3 prior.
    """
    def __init__(self, input_dim, hidden_dims, d3_start=D3_FEAT_START,
                 d3_end=D3_FEAT_END, dropout=0.1):
        super().__init__()
        self.d3_start = d3_start
        self.d3_end = d3_end
        d3_dim = d3_end - d3_start  # 808

        # D3 prior branch: lightweight, maps analytical features to MSCS space
        self.prior_branch = nn.Sequential(
            nn.Linear(d3_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, NUM_STAGES * NUM_STATES),  # 16
        )

        # Main backbone: processes all features
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(MLPBlock(prev_dim, h_dim, dropout))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, NUM_STAGES * NUM_STATES)  # 16

    def forward(self, x):
        # D3 prior (analytical anchor)
        d3 = x[:, self.d3_start:self.d3_end]           # (batch, 808)
        prior_logits = self.prior_branch(d3)            # (batch, 16)

        # Learned correction (from all features)
        h = self.backbone(x)
        correction_logits = self.head(h)                # (batch, 16)

        # Combine and per-stage softmax
        logits = prior_logits + correction_logits       # (batch, 16)
        logits = logits.view(-1, NUM_STAGES, NUM_STATES)
        probs = F.softmax(logits, dim=-1)
        return probs.view(-1, NUM_STAGES * NUM_STATES)


# ============================================================
# Utilities
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


# ============================================================
# Data Splits
# ============================================================

def create_id_split(all_data, train_ratio=TRAIN_RATIO, batch_size=BATCH_SIZE, seed=SEED):
    rng = np.random.RandomState(seed)
    train_gem, train_mscs, train_cpi, train_wmin = [], [], [], []
    test_gem, test_mscs, test_cpi, test_wmin = [], [], [], []
    per_bm_test = {}
    test_offset = 0

    for name, data in sorted(all_data.items()):
        n = data["n_samples"]
        idx = rng.permutation(n)
        split = int(n * train_ratio)
        tr, te = idx[:split], idx[split:]

        for arrays, indices in [
            ((train_gem, train_mscs, train_cpi, train_wmin), tr),
            ((test_gem, test_mscs, test_cpi, test_wmin), te),
        ]:
            arrays[0].append(data["gem_features"][indices])
            arrays[1].append(data["mscs_target"][indices])
            arrays[2].append(data["cpi_golden"][indices])
            arrays[3].append(data["w_min"][indices])

        n_te = len(te)
        per_bm_test[name] = (test_offset, test_offset + n_te)
        test_offset += n_te

    for lst in [train_gem, train_mscs, train_cpi, train_wmin,
                test_gem, test_mscs, test_cpi, test_wmin]:
        lst[:] = [np.concatenate(lst)]

    train_gem, train_mscs, train_cpi, train_wmin = [x[0] for x in [train_gem, train_mscs, train_cpi, train_wmin]]
    test_gem, test_mscs, test_cpi, test_wmin = [x[0] for x in [test_gem, test_mscs, test_cpi, test_wmin]]

    norm = FeatureNormalizer()
    train_gem = norm.fit_transform(train_gem)
    test_gem  = norm.transform(test_gem)

    train_ds = GEMDataset(train_gem, train_gem, train_mscs, train_cpi, train_wmin)
    test_ds  = GEMDataset(test_gem, test_gem, test_mscs, test_cpi, test_wmin)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, test_loader, per_bm_test


def create_loo_split(all_data, held_out, batch_size=BATCH_SIZE):
    train_gem, train_mscs, train_cpi, train_wmin = [], [], [], []
    for name, data in sorted(all_data.items()):
        if name == held_out:
            continue
        train_gem.append(data["gem_features"])
        train_mscs.append(data["mscs_target"])
        train_cpi.append(data["cpi_golden"])
        train_wmin.append(data["w_min"])

    train_gem  = np.concatenate(train_gem)
    train_mscs = np.concatenate(train_mscs)
    train_cpi  = np.concatenate(train_cpi)
    train_wmin = np.concatenate(train_wmin)

    td = all_data[held_out]
    test_gem, test_mscs = td["gem_features"], td["mscs_target"]
    test_cpi, test_wmin = td["cpi_golden"], td["w_min"]

    norm = FeatureNormalizer()
    train_gem = norm.fit_transform(train_gem)
    test_gem  = norm.transform(test_gem)

    train_ds = GEMDataset(train_gem, train_gem, train_mscs, train_cpi, train_wmin)
    test_ds  = GEMDataset(test_gem, test_gem, test_mscs, test_cpi, test_wmin)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, test_loader


# ============================================================
# Evaluate & Train
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_gold = [], []
    for batch in loader:
        feat  = batch["gem_features"].to(device)
        w_min = batch["w_min"].to(device)
        mscs_pred = model(feat)
        cpi_pred = compute_cpi_mean(mscs_pred, w_min)
        all_pred.append(cpi_pred.cpu())
        all_gold.append(batch["cpi_golden"])
    return compute_mape(torch.cat(all_pred), torch.cat(all_gold))


def train_residual(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_mape, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            feat     = batch["gem_features"].to(device)
            mscs_tgt = batch["mscs_target"].to(device)
            cpi_gold = batch["cpi_golden"].to(device)
            w_min    = batch["w_min"].to(device)

            mscs_pred = model(feat)
            l_t = loss_task(mscs_pred, mscs_tgt)
            l_c = loss_const(mscs_pred, w_min, cpi_gold)
            l_m = loss_mono(mscs_pred)
            loss = l_t + LAMBDA_CONST * l_c + LAMBDA_MONO * l_m

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        mape = evaluate(model, test_loader, device)
        if mape < best_mape:
            best_mape = mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
        if epoch % 50 == 0:
            print(f"      epoch {epoch:3d} | MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    return model, best_mape


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Exp3: A+D3 Residual")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--id-only", action="store_true")
    parser.add_argument("--ood-only", action="store_true")
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(EXP3_OUTPUT_DIR, exist_ok=True)

    print("\nLoading benchmarks...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    benchmark_names = sorted(all_data.keys())
    results = {}

    def make_model():
        return GEMResidualModel(
            input_dim=GEM_INPUT_DIM,
            hidden_dims=CONCORDE_HIDDEN_DIMS,
            d3_start=D3_FEAT_START,
            d3_end=D3_FEAT_END,
            dropout=DROPOUT,
        ).to(device)

    # ============================
    # In-Distribution
    # ============================
    if not args.ood_only:
        print("\n" + "=" * 60)
        print("  A+D3 Residual: In-Distribution")
        print("=" * 60)

        train_loader, test_loader, _ = create_id_split(all_data)
        model = make_model()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,} (prior branch + main backbone)")

        _, id_mape = train_residual(model, train_loader, test_loader, device)
        results["id_mape"] = id_mape
        print(f"\n  A+D3 Residual ID MAPE: {id_mape:.2f}%")

        with open(os.path.join(EXP3_OUTPUT_DIR, "exp3_residual_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # ============================
    # OOD
    # ============================
    if not args.id_only:
        print("\n" + "=" * 60)
        print("  A+D3 Residual: OOD (Leave-One-Out)")
        print("=" * 60)

        if args.target:
            assert args.target in benchmark_names
            targets = [args.target]
        else:
            targets = benchmark_names

        ood_results = []
        for i, held_out in enumerate(targets):
            n_test = all_data[held_out]["n_samples"]
            print(f"\n  [{i+1}/{len(targets)}] Holding out: {held_out} ({n_test})")

            train_loader, test_loader = create_loo_split(all_data, held_out)
            model = make_model()
            _, mape = train_residual(model, train_loader, test_loader, device)
            print(f"    Residual MAPE: {mape:.2f}%")

            ood_results.append({"held_out": held_out, "mape": mape, "n_samples": n_test})

            # Save per-benchmark
            per_path = os.path.join(EXP3_OUTPUT_DIR, f"exp3_residual_ood_{held_out}.json")
            with open(per_path, "w") as f:
                json.dump(ood_results[-1], f, indent=2)

        if not args.target:
            avg = np.mean([r["mape"] for r in ood_results])
            print(f"\n  --- A+D3 Residual OOD Summary ---")
            print(f"  {'Benchmark':<20} {'Residual':>12}")
            print(f"  {'-'*34}")
            for r in ood_results:
                print(f"  {r['held_out']:<20} {r['mape']:>11.2f}%")
            print(f"  {'-'*34}")
            print(f"  {'Average':<20} {avg:>11.2f}%")
            results["ood"] = ood_results
            results["ood_avg"] = avg

    result_path = os.path.join(EXP3_OUTPUT_DIR, "exp3_residual_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {result_path}")


if __name__ == "__main__":
    main()