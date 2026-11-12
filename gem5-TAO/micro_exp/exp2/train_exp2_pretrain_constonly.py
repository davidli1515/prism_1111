"""
Self-Supervised Pre-training with D3 Pseudo-Targets.

D3 analytical model provides per-stage FE/BE pressure distributions (808 dims).
We convert them to pseudo MSCS targets for weak supervision:
  L_pretrain = L_pseudo + λ₁·L_const_self

Usage:
    python train_exp2_pretrain.py --gpu 0
    python train_exp2_pretrain.py --gpu 0 --output_dir ./results/pretrain_v2
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")
from config import (
    GEM_INPUT_DIM,
    GEM_HIDDEN_DIMS, DROPOUT,
    CONCORDE_BATCH_SIZE,
    LAMBDA_CONST,
    CONCORD_DATA_DIR, BENCHMARK_NAMES,
    SEED, PATIENCE,
    BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES,
)
from model import GEMModel
from dataset import load_all_benchmarks, FeatureNormalizer

# Data path
GEM_DATA_DIR = "/scratch2/wl730/micro_traindata_packed_all/gem"

# D3 pressure distribution layout within gem_data
# 4 stages × 2 streams (FE, BE) × 101 percentile values = 808 dims
D3_FE = {
    "decode":   (3873, 3974),
    "dispatch": (3974, 4075),
    "issue":    (4075, 4176),
    "commit":   (4176, 4277),
}
D3_BE = {
    "decode":   (4277, 4378),
    "dispatch": (4378, 4479),
    "issue":    (4479, 4580),
    "commit":   (4580, 4681),
}
STAGE_ORDER = ["decode", "dispatch", "issue", "commit"]

# Hyperparameters
PRETRAIN_EPOCHS   = 1000
PRETRAIN_LR       = 1e-3
PRETRAIN_PATIENCE = 200
LAMBDA_PSEUDO     = 1.0

DEFAULT_OUTPUT = "/scratch/gem5-TAO/micro_exp/exp2/results_pretrain"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def loss_const_self(mscs_pred, w_min, eps=1e-8):
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi_per_stage = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    cpi_mean = cpi_per_stage.mean(dim=1, keepdim=True)
    diff = cpi_per_stage - cpi_mean
    return (diff ** 2).mean()


# ============================================================
# D3 → Pseudo MSCS
# ============================================================

def extract_d3_pseudo_mscs(gem_raw):
    """
    Extract per-stage FP/BP from D3 pressure distributions.
    Take mean of 101 percentile values as scalar pressure per stage.
    Construct 16-dim pseudo MSCS: [Base, FP, BP, SP] × 4 stages.

    D3 doesn't provide SP separately, so SP_pseudo = 0,
    Base_pseudo = 1 - FP - BP.

    Returns: (N, 16) pseudo MSCS targets
    """
    N = gem_raw.shape[0]
    pseudo = np.zeros((N, 16), dtype=np.float32)

    for s_idx, stage in enumerate(STAGE_ORDER):
        fe_start, fe_end = D3_FE[stage]
        be_start, be_end = D3_BE[stage]

        # Mean of 101 percentile values → scalar pressure
        fp_s = gem_raw[:, fe_start:fe_end].mean(axis=1)
        bp_s = gem_raw[:, be_start:be_end].mean(axis=1)

        # Clamp and prevent overflow
        fp_s = np.clip(fp_s, 0, 1)
        bp_s = np.clip(bp_s, 0, 1)
        total = fp_s + bp_s
        overflow = total > 0.95
        if overflow.any():
            scale = 0.95 / (total[overflow] + 1e-8)
            fp_s[overflow] *= scale
            bp_s[overflow] *= scale

        base_s = 1.0 - fp_s - bp_s

        pseudo[:, BASE_INDICES[s_idx]] = base_s
        pseudo[:, FP_INDICES[s_idx]]   = fp_s
        pseudo[:, BP_INDICES[s_idx]]   = bp_s
        pseudo[:, SP_INDICES[s_idx]]   = 0.0

    return pseudo


# ============================================================
# Data Loading
# ============================================================

def create_pretrain_loader(all_data, gem_dir, batch_size=CONCORDE_BATCH_SIZE):
    all_gem, all_wmin, all_pseudo = [], [], []

    for name, data in sorted(all_data.items()):
        all_gem.append(data["gem_features"])
        all_wmin.append(data["w_min"])

        # Load raw data to extract D3 pressures (before normalization)
        gem_path = os.path.join(gem_dir, f"{name}.npy")
        gem_raw = np.load(gem_path).astype(np.float32)
        if gem_raw.ndim == 1:
            gem_raw = gem_raw[np.newaxis, :]
        all_pseudo.append(extract_d3_pseudo_mscs(gem_raw))

    all_gem    = np.concatenate(all_gem)
    all_wmin   = np.concatenate(all_wmin)
    all_pseudo = np.concatenate(all_pseudo)

    gem_norm = FeatureNormalizer()
    all_gem = gem_norm.fit_transform(all_gem)

    ds = TensorDataset(
        torch.from_numpy(all_gem),
        torch.from_numpy(all_wmin),
        torch.from_numpy(all_pseudo),
    )
    actual_batch = min(batch_size, len(ds))
    loader = DataLoader(ds, batch_size=actual_batch, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=False)

    print(f"  Pretrain data: {len(ds):,} samples, batch={actual_batch}")
    print(f"  Pseudo MSCS stats:")
    for label, indices in [("Base", BASE_INDICES), ("FP", FP_INDICES),
                           ("BP", BP_INDICES), ("SP", SP_INDICES)]:
        vals = all_pseudo[:, indices]
        print(f"    {label}: mean={vals.mean():.4f} std={vals.std():.4f} "
              f"min={vals.min():.4f} max={vals.max():.4f}")

    return loader, gem_norm


# ============================================================
# Training
# ============================================================

def train_pretrain(model, pretrain_loader, device,
                   num_epochs=PRETRAIN_EPOCHS, lr=PRETRAIN_LR,
                   patience=PRETRAIN_PATIENCE):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    best_loss, best_state, patience_ctr = float("inf"), None, 0
    step = 0
    history = []

    print(f"\n{'='*60}")
    print(f"  Pre-training: L_pseudo + L_const_self")
    print(f"  λ_pseudo={LAMBDA_PSEUDO} λ_const={LAMBDA_CONST}")
    print(f"  Epochs: {num_epochs} | LR: {lr} | Patience: {patience}")
    print(f"{'='*60}")

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        ep_pseudo, ep_lc, nb = 0, 0, 0
        for gem_feat, w_min, pseudo_target in pretrain_loader:
            gem_feat      = gem_feat.to(device)
            w_min         = w_min.to(device)
            pseudo_target = pseudo_target.to(device)

            mscs_pred = model(gem_feat)

            l_pseudo = F.mse_loss(mscs_pred, pseudo_target)
            l_c = loss_const_self(mscs_pred, w_min)
            loss = LAMBDA_PSEUDO * l_pseudo + LAMBDA_CONST * l_c

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            ep_pseudo += l_pseudo.item()
            ep_lc += l_c.item()
            nb += 1

        avg_loss = (LAMBDA_PSEUDO * ep_pseudo + LAMBDA_CONST * ep_lc) / nb
        history.append({
            "epoch": epoch, "step": step, "loss": avg_loss,
            "L_pseudo": ep_pseudo/nb, "L_const_self": ep_lc/nb,
        })

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  [Pretrain] Early stop ep {epoch} (best_loss={best_loss:.6f})")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Pretrain] epoch {epoch:3d} (step {step}) | "
                  f"L_pseudo={ep_pseudo/nb:.6f} L_cself={ep_lc/nb:.6f} | "
                  f"total={avg_loss:.6f} | {time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    print(f"  [Pretrain] Done. Best loss={best_loss:.6f} ({time.time()-t0:.0f}s)")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="D3 Pseudo-Target Pre-training")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data: {GEM_DATA_DIR}")
    print(f"Output: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading benchmarks...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)

    pretrain_loader, gem_norm = create_pretrain_loader(all_data, GEM_DATA_DIR)

    model = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")

    model, history = train_pretrain(model, pretrain_loader, device)

    ckpt_path = os.path.join(args.output_dir, "pretrained.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    norm_path = os.path.join(args.output_dir, "gem_normalizer.npz")
    np.savez(norm_path, mean=gem_norm.mean, std=gem_norm.std)
    print(f"  Normalizer: {norm_path}")

    hist_path = os.path.join(args.output_dir, "pretrain_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History: {hist_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
