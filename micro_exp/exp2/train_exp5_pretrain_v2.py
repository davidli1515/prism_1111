"""
Type 2 Stage 1: Self-supervised pretrain (18k, Server B).
Loss: 1.0*L_const_self + 0.01*L_mono (no D3, no L_task)
Optimizer: AdamW, lr=1e-3, wd=0.3, epochs=1521, batch=50000
LR: MultiStepLR at [42%, 58%, 75%, 92%], gamma=0.5
No early stopping, track best.

Usage:
    python train_exp5_pretrain_v2.py --gpu 0
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")
from config import (
    GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT,
    D3_FEAT_END, MSCS_START, MSCS_END, WMIN_INDICES,
    BASE_INDICES,
)
from model import GEMModel, loss_mono
from dataset import FeatureNormalizer

# ============================================================
# Paths
# ============================================================
GEM_DATA_DIR = "/data/micro_traindata_packed/gem"
OUTPUT_DIR   = "/path/to/gem5/micro_exp/exp5/results_pt_ft_v2_18k"

# ============================================================
# Split (must match FT and semi exactly)
# ============================================================
SEED = 42
TRAIN_RATIO = 15 / 16

# ============================================================
# Recipe
# ============================================================
PT_LR          = 1e-3
PT_WD          = 0.3
PT_EPOCHS      = 1521
PT_BATCH       = 50000
LAMBDA_CONST_  = 1.0
LAMBDA_MONO_   = 0.01
MILESTONE_PCT  = [0.42, 0.58, 0.75, 0.92]
GAMMA          = 0.5


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def loss_const_self(mscs_pred, w_min, eps=1e-8):
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    return ((cpi - cpi.mean(dim=1, keepdim=True)) ** 2).mean()


def discover_benchmarks(gem_dir):
    return sorted([f[:-4] for f in os.listdir(gem_dir) if f.endswith(".npy")])


def create_pretrain_loader(gem_dir):
    """Load 15/16 train split from all benchmarks. Same RNG as FT/semi."""
    rng = np.random.RandomState(SEED)
    benchmarks = discover_benchmarks(gem_dir)
    all_gem, all_wmin = [], []

    for name in benchmarks:
        raw = np.load(os.path.join(gem_dir, f"{name}.npy")).astype(np.float32)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        n = raw.shape[0]
        indices = rng.permutation(n)
        split = int(n * TRAIN_RATIO)
        train_idx = indices[:split]

        all_gem.append(raw[train_idx][:, :D3_FEAT_END])
        all_wmin.append(raw[train_idx][:, WMIN_INDICES].min(axis=1))

    all_gem  = np.concatenate(all_gem)
    all_wmin = np.concatenate(all_wmin)

    # Normalizer fit on all train data
    gem_norm = FeatureNormalizer()
    all_gem = gem_norm.fit_transform(all_gem)

    ds = TensorDataset(torch.from_numpy(all_gem), torch.from_numpy(all_wmin))
    bs = min(PT_BATCH, len(ds))
    loader = DataLoader(ds, batch_size=bs, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=False)

    print(f"  Pretrain: {len(ds):,} samples (15/16 split), batch={bs}")
    return loader, gem_norm


def train_pretrain(model, loader, device):
    optimizer = optim.AdamW(model.parameters(), lr=PT_LR, weight_decay=PT_WD)

    steps_per_epoch = len(loader)
    total_steps = PT_EPOCHS * steps_per_epoch
    milestones = [int(total_steps * p) for p in MILESTONE_PCT]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=GAMMA)

    print(f"  Steps/epoch: {steps_per_epoch} | Total: {total_steps} | "
          f"Milestones: {milestones}")

    best_loss = float("inf")
    best_state = None
    step = 0
    history = []
    t0 = time.time()

    for ep in range(1, PT_EPOCHS + 1):
        model.train()
        ep_lc, ep_lm, nb = 0, 0, 0

        for gem_feat, w_min in loader:
            gem_feat = gem_feat.to(device)
            w_min = w_min.to(device)

            pred = model(gem_feat)
            lc = loss_const_self(pred, w_min)
            lm = loss_mono(pred)
            loss = LAMBDA_CONST_ * lc + LAMBDA_MONO_ * lm

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            ep_lc += lc.item(); ep_lm += lm.item(); nb += 1

        avg_loss = (LAMBDA_CONST_ * ep_lc + LAMBDA_MONO_ * ep_lm) / nb
        history.append({
            "epoch": ep, "step": step, "loss": avg_loss,
            "L_const_self": ep_lc/nb, "L_mono": ep_lm/nb,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 50 == 0 or ep == 1:
            print(f"  [Pretrain] ep {ep:4d} (step {step}) | "
                  f"L_cself={ep_lc/nb:.6f} L_mono={ep_lm/nb:.6f} | "
                  f"total={avg_loss:.6f} (best={best_loss:.6f}) | "
                  f"lr={optimizer.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    print(f"  [Pretrain] Done. Best loss: {best_loss:.6f} ({time.time()-t0:.0f}s)")
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Data: {GEM_DATA_DIR}")
    print(f"Recipe: AdamW lr={PT_LR} wd={PT_WD} epochs={PT_EPOCHS} batch={PT_BATCH}")
    print(f"Loss: {LAMBDA_CONST_}*L_const_self + {LAMBDA_MONO_}*L_mono (no D3)")
    print(f"Output: {args.output_dir}")

    loader, gem_norm = create_pretrain_loader(GEM_DATA_DIR)

    model = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    model, history = train_pretrain(model, loader, device)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "pretrained_v2_18k.pt"))
    np.savez(os.path.join(args.output_dir, "gem_normalizer_v2_18k.npz"),
             mean=gem_norm.mean, std=gem_norm.std)
    with open(os.path.join(args.output_dir, "pretrain_v2_history_18k.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Saved to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()