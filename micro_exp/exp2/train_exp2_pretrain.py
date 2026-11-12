"""
Self-Supervised Pre-training on 408K dataset (Server B).
Loss = L_pseudo (D3) + L_const_self
Uses 15/16 train split only - excludes 1/16 test.

Usage:
    python train_exp2_pretrain.py --gpu 0
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

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")
from config import (
    GEM_INPUT_DIM,
    GEM_HIDDEN_DIMS, DROPOUT,
    CONCORDE_BATCH_SIZE,
    LAMBDA_CONST,
    CONCORD_DATA_DIR, BENCHMARK_NAMES,
    SEED, PATIENCE,
    BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES,
    TRAIN_RATIO,
)
from model import GEMModel
from dataset import load_all_benchmarks, FeatureNormalizer

GEM_DATA_DIR = "/data/micro_traindata_packed_all/gem"

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

PRETRAIN_EPOCHS   = 1000
PRETRAIN_LR       = 1e-3
PRETRAIN_PATIENCE = 200
LAMBDA_PSEUDO     = 1.0

DEFAULT_OUTPUT = "/path/to/gem5/micro_exp/exp2/results_pretrain"


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); torch.backends.cudnn.deterministic = True


def loss_const_self(mscs_pred, w_min, eps=1e-8):
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi_per_stage = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    cpi_mean = cpi_per_stage.mean(dim=1, keepdim=True)
    return ((cpi_per_stage - cpi_mean) ** 2).mean()


def extract_d3_pseudo_mscs(gem_raw):
    N = gem_raw.shape[0]
    pseudo = np.zeros((N, 16), dtype=np.float32)
    for s_idx, stage in enumerate(STAGE_ORDER):
        fe_s, fe_e = D3_FE[stage]
        be_s, be_e = D3_BE[stage]
        fp_s = np.clip(gem_raw[:, fe_s:fe_e].mean(axis=1), 0, 1)
        bp_s = np.clip(gem_raw[:, be_s:be_e].mean(axis=1), 0, 1)
        total = fp_s + bp_s
        overflow = total > 0.95
        if overflow.any():
            scale = 0.95 / (total[overflow] + 1e-8)
            fp_s[overflow] *= scale; bp_s[overflow] *= scale
        pseudo[:, BASE_INDICES[s_idx]] = 1.0 - fp_s - bp_s
        pseudo[:, FP_INDICES[s_idx]]   = fp_s
        pseudo[:, BP_INDICES[s_idx]]   = bp_s
        pseudo[:, SP_INDICES[s_idx]]   = 0.0
    return pseudo


def create_pretrain_loader(all_data, gem_dir, batch_size=CONCORDE_BATCH_SIZE, seed=SEED):
    rng = np.random.RandomState(seed)
    all_gem, all_wmin, all_pseudo = [], [], []

    for name, data in sorted(all_data.items()):
        n = data["n_samples"]
        idx = rng.permutation(n)
        split = int(n * TRAIN_RATIO)  # 15/16
        tr_idx = idx[:split]

        all_gem.append(data["gem_features"][tr_idx])
        all_wmin.append(data["w_min"][tr_idx])

        gem_raw = np.load(os.path.join(gem_dir, f"{name}.npy")).astype(np.float32)
        if gem_raw.ndim == 1: gem_raw = gem_raw[np.newaxis, :]
        all_pseudo.append(extract_d3_pseudo_mscs(gem_raw[tr_idx]))

    all_gem    = np.concatenate(all_gem)
    all_wmin   = np.concatenate(all_wmin)
    all_pseudo = np.concatenate(all_pseudo)

    gem_norm = FeatureNormalizer()
    all_gem = gem_norm.fit_transform(all_gem)

    ds = TensorDataset(torch.from_numpy(all_gem), torch.from_numpy(all_wmin),
                       torch.from_numpy(all_pseudo))
    batch = min(batch_size, len(ds))
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=False)
    print(f"  Pretrain data: {len(ds):,} samples, batch={batch}")
    return loader, gem_norm


def train_pretrain(model, loader, device):
    optimizer = optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-5)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    best_loss, best_state, patience_ctr = float("inf"), None, 0
    step = 0; history = []

    print(f"\n{'='*60}")
    print(f"  Pre-training: L_pseudo + L_const_self")
    print(f"{'='*60}")

    t0 = time.time()
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        ep_pseudo, ep_lc, nb = 0, 0, 0
        for gem_feat, w_min, pseudo_target in loader:
            gem_feat = gem_feat.to(device)
            w_min = w_min.to(device)
            pseudo_target = pseudo_target.to(device)

            mscs_pred = model(gem_feat)
            l_pseudo = F.mse_loss(mscs_pred, pseudo_target)
            l_c = loss_const_self(mscs_pred, w_min)
            loss = LAMBDA_PSEUDO * l_pseudo + LAMBDA_CONST * l_c

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            step += 1; ep_pseudo += l_pseudo.item(); ep_lc += l_c.item(); nb += 1

        avg_loss = (LAMBDA_PSEUDO * ep_pseudo + LAMBDA_CONST * ep_lc) / nb
        history.append({"epoch": epoch, "step": step, "loss": avg_loss,
                        "L_pseudo": ep_pseudo/nb, "L_const_self": ep_lc/nb})

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PRETRAIN_PATIENCE:
                print(f"  [Pretrain] Early stop ep {epoch} (best={best_loss:.6f})"); break

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Pretrain] epoch {epoch:3d} (step {step}) | "
                  f"L_pseudo={ep_pseudo/nb:.6f} L_cself={ep_lc/nb:.6f} | "
                  f"total={avg_loss:.6f} | {time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    print(f"  [Pretrain] Done. Best loss={best_loss:.6f} ({time.time()-t0:.0f}s)")
    return model, history


def main():
    parser = argparse.ArgumentParser()
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
    loader, gem_norm = create_pretrain_loader(all_data, GEM_DATA_DIR)

    model = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    model, history = train_pretrain(model, loader, device)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "pretrained.pt"))
    np.savez(os.path.join(args.output_dir, "gem_normalizer.npz"),
             mean=gem_norm.mean, std=gem_norm.std)
    with open(os.path.join(args.output_dir, "pretrain_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Saved to {args.output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()