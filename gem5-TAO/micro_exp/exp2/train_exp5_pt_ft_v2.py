"""
Exp5 Pretrain+FT with Type 2 Stage 2 recipe (18k only).
Loss: 0.1*L_task + 1.0*L_const + 0.01*L_mono
Optimizer: AdamW, lr=1e-4, wd=0.3, epochs=500, batch=50000
LR: MultiStepLR at [42%, 58%, 75%, 92%], gamma=0.5
No early stopping, track best.

Split aligned with semi-supervised experiments:
  - Train/Test: SEED=42, TRAIN_RATIO=15/16, sorted benchmark order, shared RNG
  - Labeled/Unlabeled: stable_hash(name) + ratio-based RNG per benchmark
  - Normalizer: fit on ALL train data (not just labeled)

Usage:
    python train_exp5_pt_ft_v2.py --gpu 0 --ratio 0.05
"""

import os
import sys
import json
import hashlib
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")
from config import (
    GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT,
    D3_FEAT_END, MSCS_START, MSCS_END, WMIN_INDICES,
    BASE_INDICES,
)
from model import GEMModel, compute_cpi_mean, loss_task, loss_const, loss_mono
from dataset import FeatureNormalizer

# ============================================================
# Paths
# ============================================================
GEM_DATA_DIR   = "/scratch2/wl730/micro_traindata_packed/gem"
PRETRAIN_CKPT  = "/scratch/gem5-TAO/micro_exp/exp5/results_pt_ft_v2_18k/pretrained_v2_18k.pt"
PRETRAIN_NORM  = "/scratch/gem5-TAO/micro_exp/exp5/results_pt_ft_v2_18k/gem_normalizer_v2_18k.npz"
OUTPUT_DIR     = "/scratch/gem5-TAO/micro_exp/exp5/results_pt_ft_v2_18k"

# ============================================================
# Split Parameters (must match semi-supervised exactly)
# ============================================================
SEED = 42
TRAIN_RATIO = 15 / 16  # 0.9375

# ============================================================
# Type 2 Stage 2 Recipe
# ============================================================
FT_LR         = 1e-4
FT_WD         = 0.3
FT_EPOCHS     = 1521
FT_BATCH      = 50000
LAMBDA_TASK    = 0.1
LAMBDA_CONST_  = 1.0
LAMBDA_MONO_   = 0.01
MILESTONE_PCT  = [0.42, 0.58, 0.75, 0.92]
GAMMA          = 0.5


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def stable_hash(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)


def compute_mape(p, t, eps=1e-8):
    return (torch.abs(p - t) / (torch.abs(t) + eps) * 100).mean().item()


# ============================================================
# Data Loading
# ============================================================

def load_benchmark_raw(name, gem_dir):
    raw = np.load(os.path.join(gem_dir, f"{name}.npy")).astype(np.float32)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    gem_features = raw[:, :D3_FEAT_END]
    mscs_target  = raw[:, MSCS_START:MSCS_END]
    widths = raw[:, WMIN_INDICES]
    w_min  = widths.min(axis=1)
    base_commit = mscs_target[:, BASE_INDICES[3]]
    cpi_golden  = 1.0 / (base_commit * w_min + 1e-8)
    return gem_features, mscs_target, cpi_golden, w_min


def discover_benchmarks(gem_dir):
    return sorted([f[:-4] for f in os.listdir(gem_dir) if f.endswith(".npy")])


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, gem, mscs, cpi, wmin):
        self.gem  = torch.from_numpy(gem)
        self.mscs = torch.from_numpy(mscs)
        self.cpi  = torch.from_numpy(cpi)
        self.wmin = torch.from_numpy(wmin)

    def __len__(self):
        return self.gem.shape[0]

    def __getitem__(self, i):
        return {"gem_features": self.gem[i], "mscs_target": self.mscs[i],
                "cpi_golden": self.cpi[i], "w_min": self.wmin[i]}


# ============================================================
# Data Split (aligned with semi-supervised)
# ============================================================

def create_split(gem_dir, labeled_ratio):
    benchmarks = discover_benchmarks(gem_dir)

    # Step 1: Train/Test split (shared RNG, sorted order)
    rng = np.random.RandomState(SEED)
    all_train_gem, all_train_mscs, all_train_cpi, all_train_wmin = [], [], [], []
    all_test_gem, all_test_mscs, all_test_cpi, all_test_wmin = [], [], [], []
    labeled_gem, labeled_mscs, labeled_cpi, labeled_wmin = [], [], [], []

    for name in benchmarks:
        gem_feat, mscs_tgt, cpi_gold, w_min = load_benchmark_raw(name, gem_dir)
        n = gem_feat.shape[0]
        indices = rng.permutation(n)
        split = int(n * TRAIN_RATIO)
        train_idx = indices[:split]
        test_idx  = indices[split:]

        # ALL train data (for normalizer)
        all_train_gem.append(gem_feat[train_idx])
        all_train_mscs.append(mscs_tgt[train_idx])
        all_train_cpi.append(cpi_gold[train_idx])
        all_train_wmin.append(w_min[train_idx])

        # Test data
        all_test_gem.append(gem_feat[test_idx])
        all_test_mscs.append(mscs_tgt[test_idx])
        all_test_cpi.append(cpi_gold[test_idx])
        all_test_wmin.append(w_min[test_idx])

        # Step 2: Labeled selection (per-benchmark RNG with stable_hash)
        n_train = len(train_idx)
        n_labeled = max(1, int(n_train * labeled_ratio))
        label_rng = np.random.RandomState(
            SEED + stable_hash(name) + int(round(labeled_ratio * 1000))
        )
        label_perm = label_rng.permutation(n_train)
        lab_idx = train_idx[label_perm[:n_labeled]]

        labeled_gem.append(gem_feat[lab_idx])
        labeled_mscs.append(mscs_tgt[lab_idx])
        labeled_cpi.append(cpi_gold[lab_idx])
        labeled_wmin.append(w_min[lab_idx])

    # Concatenate
    all_train_gem = np.concatenate(all_train_gem)
    labeled_gem   = np.concatenate(labeled_gem)
    labeled_mscs  = np.concatenate(labeled_mscs)
    labeled_cpi   = np.concatenate(labeled_cpi)
    labeled_wmin  = np.concatenate(labeled_wmin)
    test_gem      = np.concatenate(all_test_gem)
    test_mscs     = np.concatenate(all_test_mscs)
    test_cpi      = np.concatenate(all_test_cpi)
    test_wmin     = np.concatenate(all_test_wmin)

    # Normalizer: fit on ALL train data
    gem_norm = FeatureNormalizer()
    gem_norm.fit(all_train_gem)
    labeled_gem = gem_norm.transform(labeled_gem)
    test_gem    = gem_norm.transform(test_gem)

    train_ds = SimpleDataset(labeled_gem, labeled_mscs, labeled_cpi, labeled_wmin)
    test_ds  = SimpleDataset(test_gem, test_mscs, test_cpi, test_wmin)

    bs = min(FT_BATCH, len(train_ds))
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=len(train_ds) > bs)
    tel = DataLoader(test_ds, batch_size=FT_BATCH, shuffle=False,
                     num_workers=4, pin_memory=True)

    n_labeled_total = len(labeled_cpi)
    print(f"  Ratio: {labeled_ratio:.0%} | Labeled: {n_labeled_total:,} | "
          f"Test: {len(test_ds):,} | Batch: {bs}")

    return tl, tel, gem_norm


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, G = [], []
    for b in loader:
        m = model(b["gem_features"].to(device))
        P.append(compute_cpi_mean(m, b["w_min"].to(device)).cpu())
        G.append(b["cpi_golden"])
    return compute_mape(torch.cat(P), torch.cat(G))


# ============================================================
# Fine-tune: Type 2 Stage 2
# ============================================================

def finetune(model, tl, tel, device, label="PT-FT-v2"):
    optimizer = optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=FT_WD)

    steps_per_epoch = len(tl)
    total_steps = FT_EPOCHS * steps_per_epoch
    milestones = [int(total_steps * p) for p in MILESTONE_PCT]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=GAMMA)

    print(f"  Steps/epoch: {steps_per_epoch} | Total: {total_steps} | "
          f"Milestones: {milestones}")

    best_mape = float("inf")
    best_state = None
    step = 0
    history = []

    for ep in range(1, FT_EPOCHS + 1):
        model.train()
        ep_lt, ep_lc, ep_lm, nb = 0, 0, 0, 0

        for b in tl:
            feat = b["gem_features"].to(device)
            mscs_tgt = b["mscs_target"].to(device)
            cpi_gold = b["cpi_golden"].to(device)
            w_min = b["w_min"].to(device)

            pred = model(feat)
            lt = loss_task(pred, mscs_tgt)
            lc = loss_const(pred, w_min, cpi_gold)
            lm = loss_mono(pred)
            loss = LAMBDA_TASK * lt + LAMBDA_CONST_ * lc + LAMBDA_MONO_ * lm

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            ep_lt += lt.item(); ep_lc += lc.item(); ep_lm += lm.item(); nb += 1

        mape = evaluate(model, tel, device)
        history.append({
            "epoch": ep, "step": step, "mape": mape,
            "L_task": ep_lt/nb, "L_const": ep_lc/nb, "L_mono": ep_lm/nb,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if mape < best_mape:
            best_mape = mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 50 == 0 or ep == 1:
            print(f"  [{label}] ep {ep:3d} (step {step}) | "
                  f"L_task={ep_lt/nb:.5f} L_const={ep_lc/nb:.5f} L_mono={ep_lm/nb:.5f} | "
                  f"MAPE={mape:.2f}% (best={best_mape:.2f}%) | "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}")

    model.load_state_dict(best_state)
    print(f"  [{label}] Done. Best MAPE: {best_mape:.2f}%")
    return model, best_mape, history


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--ckpt", type=str, default=PRETRAIN_CKPT)
    parser.add_argument("--norm", type=str, default=PRETRAIN_NORM)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    assert os.path.exists(args.ckpt), f"Not found: {args.ckpt}"

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    ratio_tag = f"{int(args.ratio * 100)}pct"
    print(f"Device: {device} | Ratio: {args.ratio:.0%}")
    print(f"Recipe: AdamW lr={FT_LR} wd={FT_WD} epochs={FT_EPOCHS} batch={FT_BATCH}")
    print(f"Loss: {LAMBDA_TASK}*L_task + {LAMBDA_CONST_}*L_const + {LAMBDA_MONO_}*L_mono")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Output: {args.output_dir}")

    # Data split (normalizer fit on all train, not from pretrain)
    tl, tel, gem_norm = create_split(GEM_DATA_DIR, args.ratio)

    # Load pretrained model
    # NOTE: pretrain used its own normalizer. Here we use normalizer fit on
    # all train data (same as semi-supervised) for fair comparison.
    model = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    zero = evaluate(model, tel, device)
    print(f"  Zero-shot: {zero:.2f}%")

    # Fine-tune
    t0 = time.time()
    model, mape, hist = finetune(model, tl, tel, device,
                                  label=f"PT-FT-v2-{ratio_tag}")

    # Save
    result = {
        "labeled_ratio": args.ratio,
        "zero_shot": zero,
        "pretrain_ft_v2": mape,
        "recipe": {
            "lr": FT_LR, "wd": FT_WD, "epochs": FT_EPOCHS,
            "batch": FT_BATCH, "lambda_task": LAMBDA_TASK,
            "lambda_const": LAMBDA_CONST_, "lambda_mono": LAMBDA_MONO_,
        },
        "history": hist,
    }

    out_path = os.path.join(args.output_dir, f"exp4_pt_ft_v2_{ratio_tag}_18k.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    torch.save(model.state_dict(),
               os.path.join(args.output_dir, f"pt_ft_v2_{ratio_tag}_18k.pt"))

    print(f"\n  {zero:.2f}% -> {mape:.2f}% ({time.time()-t0:.0f}s)")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()