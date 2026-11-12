"""
Pretrain + Semi-Supervised FT on 183K dataset (Server B).
Loads pretrained_18k.pt, fine-tunes with labeled + unlabeled in same batch.
  - Labeled:   L_task + L_const + L_mono
  - Unlabeled: L_const_self + L_mono

Usage:
    python train_exp2_ft_18k_semi.py --gpu 0 --ratio 0.05
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
    GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT,
    BATCH_SIZE, LAMBDA_CONST, LAMBDA_MONO,
    SEED, TRAIN_RATIO, BASE_INDICES,
)
from model import GEMModel, compute_cpi_mean, loss_task, loss_const, loss_mono
from dataset import FeatureNormalizer

GEM_DATA_DIR   = "/data/micro_traindata_packed/gem"
PRETRAIN_CKPT  = "/path/to/gem5/micro_exp/exp5/results_pretrain_18k/pretrained_18k.pt"
PRETRAIN_NORM  = "/path/to/gem5/micro_exp/exp5/results_pretrain_18k/gem_normalizer_18k.npz"
OUTPUT_DIR     = "/path/to/gem5/micro_exp/exp5/results_pretrain_18k_semi"

FT_LR = 5e-4
FT_EPOCHS = 1500
FT_PATIENCE = 200


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def compute_mape(p, t, eps=1e-8):
    return (torch.abs(p - t) / (torch.abs(t) + eps) * 100).mean().item()


def loss_const_self(mscs_pred, w_min, eps=1e-8):
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    return ((cpi - cpi.mean(dim=1, keepdim=True)) ** 2).mean()


# ============================================================
# Dataset with has_label flag
# ============================================================

class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, gem_features, mscs_target, cpi_golden, w_min, has_label):
        self.gem_features = torch.from_numpy(gem_features)
        self.mscs_target  = torch.from_numpy(mscs_target)
        self.cpi_golden   = torch.from_numpy(cpi_golden)
        self.w_min        = torch.from_numpy(w_min)
        self.has_label    = torch.from_numpy(has_label.astype(np.float32))

    def __len__(self):
        return self.gem_features.shape[0]

    def __getitem__(self, idx):
        return {
            "gem_features": self.gem_features[idx],
            "mscs_target":  self.mscs_target[idx],
            "cpi_golden":   self.cpi_golden[idx],
            "w_min":        self.w_min[idx],
            "has_label":    self.has_label[idx],
        }


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, gem_features, mscs_target, cpi_golden, w_min):
        self.gem_features = torch.from_numpy(gem_features)
        self.mscs_target  = torch.from_numpy(mscs_target)
        self.cpi_golden   = torch.from_numpy(cpi_golden)
        self.w_min        = torch.from_numpy(w_min)

    def __len__(self):
        return self.gem_features.shape[0]

    def __getitem__(self, idx):
        return {
            "gem_features": self.gem_features[idx],
            "mscs_target":  self.mscs_target[idx],
            "cpi_golden":   self.cpi_golden[idx],
            "w_min":        self.w_min[idx],
        }


# ============================================================
# Data Loading
# ============================================================

def load_benchmark(name, gem_dir):
    from config import D3_FEAT_END, CONCORDE_FEAT_END, MSCS_START, MSCS_END, WMIN_INDICES
    path = os.path.join(gem_dir, f"{name}.npy")
    raw = np.load(path).astype(np.float32)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    gem_features = raw[:, :D3_FEAT_END]
    mscs_target  = raw[:, MSCS_START:MSCS_END]
    widths = raw[:, WMIN_INDICES]
    w_min  = widths.min(axis=1)
    base_commit = mscs_target[:, BASE_INDICES[3]]
    cpi_golden  = 1.0 / (base_commit * w_min + 1e-8)
    return {
        "gem_features": gem_features,
        "mscs_target":  mscs_target,
        "cpi_golden":   cpi_golden,
        "w_min":        w_min,
        "n_samples":    raw.shape[0],
    }


def discover_benchmarks(gem_dir):
    return sorted([f[:-4] for f in os.listdir(gem_dir) if f.endswith(".npy")])


# ============================================================
# Data Split: 15/16 train (labeled + unlabeled) + 1/16 test
# ============================================================

def create_semi_split(gem_dir, ratio, gem_norm, seed=SEED):
    rng = np.random.RandomState(seed)
    benchmarks = discover_benchmarks(gem_dir)

    train_gem, train_mscs, train_cpi, train_wmin, train_label = [], [], [], [], []
    test_gem, test_mscs, test_cpi, test_wmin = [], [], [], []

    for name in benchmarks:
        data = load_benchmark(name, gem_dir)
        n = data["n_samples"]
        idx = rng.permutation(n)
        pool_split = int(n * TRAIN_RATIO)
        pool_idx, te_idx = idx[:pool_split], idx[pool_split:]

        # Within train pool: first n_labeled are labeled, rest unlabeled
        n_labeled = max(1, int(len(pool_idx) * ratio))
        has_label = np.zeros(len(pool_idx), dtype=np.float32)
        has_label[:n_labeled] = 1.0

        train_gem.append(data["gem_features"][pool_idx])
        train_mscs.append(data["mscs_target"][pool_idx])
        train_cpi.append(data["cpi_golden"][pool_idx])
        train_wmin.append(data["w_min"][pool_idx])
        train_label.append(has_label)

        test_gem.append(data["gem_features"][te_idx])
        test_mscs.append(data["mscs_target"][te_idx])
        test_cpi.append(data["cpi_golden"][te_idx])
        test_wmin.append(data["w_min"][te_idx])

    # Concatenate
    train_gem   = gem_norm.transform(np.concatenate(train_gem))
    test_gem    = gem_norm.transform(np.concatenate(test_gem))
    train_mscs  = np.concatenate(train_mscs)
    test_mscs   = np.concatenate(test_mscs)
    train_cpi   = np.concatenate(train_cpi)
    test_cpi    = np.concatenate(test_cpi)
    train_wmin  = np.concatenate(train_wmin)
    test_wmin   = np.concatenate(test_wmin)
    train_label = np.concatenate(train_label)

    n_labeled_total = int(train_label.sum())
    n_unlabeled_total = len(train_label) - n_labeled_total

    train_ds = SemiSupervisedDataset(train_gem, train_mscs, train_cpi, train_wmin, train_label)
    test_ds  = TestDataset(test_gem, test_mscs, test_cpi, test_wmin)

    bs = min(BATCH_SIZE, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True,
                              drop_last=len(train_ds) > bs)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"  Ratio: {ratio:.0%} | Labeled: {n_labeled_total:,} | "
          f"Unlabeled: {n_unlabeled_total:,} | Test: {len(test_ds):,}")

    return train_loader, test_loader, n_labeled_total, n_unlabeled_total


# ============================================================
# Semi-Supervised Loss
# ============================================================

def semi_supervised_loss(mscs_pred, mscs_target, w_min, cpi_golden, has_label):
    labeled_mask   = has_label > 0.5
    unlabeled_mask = ~labeled_mask
    n_lab = labeled_mask.sum().item()
    n_unl = unlabeled_mask.sum().item()

    total = torch.tensor(0.0, device=mscs_pred.device)
    loss_dict = {}

    if n_lab > 0:
        lp = mscs_pred[labeled_mask]
        lt = mscs_target[labeled_mask]
        lw = w_min[labeled_mask]
        lc = cpi_golden[labeled_mask]
        l_task_v  = loss_task(lp, lt)
        l_const_v = loss_const(lp, lw, lc)
        l_mono_v  = loss_mono(lp)
        total = total + l_task_v + LAMBDA_CONST * l_const_v + LAMBDA_MONO * l_mono_v
        loss_dict["L_task"] = l_task_v.item()
        loss_dict["L_const"] = l_const_v.item()
        loss_dict["L_mono_lab"] = l_mono_v.item()

    if n_unl > 0:
        up = mscs_pred[unlabeled_mask]
        uw = w_min[unlabeled_mask]
        u_const = loss_const_self(up, uw)
        u_mono  = loss_mono(up)
        total = total + LAMBDA_CONST * u_const + LAMBDA_MONO * u_mono
        loss_dict["L_const_self"] = u_const.item()
        loss_dict["L_mono_unl"] = u_mono.item()

    loss_dict["L_total"] = total.item()
    return total, loss_dict


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_gem(model, loader, device):
    model.eval()
    P, G = [], []
    for b in loader:
        m = model(b["gem_features"].to(device))
        P.append(compute_cpi_mean(m, b["w_min"].to(device)).cpu())
        G.append(b["cpi_golden"])
    return compute_mape(torch.cat(P), torch.cat(G))


# ============================================================
# Semi-Supervised Fine-tune
# ============================================================

def finetune_semi(model, train_loader, test_loader, device, label="PT-Semi-FT"):
    optimizer = optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [5000, 8000, 12000, 16000], gamma=0.5)

    best_mape, best_state, pat = float("inf"), None, 0
    step = 0
    history = []

    for ep in range(1, FT_EPOCHS + 1):
        model.train()
        ep_losses = {}
        nb = 0
        for b in train_loader:
            feat      = b["gem_features"].to(device)
            mscs_tgt  = b["mscs_target"].to(device)
            cpi_gold  = b["cpi_golden"].to(device)
            w_min     = b["w_min"].to(device)
            has_label = b["has_label"].to(device)

            mscs_pred = model(feat)
            loss, loss_dict = semi_supervised_loss(
                mscs_pred, mscs_tgt, w_min, cpi_gold, has_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            for k, v in loss_dict.items():
                ep_losses[k] = ep_losses.get(k, 0) + v
            nb += 1

        mape = evaluate_gem(model, test_loader, device)
        avg = {k: v / nb for k, v in ep_losses.items()}
        avg["test_mape"] = mape
        avg["epoch"] = ep
        avg["step"] = step
        history.append(avg)

        if mape < best_mape:
            best_mape = mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= FT_PATIENCE:
                print(f"  [{label}] Early stop ep {ep} (best={best_mape:.2f}%)")
                break

        if ep % 50 == 0 or ep == 1:
            print(f"  [{label}] ep {ep:4d} (step {step}) | "
                  f"L_task={avg.get('L_task',0):.5f} "
                  f"L_cself={avg.get('L_const_self',0):.5f} | "
                  f"MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
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
    assert os.path.exists(args.norm), f"Not found: {args.norm}"

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    ratio_tag = f"{int(args.ratio * 100)}pct"
    print(f"Device: {device}")
    print(f"Ratio: {args.ratio:.0%}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Normalizer: {args.norm}")
    print(f"Data: {GEM_DATA_DIR}")
    print(f"Output: {args.output_dir}")

    # Load pretrain normalizer
    gem_norm = FeatureNormalizer()
    nd = np.load(args.norm)
    gem_norm.mean, gem_norm.std = nd["mean"], nd["std"]

    # Create data split
    train_loader, test_loader, n_lab, n_unl = create_semi_split(
        GEM_DATA_DIR, args.ratio, gem_norm)

    # Load pretrained model
    model = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    zero = evaluate_gem(model, test_loader, device)
    print(f"  Zero-shot: {zero:.2f}%")

    # Semi-supervised fine-tune
    t0 = time.time()
    model, mape, hist = finetune_semi(
        model, train_loader, test_loader, device,
        label=f"PT-Semi-{ratio_tag}")

    # Save results
    result = {
        "labeled_ratio": args.ratio,
        "n_labeled": n_lab,
        "n_unlabeled": n_unl,
        "zero_shot": zero,
        "pretrain_semi_ft": mape,
        "history": hist,
    }

    out_path = os.path.join(args.output_dir, f"exp4_semi_{ratio_tag}_18k.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    torch.save(model.state_dict(),
               os.path.join(args.output_dir, f"pretrain_semi_ft_{ratio_tag}_18k.pt"))

    print(f"\n  {zero:.2f}% -> {mape:.2f}% ({time.time()-t0:.0f}s)")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
