"""
Exp2 Stage 2 Mixed (Server B): Semi-supervised fine-tune GEM foundation model.
Loads checkpoint from /path/to/gem5/micro_exp/exp2/foundations/{benchmark}/

Usage:
    python train_exp2_finetune_mix_serverB.py --gpu 0 --target 508.namd_r --n_samples 128
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
    GEM_INPUT_DIM,
    GEM_HIDDEN_DIMS, DROPOUT,
    BATCH_SIZE, PATIENCE,
    CONCORDE_LR, CONCORDE_WEIGHT_DECAY, CONCORDE_BATCH_SIZE, CONCORDE_EPOCHS,
    LAMBDA_CONST, LAMBDA_MONO,
    SEED,
    BASE_INDICES,
)
from model import (
    GEMModel,
    compute_cpi_mean, loss_task, loss_const, loss_mono,
)
from dataset import FeatureNormalizer, GEMDataset

GEM_DATA_DIR   = "/data/micro_traindata_packed_all/gem"
FOUNDATION_DIR = "/path/to/gem5/micro_exp/exp2/foundations"
OUTPUT_BASE    = "/path/to/gem5/micro_exp/exp2"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


def loss_const_self(mscs_pred, w_min, eps=1e-8):
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi_per_stage = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    cpi_mean = cpi_per_stage.mean(dim=1, keepdim=True)
    diff = cpi_per_stage - cpi_mean
    return (diff ** 2).mean()


def semi_supervised_loss(mscs_pred, mscs_target, w_min, cpi_golden, has_label):
    labeled_mask   = has_label > 0.5
    unlabeled_mask = ~labeled_mask
    n_labeled   = labeled_mask.sum().item()
    n_unlabeled = unlabeled_mask.sum().item()

    loss_dict = {}
    total = torch.tensor(0.0, device=mscs_pred.device)

    if n_labeled > 0:
        l_pred = mscs_pred[labeled_mask]
        l_tgt  = mscs_target[labeled_mask]
        l_wmin = w_min[labeled_mask]
        l_cpi  = cpi_golden[labeled_mask]

        l_task_val  = loss_task(l_pred, l_tgt)
        l_const_val = loss_const(l_pred, l_wmin, l_cpi)
        l_mono_val  = loss_mono(l_pred)

        labeled_loss = l_task_val + LAMBDA_CONST * l_const_val + LAMBDA_MONO * l_mono_val
        total = total + labeled_loss

        loss_dict["L_task"]     = l_task_val.item()
        loss_dict["L_const"]    = l_const_val.item()
        loss_dict["L_mono_lab"] = l_mono_val.item()

    if n_unlabeled > 0:
        u_pred = mscs_pred[unlabeled_mask]
        u_wmin = w_min[unlabeled_mask]

        u_const_self = loss_const_self(u_pred, u_wmin)
        u_mono_val   = loss_mono(u_pred)

        unlabeled_loss = LAMBDA_CONST * u_const_self + LAMBDA_MONO * u_mono_val
        total = total + unlabeled_loss

        loss_dict["L_const_self"] = u_const_self.item()
        loss_dict["L_mono_unl"]   = u_mono_val.item()

    loss_dict["L_total"]     = total.item()
    loss_dict["n_labeled"]   = n_labeled
    loss_dict["n_unlabeled"] = n_unlabeled
    return total, loss_dict


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


# ============================================================
# Data Split
# ============================================================

def create_target_mixed_split(target_data, target_name, n_samples, foundation_dir,
                              batch_size=BATCH_SIZE, seed=SEED):
    rng = np.random.RandomState(seed)
    n = target_data["n_samples"]
    idx = rng.permutation(n)
    split = min(n_samples, n - 1)

    bm_dir = os.path.join(foundation_dir, target_name)
    gem_norm = FeatureNormalizer()
    nd = np.load(os.path.join(bm_dir, "gem_norm.npz"))
    gem_norm.mean, gem_norm.std = nd["mean"], nd["std"]

    all_gem = gem_norm.transform(target_data["gem_features"])
    unlabeled_idx = idx[split:]

    # Train: all data in permuted order
    train_gem  = all_gem[idx]
    train_mscs = target_data["mscs_target"][idx]
    train_cpi  = target_data["cpi_golden"][idx]
    train_wmin = target_data["w_min"][idx]
    has_label  = np.zeros(n, dtype=np.float32)
    has_label[:split] = 1.0

    # Test: unlabeled portion
    test_gem  = all_gem[unlabeled_idx]
    test_con  = np.zeros((len(unlabeled_idx), 1), dtype=np.float32)
    test_mscs = target_data["mscs_target"][unlabeled_idx]
    test_cpi  = target_data["cpi_golden"][unlabeled_idx]
    test_wmin = target_data["w_min"][unlabeled_idx]

    train_ds = SemiSupervisedDataset(train_gem, train_mscs, train_cpi, train_wmin, has_label)
    test_ds  = GEMDataset(test_gem, test_con, test_mscs, test_cpi, test_wmin)

    actual_batch = min(batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=actual_batch, shuffle=True,
                              num_workers=4, pin_memory=True,
                              drop_last=len(train_ds) > actual_batch)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"    Train: {len(train_ds)} ({split} labeled + {n - split} unlabeled) | "
          f"Test: {len(test_ds)}")
    return train_loader, test_loader, split, len(unlabeled_idx)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_gem(model, loader, device):
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


# ============================================================
# Training
# ============================================================

def finetune_gem_mixed(model, train_loader, test_loader, device,
                       num_epochs=CONCORDE_EPOCHS, label="GEM-Mix-FT"):
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR,
                            weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    best_mape, best_state, patience_ctr = float("inf"), None, 0
    step = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = {}
        nb = 0
        for batch in train_loader:
            feat      = batch["gem_features"].to(device)
            mscs_tgt  = batch["mscs_target"].to(device)
            cpi_gold  = batch["cpi_golden"].to(device)
            w_min     = batch["w_min"].to(device)
            has_label = batch["has_label"].to(device)

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
                if isinstance(v, (int, float)):
                    epoch_losses[k] = epoch_losses.get(k, 0) + v
            nb += 1

        mape = evaluate_gem(model, test_loader, device)
        avg = {k: v / nb for k, v in epoch_losses.items()
               if k not in ("n_labeled", "n_unlabeled")}
        avg["test_mape"] = mape
        avg["epoch"] = epoch
        avg["step"] = step
        history.append(avg)

        if mape < best_mape:
            best_mape = mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"        [{label}] Early stop ep {epoch} (best={best_mape:.2f}%)")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"        [{label}] ep {epoch:3d} (step {step}) | "
                  f"L_task={avg.get('L_task',0):.6f} "
                  f"L_const={avg.get('L_const',0):.6f} "
                  f"L_cself={avg.get('L_const_self',0):.6f} "
                  f"L_mono_l={avg.get('L_mono_lab',0):.4f} "
                  f"L_mono_u={avg.get('L_mono_unl',0):.4f} | "
                  f"MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    return model, best_mape, history


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    ratio_tag = f"{args.n_samples}shot_mix_ft"
    if args.tag:
        ratio_tag = f"{ratio_tag}_{args.tag}"
    output_dir = os.path.join(OUTPUT_BASE, f"results_{ratio_tag}")

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    bm_dir = os.path.join(FOUNDATION_DIR, args.target)
    gem_ckpt = os.path.join(bm_dir, "gem_foundation.pt")
    assert os.path.exists(gem_ckpt), f"Not found: {gem_ckpt}"

    print(f"Device: {device}")
    print(f"Target: {args.target} | N labeled: {args.n_samples}")
    print(f"Foundation: {gem_ckpt}")
    print(f"Output: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading target benchmark...")
    from dataset import load_benchmark
    target_data = load_benchmark(args.target, GEM_DATA_DIR)
    print(f"    {args.target}: {target_data['n_samples']} samples")

    train_loader, test_loader, n_labeled, n_unlabeled = create_target_mixed_split(
        target_data, args.target, args.n_samples, FOUNDATION_DIR)

    results = {"target": args.target, "n_samples": args.n_samples,
               "n_labeled": n_labeled, "n_unlabeled": n_unlabeled}

    model = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(device)
    model.load_state_dict(torch.load(gem_ckpt, map_location=device))
    zero_shot = evaluate_gem(model, test_loader, device)
    print(f"    [{args.target}] Zero-shot: {zero_shot:.2f}%")
    results["gem_zero_shot"] = zero_shot

    print(f"    [{args.target}] Mixed semi-supervised fine-tune...")
    t0 = time.time()
    model, mape, hist = finetune_gem_mixed(
        model, train_loader, test_loader, device,
        label=f"{args.target} GEM-Mix-FT")
    results["gem_mix_ft"] = mape
    results["gem_mix_ft_history"] = hist
    torch.save(model.state_dict(), os.path.join(output_dir, f"gem_mix_ft_{args.target}.pt"))
    elapsed = time.time() - t0

    print(f"\n    [{args.target}] Result ({args.n_samples} labeled):")
    print(f"      Zero-shot: {zero_shot:.2f}%")
    print(f"      Mix-FT:    {mape:.2f}% ({elapsed:.0f}s)")

    json_path = os.path.join(output_dir, f"exp2_ood_{args.target}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Saved: {json_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
