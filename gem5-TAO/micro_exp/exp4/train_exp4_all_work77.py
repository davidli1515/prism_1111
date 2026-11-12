"""
Experiment 4: Label Efficiency — Semi-Supervised GEM

Ablation modes (--ablation):
  full, const_only, mono_only, supervised_only, concorde

Dataset sizes (--dataset):
  18k, 40k — controls output directory naming

Output: results/exp4_{dataset}/{ablation}/exp4_*.json
  e.g. results/exp4_40k/full/exp4_20pct.json

Usage:
    python train_exp4_all.py --dataset 40k --ratio 0.20 --gpu 0
    python train_exp4_all.py --dataset 18k --ablation concorde --ratio 0.05 --gpu 1

Imports from exp1: /scratch/gem5-TAO/micro_exp/exp1/
"""

import os
import sys
import time
import json
import hashlib
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")

from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM, MSCS_DIM,
    GEM_HIDDEN_DIMS, CONCORDE_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS,
    CONCORDE_LR, CONCORDE_WEIGHT_DECAY, CONCORDE_BATCH_SIZE, CONCORDE_EPOCHS,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    SEED, TRAIN_RATIO,
    BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES,
    WMIN_INDICES, CONCORDE_FEAT_END, D3_FEAT_END, MSCS_START, MSCS_END,
)
from model import (
    GEMModel, ConcordeModel, compute_cpi_mean,
    loss_task, loss_const, loss_mono,
)
from dataset import (
    load_all_benchmarks, FeatureNormalizer, GEMDataset,
)

try:
    from config import STEP_MILESTONES
except ImportError:
    STEP_MILESTONES = [10000, 14000, 18000, 22000]

try:
    from config import LR_GAMMA
except ImportError:
    LR_GAMMA = 0.5

# ============================================================
# Constants
# ============================================================

ALL_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
ABLATION_MODES = ["full", "const_only", "mono_only", "supervised_only", "concorde"]
DATASET_CHOICES = ["18k", "40k"]

# Dataset → data directory mapping
DATASET_PATHS = {
    "18k": {
        "gem":     "/scratch2/wl730/micro_traindata_packed/gem",
        "concord": "/scratch2/wl730/micro_traindata_packed/concord",
    },
    "40k": {
        "gem":     "/scratch2/wl730/micro_traindata_packed_all/gem",
        "concord": "/scratch2/wl730/micro_traindata_packed_all/concord",
    },
}

# Dataset-specific MLP sizes (best config per dataset)
DATASET_MODEL_DIMS = {
    "18k": {
        "gem_hidden":      [512, 256],
        "concorde_hidden": [256, 128],
    },
    "40k": {
        "gem_hidden":      [512, 256],
        "concorde_hidden": [512, 256],
    },
}


def get_output_dir(dataset, ablation, results_base="./results"):
    """
    Output directory structure:
      {results_base}/exp4_{dataset}/              (ablation=full)
      {results_base}/exp4_{dataset}_{ablation}/   (others)
    """
    base = f"{results_base}/exp4_{dataset}"
    if ablation == "full":
        return base
    return f"{base}_{ablation}"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def stable_hash(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)


# ============================================================
# Self-Supervised L_const
# ============================================================

def loss_const_self(mscs_pred, w_min, eps=1e-8):
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi_per_stage = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    cpi_mean = cpi_per_stage.mean(dim=1, keepdim=True)
    diff = cpi_per_stage - cpi_mean
    return (diff ** 2).mean()


# ============================================================
# Dataset
# ============================================================

class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, gem_features, concord_features, mscs_target,
                 cpi_golden, w_min, has_label):
        self.gem_features     = torch.from_numpy(gem_features)
        self.concord_features = torch.from_numpy(concord_features)
        self.mscs_target      = torch.from_numpy(mscs_target)
        self.cpi_golden       = torch.from_numpy(cpi_golden)
        self.w_min            = torch.from_numpy(w_min)
        self.has_label        = torch.from_numpy(has_label.astype(np.float32))

    def __len__(self):
        return self.gem_features.shape[0]

    def __getitem__(self, idx):
        return {
            "gem_features":     self.gem_features[idx],
            "concord_features": self.concord_features[idx],
            "mscs_target":      self.mscs_target[idx],
            "cpi_golden":       self.cpi_golden[idx],
            "w_min":            self.w_min[idx],
            "has_label":        self.has_label[idx],
        }


# ============================================================
# DataLoader
# ============================================================

def create_exp4_dataloaders(all_benchmark_data, labeled_ratio,
                            labeled_only=False,
                            batch_size=CONCORDE_BATCH_SIZE, seed=SEED):
    rng_split = np.random.RandomState(seed)

    train_gem, train_con, train_mscs, train_cpi, train_wmin = [], [], [], [], []
    train_has_label = []
    test_gem, test_con, test_mscs, test_cpi, test_wmin = [], [], [], [], []
    per_benchmark_test = {}

    test_offset = 0
    total_labeled = 0
    total_unlabeled = 0

    for name, data in sorted(all_benchmark_data.items()):
        n = data["n_samples"]
        indices = rng_split.permutation(n)
        split = int(n * TRAIN_RATIO)

        tr_idx = indices[:split]
        te_idx = indices[split:]

        n_train = len(tr_idx)
        n_labeled = max(1, int(n_train * labeled_ratio))

        label_rng = np.random.RandomState(
            seed + stable_hash(name) + int(round(labeled_ratio * 1000))
        )
        label_perm = label_rng.permutation(n_train)
        has_label = np.zeros(n_train, dtype=np.float32)
        has_label[label_perm[:n_labeled]] = 1.0

        total_labeled += n_labeled
        total_unlabeled += (n_train - n_labeled)

        train_gem.append(data["gem_features"][tr_idx])
        train_con.append(data["concord_features"][tr_idx])
        train_mscs.append(data["mscs_target"][tr_idx])
        train_cpi.append(data["cpi_golden"][tr_idx])
        train_wmin.append(data["w_min"][tr_idx])
        train_has_label.append(has_label)

        test_gem.append(data["gem_features"][te_idx])
        test_con.append(data["concord_features"][te_idx])
        test_mscs.append(data["mscs_target"][te_idx])
        test_cpi.append(data["cpi_golden"][te_idx])
        test_wmin.append(data["w_min"][te_idx])

        n_test = len(te_idx)
        per_benchmark_test[name] = (test_offset, test_offset + n_test)
        test_offset += n_test

    train_gem       = np.concatenate(train_gem)
    train_con       = np.concatenate(train_con)
    train_mscs      = np.concatenate(train_mscs)
    train_cpi       = np.concatenate(train_cpi)
    train_wmin      = np.concatenate(train_wmin)
    train_has_label = np.concatenate(train_has_label)

    test_gem  = np.concatenate(test_gem)
    test_con  = np.concatenate(test_con)
    test_mscs = np.concatenate(test_mscs)
    test_cpi  = np.concatenate(test_cpi)
    test_wmin = np.concatenate(test_wmin)

    gem_normalizer = FeatureNormalizer()
    train_gem = gem_normalizer.fit_transform(train_gem)
    test_gem  = gem_normalizer.transform(test_gem)

    concord_normalizer = FeatureNormalizer()
    train_con = concord_normalizer.fit_transform(train_con)
    test_con  = concord_normalizer.transform(test_con)

    if labeled_only:
        mask = train_has_label > 0.5
        train_gem       = train_gem[mask]
        train_con       = train_con[mask]
        train_mscs      = train_mscs[mask]
        train_cpi       = train_cpi[mask]
        train_wmin      = train_wmin[mask]
        train_has_label = train_has_label[mask]
        actual_unlabeled = 0
        print(f"  [labeled_only] Filtered to {mask.sum():,} labeled samples")
    else:
        actual_unlabeled = total_unlabeled

    train_ds = SemiSupervisedDataset(
        train_gem, train_con, train_mscs, train_cpi, train_wmin, train_has_label
    )
    test_ds = GEMDataset(test_gem, test_con, test_mscs, test_cpi, test_wmin)

    actual_batch_size = min(batch_size, len(train_ds))
    drop_last = len(train_ds) > actual_batch_size

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=actual_batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=drop_last,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"  Labeled ratio: {labeled_ratio*100:.0f}%")
    print(f"  Labeled: {total_labeled:,} | Unlabeled in train: {actual_unlabeled:,} | "
          f"Train size: {len(train_ds):,} | Batch: {actual_batch_size:,}")
    print(f"  Test: {len(test_ds):,}")

    return (train_loader, test_loader, gem_normalizer,
            per_benchmark_test, total_labeled, actual_unlabeled)


# ============================================================
# Optimizer & Step-based LR
# ============================================================

def create_optimizer(model):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    print(f"  Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"  Scheduler: MultiStepLR per-step (milestones={STEP_MILESTONES}, gamma={LR_GAMMA})")
    return optimizer


def step_lr(optimizer, global_step):
    n_passed = sum(1 for m in STEP_MILESTONES if global_step >= m)
    new_lr = LEARNING_RATE * (LR_GAMMA ** n_passed)
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr


# ============================================================
# Semi-Supervised Loss
# ============================================================

def semi_supervised_loss(mscs_pred, mscs_target, w_min, cpi_golden,
                         has_label, ablation="full",
                         lambda_const=LAMBDA_CONST,
                         lambda_mono=LAMBDA_MONO):
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

        l_task     = loss_task(l_pred, l_tgt)
        l_const    = loss_const(l_pred, l_wmin, l_cpi)
        l_mono_val = loss_mono(l_pred)

        labeled_loss = l_task + lambda_const * l_const + lambda_mono * l_mono_val

        loss_dict["L_task"]     = l_task.item()
        loss_dict["L_const"]    = l_const.item()
        loss_dict["L_mono_lab"] = l_mono_val.item()
        loss_dict["L_labeled"]  = labeled_loss.item()

        total = total + labeled_loss

    if n_unlabeled > 0 and ablation not in ("supervised_only",):
        u_pred = mscs_pred[unlabeled_mask]
        u_wmin = w_min[unlabeled_mask]

        unlabeled_loss = torch.tensor(0.0, device=mscs_pred.device)

        if ablation in ("full", "const_only"):
            u_const_self = loss_const_self(u_pred, u_wmin)
            unlabeled_loss = unlabeled_loss + lambda_const * u_const_self
            loss_dict["L_const_self"] = u_const_self.item()

        if ablation in ("full", "mono_only"):
            u_mono_val = loss_mono(u_pred)
            unlabeled_loss = unlabeled_loss + lambda_mono * u_mono_val
            loss_dict["L_mono_unl"] = u_mono_val.item()

        loss_dict["L_unlabeled"] = unlabeled_loss.item()
        total = total + unlabeled_loss

    loss_dict["L_total"]     = total.item()
    loss_dict["n_labeled"]   = n_labeled
    loss_dict["n_unlabeled"] = n_unlabeled

    return total, loss_dict


# ============================================================
# Evaluation
# ============================================================

def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


@torch.no_grad()
def evaluate_gem(model, loader, device):
    model.eval()
    all_cpi_pred, all_cpi_gold = [], []
    for batch in loader:
        gem_feat = batch["gem_features"].to(device)
        w_min    = batch["w_min"].to(device)
        cpi_gold = batch["cpi_golden"].to(device)
        mscs_pred = model(gem_feat)
        cpi_pred  = compute_cpi_mean(mscs_pred, w_min)
        all_cpi_pred.append(cpi_pred.cpu())
        all_cpi_gold.append(cpi_gold.cpu())
    all_cpi_pred = torch.cat(all_cpi_pred)
    all_cpi_gold = torch.cat(all_cpi_gold)
    return compute_mape(all_cpi_pred, all_cpi_gold), all_cpi_pred, all_cpi_gold


@torch.no_grad()
def evaluate_concorde(model, loader, device):
    model.eval()
    all_pred, all_gold = [], []
    for batch in loader:
        con_feat = batch["concord_features"].to(device)
        cpi_gold = batch["cpi_golden"].to(device)
        cpi_pred = model(con_feat)
        all_pred.append(cpi_pred.cpu())
        all_gold.append(cpi_gold.cpu())
    all_pred = torch.cat(all_pred)
    all_gold = torch.cat(all_gold)
    return compute_mape(all_pred, all_gold), all_pred, all_gold


def per_benchmark_mape(cpi_pred, cpi_gold, benchmark_indices):
    results = {}
    for name, (start, end) in benchmark_indices.items():
        if start == end:
            continue
        results[name] = compute_mape(cpi_pred[start:end], cpi_gold[start:end])
    return results


# ============================================================
# Training Loops
# ============================================================

def train_gem_semi(model, train_loader, test_loader, device,
                   benchmark_test_indices, labeled_ratio,
                   ablation="full", num_epochs=1500):
    optimizer = create_optimizer(model)

    best_mape = float("inf")
    best_state = None
    global_step = 0

    print(f"\n{'='*60}")
    print(f"Training GEM — labeled={labeled_ratio*100:.0f}%  ablation={ablation}")
    print(f"  Fixed {num_epochs} epochs (no early stopping)")
    print(f"{'='*60}")

    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch in train_loader:
            gem_feat  = batch["gem_features"].to(device)
            mscs_tgt  = batch["mscs_target"].to(device)
            cpi_gold  = batch["cpi_golden"].to(device)
            w_min     = batch["w_min"].to(device)
            has_label = batch["has_label"].to(device)

            mscs_pred = model(gem_feat)
            loss, loss_dict = semi_supervised_loss(
                mscs_pred, mscs_tgt, w_min, cpi_gold, has_label,
                ablation=ablation,
                lambda_const=LAMBDA_CONST, lambda_mono=LAMBDA_MONO,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            step_lr(optimizer, global_step)

            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    epoch_losses[k] += v
            n_batches += 1

        # Track best model
        test_mape, _, _ = evaluate_gem(model, test_loader, device)
        if test_mape < best_mape:
            best_mape = test_mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            avg = {k: v / n_batches for k, v in epoch_losses.items()
                   if k not in ("n_labeled", "n_unlabeled")}
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d} | "
                  f"L_task={avg.get('L_task', 0):.6f} "
                  f"L_const={avg.get('L_const', 0):.6f} "
                  f"L_cself={avg.get('L_const_self', 0):.6f} "
                  f"L_mono={avg.get('L_mono_unl', 0):.6f} | "
                  f"Test MAPE={test_mape:.2f}% best={best_mape:.2f}% | "
                  f"lr={lr:.2e} step={global_step} | {elapsed:.0f}s")

    model.load_state_dict(best_state)
    total_time = time.time() - t0
    print(f"  Best test MAPE: {best_mape:.2f}% (total time: {total_time:.0f}s)")
    return model, best_mape, total_time


def train_concorde(model, train_loader, test_loader, device,
                   labeled_ratio, num_epochs=1500):
    optimizer = create_optimizer(model)

    def relative_loss(pred, target, eps=1e-8):
        return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()

    best_mape = float("inf")
    best_state = None
    global_step = 0

    print(f"\n{'='*60}")
    print(f"Training Concorde — labeled={labeled_ratio*100:.0f}%")
    print(f"  Fixed {num_epochs} epochs (no early stopping)")
    print(f"{'='*60}")

    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            con_feat = batch["concord_features"].to(device)
            cpi_gold = batch["cpi_golden"].to(device)

            cpi_pred = model(con_feat)
            loss = relative_loss(cpi_pred, cpi_gold)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            step_lr(optimizer, global_step)

            epoch_loss += loss.item()
            n_batches += 1

        # Track best model
        test_mape, _, _ = evaluate_concorde(model, test_loader, device)
        if test_mape < best_mape:
            best_mape = test_mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = epoch_loss / n_batches
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d} | "
                  f"Loss={avg_loss:.6f} | "
                  f"Test MAPE={test_mape:.2f}% best={best_mape:.2f}% | "
                  f"lr={lr:.2e} step={global_step} | {elapsed:.0f}s")

    model.load_state_dict(best_state)
    total_time = time.time() - t0
    print(f"  Best Concorde test MAPE: {best_mape:.2f}% (total time: {total_time:.0f}s)")
    return model, best_mape, total_time


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Exp4: Label Efficiency")
    parser.add_argument("--ratio", type=float, default=None,
                        help="Single labeled ratio. If not set, runs all.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ablation", type=str, default="full",
                        choices=ABLATION_MODES)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=DATASET_CHOICES,
                        help="Dataset size tag: 18k or 40k")
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Base directory for results (default: ./results)")
    args = parser.parse_args()

    ratios = [args.ratio] if args.ratio is not None else ALL_RATIOS
    ablation = args.ablation
    is_concorde = (ablation == "concorde")
    is_labeled_only = ablation in ("supervised_only", "concorde")

    out_dir = get_output_dir(args.dataset, ablation, args.results_dir)

    # Resolve data paths based on dataset choice
    gem_data_dir     = DATASET_PATHS[args.dataset]["gem"]
    concord_data_dir = DATASET_PATHS[args.dataset]["concord"]

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"  GEM data:     {gem_data_dir}")
    print(f"  Concord data: {concord_data_dir}")
    print(f"Ablation: {ablation}")
    print(f"Ratios: {[f'{r*100:.0f}%' for r in ratios]}")
    print(f"Output: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1/2] Loading benchmarks...")
    all_data = load_all_benchmarks(gem_data_dir, concord_data_dir, BENCHMARK_NAMES)
    assert len(all_data) > 0, "No benchmarks found!"

    for ratio in ratios:
        ratio_key = f"{int(round(ratio*100))}pct"

        print(f"\n{'#'*60}")
        print(f"# Dataset={args.dataset} Labeled={ratio*100:.0f}% "
              f"Ablation={ablation} GPU={args.gpu}")
        print(f"{'#'*60}")

        set_seed(SEED)

        (train_loader, test_loader, gem_normalizer,
         benchmark_test_indices,
         n_labeled, n_unlabeled) = create_exp4_dataloaders(
            all_data, ratio, labeled_only=is_labeled_only
        )

        print("\n[2/2] Training...")

        # Dataset-specific model dimensions
        gem_hdims = DATASET_MODEL_DIMS[args.dataset]["gem_hidden"]
        con_hdims = DATASET_MODEL_DIMS[args.dataset]["concorde_hidden"]

        if is_concorde:
            model = ConcordeModel(
                input_dim=CONCORDE_INPUT_DIM,
                hidden_dims=con_hdims,
                dropout=0.0,
            ).to(device)
            print(f"  Concorde arch: {CONCORDE_INPUT_DIM} → {con_hdims} → 1")
            print(f"  Concorde params: {sum(p.numel() for p in model.parameters()):,}")

            model, best_mape, train_time = train_concorde(
                model, train_loader, test_loader, device, ratio,
            )
            test_mape, cpi_pred, cpi_gold = evaluate_concorde(
                model, test_loader, device
            )
        else:
            model = GEMModel(
                input_dim=GEM_INPUT_DIM,
                hidden_dims=gem_hdims,
                dropout=DROPOUT,
            ).to(device)
            print(f"  GEM arch: {GEM_INPUT_DIM} → {gem_hdims} → 16")
            print(f"  GEM params: {sum(p.numel() for p in model.parameters()):,}")

            model, best_mape, train_time = train_gem_semi(
                model, train_loader, test_loader, device,
                benchmark_test_indices, ratio,
                ablation=ablation,
            )
            test_mape, cpi_pred, cpi_gold = evaluate_gem(
                model, test_loader, device
            )

        per_bm = per_benchmark_mape(cpi_pred, cpi_gold, benchmark_test_indices)

        print(f"\nResults for dataset={args.dataset} labeled={ratio*100:.0f}% "
              f"ablation={ablation}:")
        print(f"  Overall MAPE: {test_mape:.2f}%")
        print(f"  {'Benchmark':<20} {'MAPE%':>10}")
        print(f"  {'-'*32}")
        for name in sorted(per_bm.keys()):
            print(f"  {name:<20} {per_bm[name]:>9.2f}%")

        result = {
            "dataset": args.dataset,
            "labeled_ratio": ratio,
            "ablation": ablation,
            "n_labeled": n_labeled,
            "n_unlabeled": n_unlabeled,
            "overall_mape": test_mape,
            "per_benchmark": per_bm,
            "train_time_seconds": train_time,
        }

        result_path = os.path.join(out_dir, f"exp4_{ratio_key}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Result saved to {result_path}")

        ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(ckpt_dir, f"{ablation}_{ratio_key}.pt"))

    print("\nDone.")


if __name__ == "__main__":
    main()