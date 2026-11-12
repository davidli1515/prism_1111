"""
Experiment 4: Label Efficiency — Semi-Supervised GEM / PRISM
Recipe: exp1_v3_run2 (AdamW, dynamic milestones, λ_task=0.1, L_const=relative error)

Ablation modes (--ablation):
  full, const_only, mono_only, supervised_only, concorde

Usage:
    python train_exp4_all.py --dataset 18k --ratio 0.20 --gpu 0
    python train_exp4_all.py --dataset 18k --ablation concorde --ratio 0.50 --gpu 1
"""

import os, sys, time, json, hashlib, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")

from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM, MSCS_DIM,
    DROPOUT, BENCHMARK_NAMES,
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

# ============================================================
# Recipe v77 (matching exp1_v3_run2)
# ============================================================
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.3
BATCH_SIZE = 50000
NUM_EPOCHS = 1521
PATIENCE = 100
LR_GAMMA = 0.5
MILESTONE_FRACS = [0.42, 0.58, 0.75, 0.92]

# Fixed milestones computed from full 18k dataset (train=217476, batch=50000, epochs=1521)
# All variants use these regardless of dataloader size for fair comparison
FIXED_MILESTONES_18K = [2555, 3528, 4563, 5597]

# Loss weights: λ_task=0.1, λ_const=1.0, λ_mono=0.01
LAMBDA_TASK = 0.1
LAMBDA_CONST = 1.0
LAMBDA_MONO = 0.01
LAMBDA_UNLAB = 1.0   # separate weight for unlabeled physics loss
RAMP_EPOCHS = 0      # linear ramp-up epochs for unlabeled loss (0 = disabled)

ALL_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
              0.60, 0.70, 0.80, 0.90]
ABLATION_MODES = ["full", "const_only", "mono_only", "supervised_only", "concorde"]
DATASET_CHOICES = ["18k", "40k"]

DATASET_PATHS = {
    "18k": {
        "gem":     "/data/micro_traindata_packed/gem",
        "concord": "/data/micro_traindata_packed/concord",
    },
    "40k": {
        "gem":     "/data/micro_traindata_packed_all/gem",
        "concord": "/data/micro_traindata_packed_all/concord",
    },
}

# All [512, 256] — unified across datasets
DATASET_MODEL_DIMS = {
    "18k": {"gem_hidden": [512, 256], "concorde_hidden": [512, 256]},
    "40k": {"gem_hidden": [512, 256], "concorde_hidden": [512, 256]},
}


def get_output_dir(dataset, ablation, results_base="./results"):
    base = f"{results_base}/exp4_{dataset}"
    if ablation == "full":
        return base
    return f"{base}_{ablation}"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); torch.backends.cudnn.deterministic = True

def stable_hash(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)


# ============================================================
# L_const_self (no golden CPI needed, for unlabeled data)
# ============================================================

def loss_const_self(mscs_pred, w_min, eps=1e-8):
    """Self-consistency: variance of CPI across 4 stages (no golden needed)"""
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi_per_stage = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    cpi_mean = cpi_per_stage.mean(dim=1, keepdim=True)
    diff = cpi_per_stage - cpi_mean
    return (diff ** 2).mean()


def weighted_loss_task(mscs_pred, mscs_target, base_weight=1.0,
                       adaptive=False, w_min=0.3, w_max=5.0,
                       renorm_total=False, eps=1e-8):
    """Three decoupled knobs:
    base_weight : static per-dim emphasis on Base dims
    adaptive    : per-sample targeting via batch-normalized relative error,
                  hard samples >1, easy <1, batch mean ~1.
                  Does NOT decay as training progresses.
    renorm_total: True = pure redistribution (no L_task inflation),
                  preserves L_const's anchor share under grad_clip.
    Returns (loss, info_dict).
    """
    err2 = (mscs_pred - mscs_target) ** 2          # (B, D)
    D = mscs_pred.shape[1]

    dim_w = torch.ones(D, device=mscs_pred.device)
    dim_w[BASE_INDICES] = base_weight
    W = dim_w.unsqueeze(0).expand_as(err2).clone()  # (B, D)

    info = {}
    if adaptive:
        with torch.no_grad():
            bp = mscs_pred[:, BASE_INDICES]
            bt = mscs_target[:, BASE_INDICES]
            rel = (torch.abs(bp - bt) / (torch.abs(bt) + eps)).mean(dim=1)  # (B,)
            sample_w = (rel / (rel.mean() + eps)).clamp(w_min, w_max)
            # Apply targeting ONLY to Base dims
            W[:, BASE_INDICES] = W[:, BASE_INDICES] * sample_w.unsqueeze(1)
            info = {"adapt_w_mean": sample_w.mean().item(),
                    "adapt_w_max":  sample_w.max().item(),
                    "adapt_w_min":  sample_w.min().item(),
                    "base_rel_err_mean": rel.mean().item()}

    weighted = W * err2
    if renorm_total:
        with torch.no_grad():
            scale = err2.mean() / (weighted.mean() + eps)
            weighted = weighted * scale
    return weighted.mean(), info


# ============================================================
# ReLoBRaLo: Relative Loss Balancing with Random Lookback
# (Bischof & Kraus, CMAME 2025)
# ============================================================

class ReLoBRaLo:
    """
    Adaptive loss weighting for multi-objective training.
    Tracks relative change of each loss term and rebalances via softmax.
    
    Args:
        n_losses: number of loss terms to balance
        alpha: exponential smoothing factor (higher = more stable)
        temperature: softmax temperature (lower = more aggressive rebalancing)
        lookback_max: max random lookback window size
    """
    def __init__(self, n_losses, alpha=0.999, temperature=1.0, lookback_max=10):
        self.n_losses = n_losses
        self.alpha = alpha
        self.temperature = temperature
        self.lookback_max = lookback_max
        self.history = [[] for _ in range(n_losses)]
        self.weights = np.ones(n_losses)  # start with equal weights

    def update(self, losses):
        """
        Update weights based on relative loss changes.
        Args: losses: list of float, current loss values for each term
        Returns: np.array of updated weights
        """
        for i, l in enumerate(losses):
            self.history[i].append(l)

        if len(self.history[0]) < 2:
            return self.weights

        # Random lookback
        max_tau = min(len(self.history[0]) - 1, self.lookback_max)
        tau = np.random.randint(1, max_tau + 1)

        # Relative change: loss_now / loss_past
        rho = np.zeros(self.n_losses)
        for i in range(self.n_losses):
            prev = self.history[i][-1 - tau]
            curr = self.history[i][-1]
            rho[i] = curr / (prev + 1e-12)

        # Softmax to get normalized weights, scaled by n_losses
        rho = rho / self.temperature
        exp_rho = np.exp(rho - np.max(rho))
        new_weights = exp_rho / exp_rho.sum() * self.n_losses

        # Exponential moving average for stability
        self.weights = self.alpha * self.weights + (1.0 - self.alpha) * new_weights

        return self.weights

    def get_weights(self):
        return self.weights


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
                            batch_size=BATCH_SIZE, seed=SEED):
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
# Dynamic LR Schedule
# ============================================================

def get_milestones(dataset):
    """Return fixed milestones for the given dataset."""
    if dataset == "18k":
        return FIXED_MILESTONES_18K
    else:
        # 40k: compute dynamically (only needed if running 40k)
        return None  # will be computed at train time


def step_lr(optimizer, global_step, milestones):
    n_passed = sum(1 for m in milestones if global_step >= m)
    new_lr = LEARNING_RATE * (LR_GAMMA ** n_passed)
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr


# ============================================================
# Semi-Supervised Loss (λ_task=0.1, L_const=relative error)
# ============================================================

def semi_supervised_loss(mscs_pred, mscs_target, w_min, cpi_golden,
                         has_label, ablation="full",
                         lambda_task=LAMBDA_TASK, lambda_const=LAMBDA_CONST,
                         lambda_mono=LAMBDA_MONO, unlab_scale=1.0,
                         base_weight=1.0, adaptive_task=False, w_max=5.0,
                         adapt_w_min=0.3, renorm_total=False,
                         no_mono_labeled=False):
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

        l_task_result = weighted_loss_task(l_pred, l_tgt, base_weight,
                                          adaptive=adaptive_task, w_min=adapt_w_min,
                                          w_max=w_max, renorm_total=renorm_total)
        if adaptive_task or base_weight != 1.0:
            l_task, adapt_stats = l_task_result
            loss_dict.update(adapt_stats)
        else:
            l_task, _ = l_task_result
        l_const    = loss_const(l_pred, l_wmin, l_cpi)
        l_mono_val = loss_mono(l_pred)

        labeled_loss = (lambda_task * l_task
                        + lambda_const * l_const)
        if not no_mono_labeled:
            labeled_loss = labeled_loss + lambda_mono * l_mono_val

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
        total = total + unlab_scale * unlabeled_loss

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
    return compute_mape(torch.cat(all_cpi_pred), torch.cat(all_cpi_gold)), \
           torch.cat(all_cpi_pred), torch.cat(all_cpi_gold)

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
    return compute_mape(torch.cat(all_pred), torch.cat(all_gold)), \
           torch.cat(all_pred), torch.cat(all_gold)

def per_benchmark_mape(cpi_pred, cpi_gold, benchmark_indices):
    results = {}
    for name, (start, end) in benchmark_indices.items():
        if start == end: continue
        results[name] = compute_mape(cpi_pred[start:end], cpi_gold[start:end])
    return results


# ============================================================
# Training Loops (with early stopping + dynamic milestones)
# ============================================================

def train_gem_semi(model, train_loader, test_loader, device,
                   benchmark_test_indices, labeled_ratio,
                   ablation="full", num_epochs=NUM_EPOCHS, dataset="18k",
                   use_relobralo=False, use_pcgrad=False, use_swa=False,
                   lambda_unlab=LAMBDA_UNLAB, ramp_epochs=RAMP_EPOCHS,
                   base_weight=1.0, adaptive_task=False, w_max=5.0,
                   adapt_w_min=0.3, renorm_total=False,
                   no_mono_labeled=False):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    milestones = FIXED_MILESTONES_18K

    # ReLoBRaLo
    balancer = None
    if use_relobralo:
        balancer = ReLoBRaLo(n_losses=3, alpha=0.95, temperature=0.5,
                             lookback_max=10)
        balancer.weights = np.array([LAMBDA_TASK, LAMBDA_CONST, LAMBDA_MONO])

    # SWA: start at 85% of training
    swa_model = None
    swa_start = int(num_epochs * 0.85)
    swa_count = 0
    if use_swa:
        from copy import deepcopy
        swa_state = None  # will init on first SWA update

    best_mape = float("inf")
    best_state = None
    global_step = 0

    print(f"\n{'='*60}")
    print(f"Training PRISM — labeled={labeled_ratio*100:.0f}%  ablation={ablation}")
    print(f"  {num_epochs} epochs, no early stopping, steps/epoch={steps_per_epoch}")
    print(f"  Fixed milestones={milestones}")
    flags = []
    if use_pcgrad:  flags.append("PCGrad")
    if use_swa:     flags.append(f"SWA(start={swa_start})")
    if use_relobralo: flags.append("ReLoBRaLo")
    if lambda_unlab != 1.0: flags.append(f"λ_unlab={lambda_unlab}")
    if ramp_epochs > 0: flags.append(f"ramp={ramp_epochs}ep")
    if flags:
        print(f"  Enabled: {', '.join(flags)}")
    else:
        print(f"  Loss: {LAMBDA_TASK}×L_task + {LAMBDA_CONST}×L_const(rel) + {LAMBDA_MONO}×L_mono")
    print(f"{'='*60}")

    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        # Ramp-up: linearly increase unlabeled weight from 0 to lambda_unlab
        if ramp_epochs > 0 and epoch <= ramp_epochs:
            unlab_scale = lambda_unlab * (epoch / ramp_epochs)
        else:
            unlab_scale = lambda_unlab

        for batch in train_loader:
            gem_feat  = batch["gem_features"].to(device)
            mscs_tgt  = batch["mscs_target"].to(device)
            cpi_gold  = batch["cpi_golden"].to(device)
            w_min     = batch["w_min"].to(device)
            has_label = batch["has_label"].to(device)

            # Get current loss weights
            if balancer is not None:
                w = balancer.get_weights()
                lt, lc, lm = float(w[0]), float(w[1]), float(w[2])
            else:
                lt, lc, lm = LAMBDA_TASK, LAMBDA_CONST, LAMBDA_MONO

            if use_pcgrad:
                # === PCGrad: separate backward for L_task vs L_physics ===
                labeled_mask   = has_label > 0.5
                unlabeled_mask = ~labeled_mask
                n_lab = labeled_mask.sum().item()
                n_unl = unlabeled_mask.sum().item()

                mscs_pred = model(gem_feat)
                loss_dict = {}

                # Compute individual losses
                l_task_val = torch.tensor(0.0, device=device)
                l_phys_val = torch.tensor(0.0, device=device)

                if n_lab > 0:
                    l_pred = mscs_pred[labeled_mask]
                    l_tgt  = mscs_tgt[labeled_mask]
                    l_wmin = w_min[labeled_mask]
                    l_cpi  = cpi_gold[labeled_mask]

                    l_task_raw  = loss_task(l_pred, l_tgt)
                    l_const_raw = loss_const(l_pred, l_wmin, l_cpi)
                    l_mono_raw  = loss_mono(l_pred)

                    l_task_val = lt * l_task_raw
                    l_phys_val = l_phys_val + lc * l_const_raw + lm * l_mono_raw

                    loss_dict["L_task"]  = l_task_raw.item()
                    loss_dict["L_const"] = l_const_raw.item()

                if n_unl > 0 and ablation not in ("supervised_only",):
                    u_pred = mscs_pred[unlabeled_mask]
                    u_wmin = w_min[unlabeled_mask]

                    if ablation in ("full", "const_only"):
                        u_cs = loss_const_self(u_pred, u_wmin)
                        l_phys_val = l_phys_val + lc * u_cs
                        loss_dict["L_const_self"] = u_cs.item()
                    if ablation in ("full", "mono_only"):
                        u_mo = loss_mono(u_pred)
                        l_phys_val = l_phys_val + lm * u_mo
                        loss_dict["L_mono_unl"] = u_mo.item()

                loss_dict["L_total"] = (l_task_val + l_phys_val).item()

                # Backward pass 1: L_task gradient
                optimizer.zero_grad()
                if l_task_val.requires_grad:
                    l_task_val.backward(retain_graph=True)
                g_task = []
                for p in model.parameters():
                    g_task.append(p.grad.clone() if p.grad is not None
                                 else torch.zeros_like(p))

                # Backward pass 2: L_physics gradient
                optimizer.zero_grad()
                if l_phys_val.requires_grad:
                    l_phys_val.backward()
                g_phys = []
                for p in model.parameters():
                    g_phys.append(p.grad.clone() if p.grad is not None
                                 else torch.zeros_like(p))

                # Project: if conflict, remove conflicting component of g_task
                optimizer.zero_grad()
                for p, gt, gp in zip(model.parameters(), g_task, g_phys):
                    dot = (gt * gp).sum()
                    if dot < 0:
                        # Project out the conflicting component
                        gt = gt - (dot / (gp.norm()**2 + 1e-12)) * gp
                    p.grad = gt + gp

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            else:
                # === Standard backward ===
                mscs_pred = model(gem_feat)
                loss, loss_dict = semi_supervised_loss(
                    mscs_pred, mscs_tgt, w_min, cpi_gold, has_label,
                    ablation=ablation,
                    lambda_task=lt, lambda_const=lc, lambda_mono=lm,
                    unlab_scale=unlab_scale,
                    base_weight=base_weight,
                    adaptive_task=adaptive_task,
                    w_max=w_max,
                    adapt_w_min=adapt_w_min,
                    renorm_total=renorm_total,
                    no_mono_labeled=no_mono_labeled,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            global_step += 1
            step_lr(optimizer, global_step, milestones)

            # Update ReLoBRaLo
            if balancer is not None:
                balancer.update([
                    loss_dict.get("L_task", 0.0),
                    loss_dict.get("L_const", loss_dict.get("L_const_self", 0.0)),
                    loss_dict.get("L_mono_lab", loss_dict.get("L_mono_unl", 0.0)),
                ])

            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    epoch_losses[k] += v
            n_batches += 1

        # SWA: accumulate averaged weights after swa_start
        if use_swa and epoch >= swa_start:
            if swa_count == 0:
                # Initialize SWA state as copy of current model
                swa_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                # Running average: swa = (swa * count + current) / (count + 1)
                for k, v in model.state_dict().items():
                    swa_state[k] = (swa_state[k] * swa_count + v.cpu()) / (swa_count + 1)
                swa_count += 1

        # Track best (non-SWA model)
        test_mape, _, _ = evaluate_gem(model, test_loader, device)
        if test_mape < best_mape:
            best_mape = test_mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            avg = {k: v / n_batches for k, v in epoch_losses.items()
                   if k not in ("n_labeled", "n_unlabeled")}
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            w_str = ""
            if balancer is not None:
                w = balancer.get_weights()
                w_str = f" λ=[{w[0]:.3f},{w[1]:.3f},{w[2]:.3f}]"
            swa_str = f" swa={swa_count}" if use_swa and swa_count > 0 else ""
            adapt_str = ""
            if adaptive_task and "adapt_w_mean" in avg:
                adapt_str = f" aw={avg['adapt_w_mean']:.2f}/{avg.get('adapt_w_min',0):.2f}/{avg['adapt_w_max']:.2f} rerr={avg['base_rel_err_mean']:.4f}"
            print(f"  Epoch {epoch:3d} | "
                  f"L_task={avg.get('L_task', 0):.6f} "
                  f"L_const={avg.get('L_const', 0):.6f} "
                  f"L_cself={avg.get('L_const_self', 0):.6f} "
                  f"L_mono={avg.get('L_mono_unl', 0):.6f} | "
                  f"Test MAPE={test_mape:.2f}% best={best_mape:.2f}% | "
                  f"lr={lr:.2e} step={global_step}{w_str}{swa_str}{adapt_str} | {elapsed:.0f}s")

    # Final: compare best snapshot vs SWA model
    if use_swa and swa_count > 0:
        model.load_state_dict({k: v.to(device) for k, v in swa_state.items()})
        swa_mape, _, _ = evaluate_gem(model, test_loader, device)
        print(f"  SWA MAPE: {swa_mape:.2f}% (averaged {swa_count} checkpoints)")
        if swa_mape < best_mape:
            print(f"  SWA wins! {swa_mape:.2f}% < {best_mape:.2f}%")
            best_mape = swa_mape
            best_state = {k: v.cpu().clone() for k, v in swa_state.items()}
        else:
            print(f"  Best snapshot wins: {best_mape:.2f}% < SWA {swa_mape:.2f}%")
            model.load_state_dict(best_state)
    else:
        model.load_state_dict(best_state)

    total_time = time.time() - t0
    print(f"  Final best MAPE: {best_mape:.2f}% (total time: {total_time:.0f}s)")
    return model, best_mape, total_time


def train_concorde(model, train_loader, test_loader, device,
                   labeled_ratio, num_epochs=NUM_EPOCHS):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    milestones = FIXED_MILESTONES_18K

    def relative_loss(pred, target, eps=1e-8):
        return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()

    best_mape = float("inf")
    best_state = None
    global_step = 0

    print(f"\n{'='*60}")
    print(f"Training Concorde — labeled={labeled_ratio*100:.0f}%")
    print(f"  {num_epochs} epochs, no early stopping, steps/epoch={steps_per_epoch}")
    print(f"  Fixed milestones={milestones}")
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
            step_lr(optimizer, global_step, milestones)

            epoch_loss += loss.item()
            n_batches += 1

        # Track best
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
    parser = argparse.ArgumentParser(description="Exp4: Label Efficiency (recipe v77)")
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ablation", type=str, default="full",
                        choices=ABLATION_MODES)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=DATASET_CHOICES)
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--relobralo", action="store_true",
                        help="Enable ReLoBRaLo adaptive loss weighting")
    parser.add_argument("--pcgrad", action="store_true",
                        help="Enable PCGrad (project conflicting gradients)")
    parser.add_argument("--swa", action="store_true",
                        help="Enable SWA (average weights from last 15% epochs)")
    parser.add_argument("--lambda-task", type=float, default=None,
                        help="Override λ_task (default: 0.1)")
    parser.add_argument("--lambda-unlab", type=float, default=LAMBDA_UNLAB,
                        help="Weight for unlabeled physics loss (default: 1.0)")
    parser.add_argument("--ramp-epochs", type=int, default=RAMP_EPOCHS,
                        help="Linear ramp-up epochs for unlabeled loss (default: 0=disabled)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed (default: from config)")
    parser.add_argument("--base-weight", type=float, default=1.0,
                        help="Extra weight for Base dims in L_task (default: 1.0 = equal)")
    parser.add_argument("--adaptive-task", action="store_true",
                        help="Adaptive per-sample L_task weighting based on Base error")
    parser.add_argument("--no-mono-labeled", action="store_true",
                        help="Remove L_mono from labeled loss (keep on unlabeled)")
    parser.add_argument("--w-max", type=float, default=5.0,
                        help="Max adaptive weight for Base dims (default: 5.0)")
    parser.add_argument("--w-min", type=float, default=0.3,
                        help="Min adaptive weight for Base dims (default: 0.3)")
    parser.add_argument("--renorm-total", action="store_true",
                        help="Renormalize L_task total to match unweighted MSE magnitude")
    args = parser.parse_args()

    ratios = [args.ratio] if args.ratio is not None else ALL_RATIOS
    ablation = args.ablation
    is_concorde = (ablation == "concorde")
    is_labeled_only = ablation in ("supervised_only", "concorde")

    # Override lambda_task if specified
    global LAMBDA_TASK
    if args.lambda_task is not None:
        LAMBDA_TASK = args.lambda_task
        print(f"  λ_task overridden to {LAMBDA_TASK}")

    # Override seed if specified
    global SEED
    actual_seed = args.seed if args.seed is not None else SEED
    if args.seed is not None:
        print(f"  Seed overridden to {actual_seed}")

    out_dir = get_output_dir(args.dataset, ablation, args.results_dir)

    gem_data_dir     = DATASET_PATHS[args.dataset]["gem"]
    concord_data_dir = DATASET_PATHS[args.dataset]["concord"]

    set_seed(actual_seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"  GEM data:     {gem_data_dir}")
    print(f"  Concord data: {concord_data_dir}")
    print(f"Ablation: {ablation}")
    print(f"Seed: {actual_seed}")
    print(f"Recipe: v77 (λ_task={LAMBDA_TASK}, L_const=relative, "
          f"milestones={MILESTONE_FRACS})")
    print(f"Ratios: {[f'{r*100:.0f}%' for r in ratios]}")
    print(f"Output: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1/2] Loading benchmarks...")
    all_data = load_all_benchmarks(gem_data_dir, concord_data_dir, BENCHMARK_NAMES)
    assert len(all_data) > 0

    for ratio in ratios:
        ratio_key = f"{int(round(ratio*100))}pct"

        print(f"\n{'#'*60}")
        print(f"# Dataset={args.dataset} Labeled={ratio*100:.0f}% "
              f"Ablation={ablation} GPU={args.gpu}")
        print(f"{'#'*60}")

        set_seed(actual_seed)

        (train_loader, test_loader, gem_normalizer,
         benchmark_test_indices,
         n_labeled, n_unlabeled) = create_exp4_dataloaders(
            all_data, ratio, labeled_only=is_labeled_only
        )

        print("\n[2/2] Training...")

        gem_hdims = DATASET_MODEL_DIMS[args.dataset]["gem_hidden"]
        con_hdims = DATASET_MODEL_DIMS[args.dataset]["concorde_hidden"]

        if is_concorde:
            model = ConcordeModel(
                input_dim=CONCORDE_INPUT_DIM,
                hidden_dims=con_hdims,
                dropout=0.0,
            ).to(device)
            print(f"  Concorde arch: {CONCORDE_INPUT_DIM} -> {con_hdims} -> 1")
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
            print(f"  GEM arch: {GEM_INPUT_DIM} -> {gem_hdims} -> 16")
            print(f"  GEM params: {sum(p.numel() for p in model.parameters()):,}")

            model, best_mape, train_time = train_gem_semi(
                model, train_loader, test_loader, device,
                benchmark_test_indices, ratio,
                ablation=ablation,
                use_relobralo=args.relobralo,
                use_pcgrad=args.pcgrad,
                use_swa=args.swa,
                lambda_unlab=args.lambda_unlab,
                ramp_epochs=args.ramp_epochs,
                base_weight=args.base_weight,
                adaptive_task=args.adaptive_task,
                w_max=args.w_max,
                adapt_w_min=args.w_min,
                renorm_total=args.renorm_total,
                no_mono_labeled=args.no_mono_labeled,
            )
            test_mape, cpi_pred, cpi_gold = evaluate_gem(
                model, test_loader, device
            )

        per_bm = per_benchmark_mape(cpi_pred, cpi_gold, benchmark_test_indices)

        print(f"\nResults for dataset={args.dataset} labeled={ratio*100:.0f}% "
              f"ablation={ablation}:")
        print(f"  Overall MAPE: {test_mape:.2f}%")
        for name in sorted(per_bm.keys()):
            print(f"  {name:<20} {per_bm[name]:>9.2f}%")

        result = {
            "dataset": args.dataset,
            "labeled_ratio": ratio,
            "ablation": ablation,
            "recipe": "v77",
            "lambda_task": LAMBDA_TASK,
            "lambda_const": LAMBDA_CONST,
            "lambda_mono": LAMBDA_MONO,
            "n_labeled": n_labeled,
            "n_unlabeled": n_unlabeled,
            "overall_mape": test_mape,
            "per_benchmark": per_bm,
            "train_time_seconds": train_time,
        }

        # Include lambda_task in filename if overridden
        if args.lambda_task is not None:
            result_path = os.path.join(out_dir, f"exp4_{ratio_key}_lt{LAMBDA_TASK:.2f}.json")
        elif args.seed is not None:
            result_path = os.path.join(out_dir, f"exp4_{ratio_key}_s{actual_seed}.json")
        else:
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