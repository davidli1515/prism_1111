"""
Experiment 4 Extension: Extra Unlabeled Data from 40k dataset

Uses 18k's train/test split, adds 40k's extra samples (not in 18k) as
additional unlabeled data. Tests on 18k's original test set for direct
comparison with standard 18k results.

This proves: more unlabeled data → better label efficiency.

Usage:
    python train_exp4_extra_unlabeled.py --ratio 0.20 --gpu 0
    python train_exp4_extra_unlabeled.py --ratio 0.05 --gpu 1

Output: results_{dir}/exp4_18k_extra_unlabeled/exp4_{ratio}pct.json
"""

import os, sys, time, json, hashlib, argparse
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")

from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM, MSCS_DIM,
    GEM_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY,
    CONCORDE_BATCH_SIZE,
    BENCHMARK_NAMES,
    SEED, TRAIN_RATIO,
    LAMBDA_CONST, LAMBDA_MONO,
    BASE_INDICES,
)
from model import (
    GEMModel, compute_cpi_mean,
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

ALL_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); torch.backends.cudnn.deterministic = True


def stable_hash(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)


def step_lr(optimizer, global_step):
    n_passed = sum(1 for m in STEP_MILESTONES if global_step >= m)
    new_lr = LEARNING_RATE * (LR_GAMMA ** n_passed)
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr


def loss_const_self(mscs_pred, w_min, eps=1e-8):
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi_per_stage = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    cpi_mean = cpi_per_stage.mean(dim=1, keepdim=True)
    diff = cpi_per_stage - cpi_mean
    return (diff ** 2).mean()


def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


# ============================================================
# Dataset with extra unlabeled
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


def find_extra_indices(data_18k_bench, data_40k_bench):
    """
    Find rows in 40k that are NOT in 18k using byte-level hash comparison.
    Returns indices into 40k arrays for extra samples.
    """
    gem_18k = data_18k_bench["gem_features"]
    gem_40k = data_40k_bench["gem_features"]

    set_18k = set()
    for i in range(len(gem_18k)):
        set_18k.add(gem_18k[i].tobytes())

    extra_idx = []
    for i in range(len(gem_40k)):
        if gem_40k[i].tobytes() not in set_18k:
            extra_idx.append(i)

    return np.array(extra_idx, dtype=np.int64)


def create_dataloaders_with_extra(data_18k, data_40k, labeled_ratio,
                                  batch_size=CONCORDE_BATCH_SIZE, seed=SEED):
    """
    Create dataloaders:
    - Train/test split from 18k (unchanged)
    - Extra unlabeled from 40k (rows not in 18k)
    - Normalizer fit on ALL train data (18k train + extra)
    """
    rng_split = np.random.RandomState(seed)

    train_gem, train_mscs, train_cpi, train_wmin = [], [], [], []
    train_has_label = []
    test_gem, test_mscs, test_cpi, test_wmin = [], [], [], []
    per_benchmark_test = {}

    test_offset = 0
    total_labeled = 0
    total_unlabeled_18k = 0
    total_extra = 0

    for name in sorted(data_18k.keys()):
        d18 = data_18k[name]
        d40 = data_40k[name]

        n = d18["n_samples"]
        indices = rng_split.permutation(n)
        split = int(n * TRAIN_RATIO)
        tr_idx = indices[:split]
        te_idx = indices[split:]

        # --- 18k train: split into labeled / unlabeled ---
        n_train = len(tr_idx)
        n_labeled = max(1, int(n_train * labeled_ratio))

        label_rng = np.random.RandomState(
            seed + stable_hash(name) + int(round(labeled_ratio * 1000))
        )
        label_perm = label_rng.permutation(n_train)
        has_label = np.zeros(n_train, dtype=np.float32)
        has_label[label_perm[:n_labeled]] = 1.0

        total_labeled += n_labeled
        total_unlabeled_18k += (n_train - n_labeled)

        train_gem.append(d18["gem_features"][tr_idx])
        train_mscs.append(d18["mscs_target"][tr_idx])
        train_cpi.append(d18["cpi_golden"][tr_idx])
        train_wmin.append(d18["w_min"][tr_idx])
        train_has_label.append(has_label)

        # --- 18k test (unchanged) ---
        test_gem.append(d18["gem_features"][te_idx])
        test_mscs.append(d18["mscs_target"][te_idx])
        test_cpi.append(d18["cpi_golden"][te_idx])
        test_wmin.append(d18["w_min"][te_idx])

        n_test = len(te_idx)
        per_benchmark_test[name] = (test_offset, test_offset + n_test)
        test_offset += n_test

        # --- Extra unlabeled from 40k ---
        extra_idx = find_extra_indices(d18, d40)
        if len(extra_idx) > 0:
            train_gem.append(d40["gem_features"][extra_idx])
            train_mscs.append(d40["mscs_target"][extra_idx])
            train_cpi.append(d40["cpi_golden"][extra_idx])
            train_wmin.append(d40["w_min"][extra_idx])
            train_has_label.append(np.zeros(len(extra_idx), dtype=np.float32))
            total_extra += len(extra_idx)

    train_gem       = np.concatenate(train_gem)
    train_mscs      = np.concatenate(train_mscs)
    train_cpi       = np.concatenate(train_cpi)
    train_wmin      = np.concatenate(train_wmin)
    train_has_label = np.concatenate(train_has_label)

    test_gem  = np.concatenate(test_gem)
    test_mscs = np.concatenate(test_mscs)
    test_cpi  = np.concatenate(test_cpi)
    test_wmin = np.concatenate(test_wmin)

    # Normalize on ALL train data
    gem_norm = FeatureNormalizer()
    train_gem = gem_norm.fit_transform(train_gem)
    test_gem  = gem_norm.transform(test_gem)

    train_ds = SemiSupervisedDataset(train_gem, train_mscs, train_cpi, train_wmin, train_has_label)
    test_ds  = SemiSupervisedDataset(test_gem, test_mscs, test_cpi, test_wmin,
                                     np.ones(len(test_gem), dtype=np.float32))

    actual_bs = min(batch_size, len(train_ds))
    drop_last = len(train_ds) > actual_bs

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=actual_bs, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=drop_last,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"  Labeled ratio: {labeled_ratio*100:.0f}%")
    print(f"  18k labeled: {total_labeled:,}")
    print(f"  18k unlabeled: {total_unlabeled_18k:,}")
    print(f"  40k extra unlabeled: {total_extra:,}")
    print(f"  Total train: {len(train_ds):,} | Batch: {actual_bs:,}")
    print(f"  Test (18k only): {len(test_ds):,}")

    return (train_loader, test_loader, per_benchmark_test,
            total_labeled, total_unlabeled_18k, total_extra)


# ============================================================
# Semi-supervised loss (same as train_exp4_all.py)
# ============================================================

def semi_supervised_loss(mscs_pred, mscs_target, w_min, cpi_golden,
                         has_label, lambda_const=LAMBDA_CONST,
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
        loss_dict["L_task"]  = l_task.item()
        loss_dict["L_const"] = l_const.item()
        total = total + labeled_loss

    if n_unlabeled > 0:
        u_pred = mscs_pred[unlabeled_mask]
        u_wmin = w_min[unlabeled_mask]

        u_const_self = loss_const_self(u_pred, u_wmin)
        u_mono_val   = loss_mono(u_pred)
        unlabeled_loss = lambda_const * u_const_self + lambda_mono * u_mono_val

        loss_dict["L_const_self"] = u_const_self.item()
        loss_dict["L_mono_unl"]   = u_mono_val.item()
        total = total + unlabeled_loss

    loss_dict["L_total"] = total.item()
    return total, loss_dict


# ============================================================
# Eval
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_cpi_pred, all_cpi_gold = [], []
    for batch in loader:
        feat = batch["gem_features"].to(device)
        w_min = batch["w_min"].to(device)
        cpi_gold = batch["cpi_golden"].to(device)
        mscs_pred = model(feat)
        cpi_pred = compute_cpi_mean(mscs_pred, w_min)
        all_cpi_pred.append(cpi_pred.cpu())
        all_cpi_gold.append(cpi_gold.cpu())
    all_cpi_pred = torch.cat(all_cpi_pred)
    all_cpi_gold = torch.cat(all_cpi_gold)
    return compute_mape(all_cpi_pred, all_cpi_gold), all_cpi_pred, all_cpi_gold


def per_benchmark_mape(cpi_pred, cpi_gold, benchmark_indices):
    results = {}
    for name, (start, end) in benchmark_indices.items():
        if start == end: continue
        results[name] = compute_mape(cpi_pred[start:end], cpi_gold[start:end])
    return results


# ============================================================
# Training
# ============================================================

def train(model, train_loader, test_loader, device, labeled_ratio, num_epochs=1500):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)

    best_mape = float("inf")
    best_state = None
    global_step = 0
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Training — labeled={labeled_ratio*100:.0f}% + extra unlabeled from 40k")
    print(f"  Fixed {num_epochs} epochs (no early stopping)")
    print(f"  Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"  Scheduler: step-based milestones={STEP_MILESTONES}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch in train_loader:
            feat      = batch["gem_features"].to(device)
            mscs_tgt  = batch["mscs_target"].to(device)
            cpi_gold  = batch["cpi_golden"].to(device)
            w_min     = batch["w_min"].to(device)
            has_label = batch["has_label"].to(device)

            mscs_pred = model(feat)
            loss, loss_dict = semi_supervised_loss(
                mscs_pred, mscs_tgt, w_min, cpi_gold, has_label,
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

        # Track best
        test_mape, _, _ = evaluate(model, test_loader, device)
        if test_mape < best_mape:
            best_mape = test_mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            avg = {k: v / n_batches for k, v in epoch_losses.items()}
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d} | "
                  f"L_task={avg.get('L_task', 0):.6f} "
                  f"L_const={avg.get('L_const', 0):.6f} "
                  f"L_cself={avg.get('L_const_self', 0):.6f} | "
                  f"Test MAPE={test_mape:.2f}% best={best_mape:.2f}% | "
                  f"lr={lr:.2e} step={global_step} | {elapsed:.0f}s")

    model.load_state_dict(best_state)
    total_time = time.time() - t0
    print(f"  Best test MAPE: {best_mape:.2f}% (total time: {total_time:.0f}s)")
    return model, best_mape, total_time


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp4 Extension: 18k + extra unlabeled from 40k")
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="./results")
    args = parser.parse_args()

    ratios = [args.ratio] if args.ratio is not None else ALL_RATIOS
    out_dir = f"{args.results_dir}/exp4_18k_extra_unlabeled"

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1/3] Loading 18k benchmarks...")
    data_18k = load_all_benchmarks(
        DATASET_PATHS["18k"]["gem"],
        DATASET_PATHS["18k"]["concord"],
        BENCHMARK_NAMES,
    )

    print("[2/3] Loading 40k benchmarks...")
    data_40k = load_all_benchmarks(
        DATASET_PATHS["40k"]["gem"],
        DATASET_PATHS["40k"]["concord"],
        BENCHMARK_NAMES,
    )

    for ratio in ratios:
        ratio_key = f"{int(round(ratio*100))}pct"

        print(f"\n{'#'*60}")
        print(f"# 18k + extra unlabeled | Labeled={ratio*100:.0f}% | GPU={args.gpu}")
        print(f"{'#'*60}")

        set_seed(SEED)

        (train_loader, test_loader, benchmark_test_indices,
         n_labeled, n_unlabeled_18k, n_extra) = create_dataloaders_with_extra(
            data_18k, data_40k, ratio,
        )

        print("\n[3/3] Training...")
        model = GEMModel(
            input_dim=GEM_INPUT_DIM,
            hidden_dims=[512, 256],
            dropout=DROPOUT,
        ).to(device)
        print(f"  GEM params: {sum(p.numel() for p in model.parameters()):,}")

        model, best_mape, train_time = train(
            model, train_loader, test_loader, device, ratio,
        )

        test_mape, cpi_pred, cpi_gold = evaluate(model, test_loader, device)
        per_bm = per_benchmark_mape(cpi_pred, cpi_gold, benchmark_test_indices)

        print(f"\nResults:")
        print(f"  Overall MAPE: {test_mape:.2f}%")
        for name in sorted(per_bm.keys()):
            print(f"  {name:<20} {per_bm[name]:>9.2f}%")

        result = {
            "dataset": "18k_extra_unlabeled",
            "labeled_ratio": ratio,
            "n_labeled": n_labeled,
            "n_unlabeled_18k": n_unlabeled_18k,
            "n_extra_40k": n_extra,
            "overall_mape": test_mape,
            "per_benchmark": per_bm,
            "train_time_seconds": train_time,
        }

        result_path = os.path.join(out_dir, f"exp4_{ratio_key}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {result_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
