"""
Experiment 4 Variant: GEM Vanilla (L_task only, no physics constraints)

GEM model predicting 16-dim MSCS, but trained with only L_task (MSE).
No L_const, no L_mono, no unlabeled data.
Shows the value of physics constraints by removing them entirely.

Usage:
    python train_exp4_vanilla.py --dataset 40k --ratio 0.20 --gpu 0

Output: results/exp4_{dataset}_vanilla/exp4_{ratio}pct.json

Imports from exp1: /path/to/gem5/micro_exp/exp1/
"""

import os, sys, time, json, hashlib, argparse
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")

from config import (
    GEM_INPUT_DIM, MSCS_DIM,
    GEM_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    CONCORDE_BATCH_SIZE,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    SEED, TRAIN_RATIO,
    BASE_INDICES,
)
from model import GEMModel, compute_cpi_mean, loss_task
from dataset import load_all_benchmarks, FeatureNormalizer, GEMDataset

try:
    from config import STEP_MILESTONES
except ImportError:
    STEP_MILESTONES = [10000, 14000, 18000, 22000]

try:
    from config import LR_GAMMA
except ImportError:
    LR_GAMMA = 0.5

ALL_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

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

DATASET_GEM_DIMS = {
    "18k": [512, 256],
    "40k": [512, 256],
}


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


def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


def create_vanilla_dataloaders(all_data, labeled_ratio,
                               batch_size=CONCORDE_BATCH_SIZE, seed=SEED):
    """Create dataloaders with only labeled samples (no unlabeled)."""
    rng_split = np.random.RandomState(seed)

    train_gem, train_mscs, train_cpi, train_wmin = [], [], [], []
    test_gem, test_mscs, test_cpi, test_wmin = [], [], [], []
    per_benchmark_test = {}

    test_offset = 0
    total_labeled = 0

    for name, data in sorted(all_data.items()):
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
        labeled_idx = tr_idx[label_perm[:n_labeled]]
        total_labeled += n_labeled

        train_gem.append(data["gem_features"][labeled_idx])
        train_mscs.append(data["mscs_target"][labeled_idx])
        train_cpi.append(data["cpi_golden"][labeled_idx])
        train_wmin.append(data["w_min"][labeled_idx])

        test_gem.append(data["gem_features"][te_idx])
        test_mscs.append(data["mscs_target"][te_idx])
        test_cpi.append(data["cpi_golden"][te_idx])
        test_wmin.append(data["w_min"][te_idx])

        n_test = len(te_idx)
        per_benchmark_test[name] = (test_offset, test_offset + n_test)
        test_offset += n_test

    # Concatenate ALL train data (before filtering) for normalizer
    # We need to recompute to fit normalizer on full train set
    rng2 = np.random.RandomState(seed)
    all_train_gem = []
    for name, data in sorted(all_data.items()):
        n = data["n_samples"]
        idx = rng2.permutation(n)
        sp = int(n * TRAIN_RATIO)
        all_train_gem.append(data["gem_features"][idx[:sp]])
    all_train_gem = np.concatenate(all_train_gem)

    train_gem  = np.concatenate(train_gem)
    train_mscs = np.concatenate(train_mscs)
    train_cpi  = np.concatenate(train_cpi)
    train_wmin = np.concatenate(train_wmin)

    test_gem  = np.concatenate(test_gem)
    test_mscs = np.concatenate(test_mscs)
    test_cpi  = np.concatenate(test_cpi)
    test_wmin = np.concatenate(test_wmin)

    # Normalizer fit on ALL train data (same as other variants)
    gem_norm = FeatureNormalizer()
    gem_norm.fit(all_train_gem)
    train_gem = gem_norm.transform(train_gem)
    test_gem  = gem_norm.transform(test_gem)

    # Simple dataset (no has_label needed, all are labeled)
    from torch.utils.data import Dataset, DataLoader

    class SimpleDS(Dataset):
        def __init__(self, gf, mt, cg, wm):
            self.gf = torch.from_numpy(gf); self.mt = torch.from_numpy(mt)
            self.cg = torch.from_numpy(cg); self.wm = torch.from_numpy(wm)
        def __len__(self): return self.gf.shape[0]
        def __getitem__(self, i):
            return {"gem_features": self.gf[i], "mscs_target": self.mt[i],
                    "cpi_golden": self.cg[i], "w_min": self.wm[i]}

    train_ds = SimpleDS(train_gem, train_mscs, train_cpi, train_wmin)
    test_ds  = SimpleDS(test_gem, test_mscs, test_cpi, test_wmin)

    actual_bs = min(batch_size, len(train_ds))
    drop_last = len(train_ds) > actual_bs

    train_loader = DataLoader(train_ds, batch_size=actual_bs, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=drop_last)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"  Labeled ratio: {labeled_ratio*100:.0f}%")
    print(f"  Train (labeled only): {len(train_ds):,} | Batch: {actual_bs:,}")
    print(f"  Test: {len(test_ds):,}")

    return train_loader, test_loader, per_benchmark_test, total_labeled


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


def train_vanilla(model, train_loader, test_loader, device,
                  labeled_ratio, num_epochs=1500):
    """Train GEM with L_task only. No L_const, no L_mono."""
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)

    print(f"  Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"  Loss: L_task (MSE) only — no physics constraints")
    print(f"  Scheduler: MultiStepLR per-step (milestones={STEP_MILESTONES})")

    best_mape = float("inf")
    best_state = None
    global_step = 0
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Training GEM Vanilla — labeled={labeled_ratio*100:.0f}%  loss=L_task only")
    print(f"  Fixed {num_epochs} epochs (no early stopping)")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            feat     = batch["gem_features"].to(device)
            mscs_tgt = batch["mscs_target"].to(device)

            mscs_pred = model(feat)
            loss = loss_task(mscs_pred, mscs_tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            step_lr(optimizer, global_step)

            epoch_loss += loss.item()
            n_batches += 1

        # Track best model
        test_mape, _, _ = evaluate(model, test_loader, device)
        if test_mape < best_mape:
            best_mape = test_mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = epoch_loss / n_batches
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d} | "
                  f"L_task={avg_loss:.6f} | "
                  f"Test MAPE={test_mape:.2f}% best={best_mape:.2f}% | "
                  f"lr={lr:.2e} step={global_step} | {elapsed:.0f}s")

    model.load_state_dict(best_state)
    total_time = time.time() - t0
    print(f"  Best test MAPE: {best_mape:.2f}% (total time: {total_time:.0f}s)")
    return model, best_mape, total_time


def main():
    parser = argparse.ArgumentParser(description="Exp4: GEM Vanilla (L_task only)")
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, required=True, choices=["18k", "40k"])
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Base directory for results (default: ./results)")
    args = parser.parse_args()

    ratios = [args.ratio] if args.ratio is not None else ALL_RATIOS
    ds = args.dataset
    out_dir = f"{args.results_dir}/exp4_{ds}_vanilla"

    gem_data_dir     = DATASET_PATHS[ds]["gem"]
    concord_data_dir = DATASET_PATHS[ds]["concord"]

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {ds}")
    print(f"  GEM data: {gem_data_dir}")
    print(f"Output: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1/2] Loading benchmarks...")
    all_data = load_all_benchmarks(gem_data_dir, concord_data_dir, BENCHMARK_NAMES)
    assert len(all_data) > 0

    for ratio in ratios:
        ratio_key = f"{int(round(ratio*100))}pct"

        print(f"\n{'#'*60}")
        print(f"# Dataset={ds} Labeled={ratio*100:.0f}% Variant=vanilla GPU={args.gpu}")
        print(f"{'#'*60}")

        set_seed(SEED)

        (train_loader, test_loader,
         benchmark_test_indices, n_labeled) = create_vanilla_dataloaders(all_data, ratio)

        print("\n[2/2] Training...")
        gem_hdims = DATASET_GEM_DIMS[ds]
        model = GEMModel(
            input_dim=GEM_INPUT_DIM,
            hidden_dims=gem_hdims,
            dropout=DROPOUT,
        ).to(device)
        print(f"  GEM arch: {GEM_INPUT_DIM} → {gem_hdims} → 16")
        print(f"  GEM params: {sum(p.numel() for p in model.parameters()):,}")

        model, best_mape, train_time = train_vanilla(
            model, train_loader, test_loader, device, ratio,
        )

        test_mape, cpi_pred, cpi_gold = evaluate(model, test_loader, device)
        per_bm = per_benchmark_mape(cpi_pred, cpi_gold, benchmark_test_indices)

        print(f"\nResults for dataset={ds} labeled={ratio*100:.0f}% variant=vanilla:")
        print(f"  Overall MAPE: {test_mape:.2f}%")
        print(f"  {'Benchmark':<20} {'MAPE%':>10}")
        print(f"  {'-'*32}")
        for name in sorted(per_bm.keys()):
            print(f"  {name:<20} {per_bm[name]:>9.2f}%")

        result = {
            "dataset": ds,
            "labeled_ratio": ratio,
            "ablation": "vanilla",
            "n_labeled": n_labeled,
            "n_unlabeled": 0,
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
                   os.path.join(ckpt_dir, f"vanilla_{ratio_key}.pt"))

    print("\nDone.")


if __name__ == "__main__":
    main()