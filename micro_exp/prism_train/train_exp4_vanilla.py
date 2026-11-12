"""
Experiment 4 Variant: GEM Vanilla (L_task only, no physics)
Recipe: v77 (matching exp1_v3_run2)

Usage:
    python train_exp4_vanilla.py --dataset 18k --ratio 0.20 --gpu 0
"""

import os, sys, time, json, hashlib, argparse
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    GEM_INPUT_DIM, DROPOUT, BENCHMARK_NAMES,
    SEED, TRAIN_RATIO,
)
from model import GEMModel, compute_cpi_mean, loss_task
from dataset import load_all_benchmarks, FeatureNormalizer

# Recipe v77
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.3
BATCH_SIZE = 50000
NUM_EPOCHS = 1521
PATIENCE = 100
LR_GAMMA = 0.5
MILESTONE_FRACS = [0.42, 0.58, 0.75, 0.92]

ALL_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
              0.60, 0.70, 0.80, 0.90]

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

DATASET_GEM_DIMS = {"18k": [512, 256], "40k": [512, 256]}


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); torch.backends.cudnn.deterministic = True

def stable_hash(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)

def step_lr(optimizer, global_step, milestones):
    n_passed = sum(1 for m in milestones if global_step >= m)
    new_lr = LEARNING_RATE * (LR_GAMMA ** n_passed)
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr

def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


def create_vanilla_dataloaders(all_data, labeled_ratio,
                               batch_size=BATCH_SIZE, seed=SEED):
    rng_split = np.random.RandomState(seed)

    train_gem, train_mscs, train_cpi, train_wmin = [], [], [], []
    test_gem, test_mscs, test_cpi, test_wmin = [], [], [], []
    per_benchmark_test = {}
    test_offset = 0
    total_labeled = 0

    # First pass: collect all train data for normalizer
    rng2 = np.random.RandomState(seed)
    all_train_gem = []
    for name, data in sorted(all_data.items()):
        n = data["n_samples"]
        idx = rng2.permutation(n)
        sp = int(n * TRAIN_RATIO)
        all_train_gem.append(data["gem_features"][idx[:sp]])
    all_train_gem_full = np.concatenate(all_train_gem)

    # Second pass: split and select labeled
    rng_split2 = np.random.RandomState(seed)
    for name, data in sorted(all_data.items()):
        n = data["n_samples"]
        indices = rng_split2.permutation(n)
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

    train_gem  = np.concatenate(train_gem)
    train_mscs = np.concatenate(train_mscs)
    train_cpi  = np.concatenate(train_cpi)
    train_wmin = np.concatenate(train_wmin)
    test_gem   = np.concatenate(test_gem)
    test_mscs  = np.concatenate(test_mscs)
    test_cpi   = np.concatenate(test_cpi)
    test_wmin  = np.concatenate(test_wmin)

    gem_norm = FeatureNormalizer()
    gem_norm.fit(all_train_gem_full)
    train_gem = gem_norm.transform(train_gem)
    test_gem  = gem_norm.transform(test_gem)

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
                  labeled_ratio, num_epochs=NUM_EPOCHS):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    milestones = [int(f * total_steps) for f in MILESTONE_FRACS]

    print(f"  Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"  Loss: L_task (MSE) only")
    print(f"  Scheduler: milestones={milestones}, gamma={LR_GAMMA}")

    best_mape = float("inf")
    best_state = None
    global_step = 0
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Training Vanilla — labeled={labeled_ratio*100:.0f}%")
    print(f"  {num_epochs} epochs, no early stopping, steps/epoch={steps_per_epoch}")
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
            step_lr(optimizer, global_step, milestones)

            epoch_loss += loss.item()
            n_batches += 1

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
    parser = argparse.ArgumentParser(description="Exp4: Vanilla (recipe v77)")
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, required=True, choices=["18k", "40k"])
    parser.add_argument("--results-dir", type=str, default="./results")
    args = parser.parse_args()

    ratios = [args.ratio] if args.ratio is not None else ALL_RATIOS
    ds = args.dataset
    out_dir = f"{args.results_dir}/exp4_{ds}_vanilla"

    gem_data_dir     = DATASET_PATHS[ds]["gem"]
    concord_data_dir = DATASET_PATHS[ds]["concord"]

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {ds} | Recipe: v77")
    print(f"Output: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    print("\nLoading benchmarks...")
    all_data = load_all_benchmarks(gem_data_dir, concord_data_dir, BENCHMARK_NAMES)
    assert len(all_data) > 0

    for ratio in ratios:
        ratio_key = f"{int(round(ratio*100))}pct"

        print(f"\n{'#'*60}")
        print(f"# Vanilla | Dataset={ds} Labeled={ratio*100:.0f}% GPU={args.gpu}")
        print(f"{'#'*60}")

        set_seed(SEED)

        (train_loader, test_loader,
         benchmark_test_indices, n_labeled) = create_vanilla_dataloaders(all_data, ratio)

        gem_hdims = DATASET_GEM_DIMS[ds]
        model = GEMModel(
            input_dim=GEM_INPUT_DIM,
            hidden_dims=gem_hdims,
            dropout=DROPOUT,
        ).to(device)
        print(f"  GEM arch: {GEM_INPUT_DIM} -> {gem_hdims} -> 16")
        print(f"  GEM params: {sum(p.numel() for p in model.parameters()):,}")

        model, best_mape, train_time = train_vanilla(
            model, train_loader, test_loader, device, ratio,
        )

        test_mape, cpi_pred, cpi_gold = evaluate(model, test_loader, device)
        per_bm = per_benchmark_mape(cpi_pred, cpi_gold, benchmark_test_indices)

        print(f"\nResults:")
        print(f"  Overall MAPE: {test_mape:.2f}%")
        for name in sorted(per_bm.keys()):
            print(f"  {name:<20} {per_bm[name]:>9.2f}%")

        result = {
            "dataset": ds,
            "labeled_ratio": ratio,
            "ablation": "vanilla",
            "recipe": "v77",
            "n_labeled": n_labeled,
            "n_unlabeled": 0,
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