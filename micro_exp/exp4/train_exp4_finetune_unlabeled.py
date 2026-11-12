"""
Pretrain+Finetune Type 1, Stage 2:
  Load supervised_only checkpoint → finetune with unlabeled data (L_const_self only)

Stage 1 = supervised_only training (already done, checkpoints in results_run2)
Stage 2 = this script: physics-only finetune on unlabeled data

Usage:
    python train_exp4_finetune_unlabeled.py --ratio 0.20 --gpu 0
    python train_exp4_finetune_unlabeled.py --ratio 0.20 --gpu 0 --extra  # with 40k extra

Finetune Recipe:
    lr=1e-4 (10x lower), 500 epochs, L_const_self + L_mono (full physics)
    Controlled comparison: same loss as semi-supervised, only difference is sequential vs joint
"""

import os, sys, time, json, hashlib, argparse
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")

from config import (
    GEM_INPUT_DIM, DROPOUT, BENCHMARK_NAMES,
    SEED, TRAIN_RATIO, BASE_INDICES,
)
from model import GEMModel, compute_cpi_mean, loss_const, loss_mono
from dataset import load_all_benchmarks, FeatureNormalizer

# ============================================================
# Finetune Recipe
# ============================================================
FT_LR = 1e-4            # 10x lower than Stage 1
WEIGHT_DECAY = 0.3
BATCH_SIZE = 50000
FT_EPOCHS = 1521
LR_GAMMA = 0.5
MILESTONE_FRACS = [0.42, 0.58, 0.75, 0.92]

ALL_RATIOS = [0.05, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

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

CKPT_DIR = "/path/to/gem5/micro_exp/exp4/results_run2/exp4_18k_supervised_only/checkpoints"


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); torch.backends.cudnn.deterministic = True

def stable_hash(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)

def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()

def loss_const_self(mscs_pred, w_min, eps=1e-8):
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi_per_stage = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    cpi_mean = cpi_per_stage.mean(dim=1, keepdim=True)
    diff = cpi_per_stage - cpi_mean
    return (diff ** 2).mean()


def find_extra_indices(d18, d40):
    set_18k = set()
    for i in range(len(d18["gem_features"])):
        set_18k.add(d18["gem_features"][i].tobytes())
    return np.array([i for i in range(len(d40["gem_features"]))
                     if d40["gem_features"][i].tobytes() not in set_18k], dtype=np.int64)


# ============================================================
# Create unlabeled-only dataloader
# ============================================================

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, gem_features, w_min):
        self.gem_features = torch.from_numpy(gem_features)
        self.w_min = torch.from_numpy(w_min)

    def __len__(self):
        return self.gem_features.shape[0]

    def __getitem__(self, idx):
        return {"gem_features": self.gem_features[idx],
                "w_min": self.w_min[idx]}


def create_finetune_dataloaders(data_18k, labeled_ratio, data_40k=None,
                                 batch_size=BATCH_SIZE, seed=SEED):
    """
    Create dataloaders for finetune:
    - Unlabeled: 18k train set's unlabeled portion (+ optional 40k extra)
    - Test: 18k test set (same as always)
    - Normalizer: fit on ALL train data (for consistency with supervised_only)
    """
    rng_split = np.random.RandomState(seed)

    all_train_gem = []
    unlabeled_gem, unlabeled_wmin = [], []
    test_gem, test_mscs, test_cpi, test_wmin = [], [], [], []
    per_benchmark_test = {}
    test_offset = 0
    n_unlabeled_18k = 0
    n_extra = 0

    for name in sorted(data_18k.keys()):
        d18 = data_18k[name]
        n = d18["n_samples"]
        indices = rng_split.permutation(n)
        split = int(n * TRAIN_RATIO)
        tr_idx = indices[:split]
        te_idx = indices[split:]

        # ALL train data for normalizer
        all_train_gem.append(d18["gem_features"][tr_idx])

        # Split train into labeled/unlabeled (same split as supervised_only)
        n_train = len(tr_idx)
        n_labeled = max(1, int(n_train * labeled_ratio))

        label_rng = np.random.RandomState(
            seed + stable_hash(name) + int(round(labeled_ratio * 1000))
        )
        label_perm = label_rng.permutation(n_train)
        unlabeled_perm = label_perm[n_labeled:]  # unlabeled indices within tr_idx
        unlabeled_idx = tr_idx[unlabeled_perm]

        unlabeled_gem.append(d18["gem_features"][unlabeled_idx])
        unlabeled_wmin.append(d18["w_min"][unlabeled_idx])
        n_unlabeled_18k += len(unlabeled_idx)

        # Extra unlabeled from 40k
        if data_40k is not None and name in data_40k:
            extra_idx = find_extra_indices(d18, data_40k[name])
            if len(extra_idx) > 0:
                unlabeled_gem.append(data_40k[name]["gem_features"][extra_idx])
                unlabeled_wmin.append(data_40k[name]["w_min"][extra_idx])
                all_train_gem.append(data_40k[name]["gem_features"][extra_idx])
                n_extra += len(extra_idx)

        # Test
        test_gem.append(d18["gem_features"][te_idx])
        test_mscs.append(d18["mscs_target"][te_idx])
        test_cpi.append(d18["cpi_golden"][te_idx])
        test_wmin.append(d18["w_min"][te_idx])

        n_test = len(te_idx)
        per_benchmark_test[name] = (test_offset, test_offset + n_test)
        test_offset += n_test

    # Normalize
    gem_norm = FeatureNormalizer()
    gem_norm.fit(np.concatenate(all_train_gem))

    unlabeled_gem = gem_norm.transform(np.concatenate(unlabeled_gem))
    unlabeled_wmin = np.concatenate(unlabeled_wmin)

    test_gem  = gem_norm.transform(np.concatenate(test_gem))
    test_mscs = np.concatenate(test_mscs)
    test_cpi  = np.concatenate(test_cpi)
    test_wmin = np.concatenate(test_wmin)

    # Dataloaders
    unlabeled_ds = UnlabeledDataset(unlabeled_gem, unlabeled_wmin)
    actual_bs = min(batch_size, len(unlabeled_ds))

    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_ds, batch_size=actual_bs, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=len(unlabeled_ds) > actual_bs,
    )

    # Test loader (full features for eval)
    from dataset import GEMDataset
    test_ds = GEMDataset(test_gem, np.zeros_like(test_gem[:, :1]),  # dummy concord
                          test_mscs, test_cpi, test_wmin)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"  Labeled ratio: {labeled_ratio*100:.0f}%")
    print(f"  Unlabeled (18k): {n_unlabeled_18k:,}")
    if n_extra > 0:
        print(f"  Extra (40k): {n_extra:,}")
    print(f"  Total unlabeled: {len(unlabeled_ds):,} | Batch: {actual_bs:,}")
    print(f"  Test: {len(test_ds):,}")

    return unlabeled_loader, test_loader, per_benchmark_test, gem_norm


# ============================================================
# Evaluation
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
    return compute_mape(torch.cat(all_cpi_pred), torch.cat(all_cpi_gold)), \
           torch.cat(all_cpi_pred), torch.cat(all_cpi_gold)

def per_benchmark_mape(cpi_pred, cpi_gold, bm_idx):
    results = {}
    for name, (s, e) in bm_idx.items():
        if s < e:
            results[name] = compute_mape(cpi_pred[s:e], cpi_gold[s:e])
    return results


# ============================================================
# Finetune
# ============================================================

def finetune(model, unlabeled_loader, test_loader, device, labeled_ratio):
    optimizer = optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(unlabeled_loader)
    total_steps = FT_EPOCHS * steps_per_epoch
    milestones = [int(f * total_steps) for f in MILESTONE_FRACS]

    best_mape = float("inf")
    best_state = None
    global_step = 0
    t0 = time.time()

    # Eval before finetune
    pre_mape, _, _ = evaluate(model, test_loader, device)
    print(f"  Pre-finetune MAPE: {pre_mape:.2f}%")

    print(f"\n{'='*60}")
    print(f"Finetune Stage 2 — L_const_self + L_mono (full physics)")
    print(f"  {FT_EPOCHS} epochs, lr={FT_LR}, steps/epoch={steps_per_epoch}")
    print(f"  Milestones={milestones}")
    print(f"{'='*60}")

    LAMBDA_CONST = 1.0
    LAMBDA_MONO = 0.01

    for epoch in range(1, FT_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in unlabeled_loader:
            feat  = batch["gem_features"].to(device)
            w_min = batch["w_min"].to(device)

            mscs_pred = model(feat)
            l_const_s = loss_const_self(mscs_pred, w_min)
            l_mono_val = loss_mono(mscs_pred)
            loss = LAMBDA_CONST * l_const_s + LAMBDA_MONO * l_mono_val

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            # Step LR
            n_passed = sum(1 for m in milestones if global_step >= m)
            new_lr = FT_LR * (LR_GAMMA ** n_passed)
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

            epoch_loss += loss.item()
            n_batches += 1

        test_mape, _, _ = evaluate(model, test_loader, device)
        if test_mape < best_mape:
            best_mape = test_mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d} | L_cself={epoch_loss/n_batches:.6f} | "
                  f"Test MAPE={test_mape:.2f}% best={best_mape:.2f}% | "
                  f"lr={lr:.2e} step={global_step} | {elapsed:.0f}s")

    model.load_state_dict(best_state)
    total_time = time.time() - t0
    print(f"  Pre-finetune: {pre_mape:.2f}% → Post-finetune: {best_mape:.2f}% "
          f"(Δ={pre_mape - best_mape:+.2f}%) | {total_time:.0f}s")
    return model, best_mape, pre_mape, total_time


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Type 1 Stage 2: finetune supervised_only with unlabeled data")
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--extra", action="store_true",
                        help="Include 40k extra unlabeled data")
    parser.add_argument("--ckpt-dir", type=str, default=CKPT_DIR,
                        help="Directory with supervised_only checkpoints")
    parser.add_argument("--results-dir", type=str, default="./results_run2")
    args = parser.parse_args()

    ratios = [args.ratio] if args.ratio is not None else ALL_RATIOS
    out_dir = f"{args.results_dir}/exp4_18k_pretrain_ft_type1"
    if args.extra:
        out_dir += "_extra"

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {args.ckpt_dir}")
    print(f"Output: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    print("\nLoading 18k data...")
    data_18k = load_all_benchmarks(
        DATASET_PATHS["18k"]["gem"],
        DATASET_PATHS["18k"]["concord"],
        BENCHMARK_NAMES,
    )

    data_40k = None
    if args.extra:
        print("Loading 40k data...")
        data_40k = load_all_benchmarks(
            DATASET_PATHS["40k"]["gem"],
            DATASET_PATHS["40k"]["concord"],
            BENCHMARK_NAMES,
        )

    for ratio in ratios:
        ratio_key = f"{int(round(ratio*100))}pct"
        ckpt_path = os.path.join(args.ckpt_dir, f"supervised_only_{ratio_key}.pt")

        if not os.path.exists(ckpt_path):
            print(f"\nWARNING: Checkpoint not found: {ckpt_path}, skipping {ratio_key}")
            continue

        print(f"\n{'#'*60}")
        print(f"# Type 1 Pretrain+FT | Labeled={ratio*100:.0f}% | "
              f"Extra={'yes' if args.extra else 'no'}")
        print(f"{'#'*60}")

        set_seed(SEED)

        # Create unlabeled dataloader
        (unlabeled_loader, test_loader,
         benchmark_test_indices, gem_norm) = create_finetune_dataloaders(
            data_18k, ratio, data_40k=data_40k,
        )

        # Load Stage 1 checkpoint
        model = GEMModel(GEM_INPUT_DIM, [512, 256], DROPOUT).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  Loaded checkpoint: {ckpt_path}")

        # Finetune
        model, best_mape, pre_mape, train_time = finetune(
            model, unlabeled_loader, test_loader, device, ratio,
        )

        # Final eval
        test_mape, cpi_pred, cpi_gold = evaluate(model, test_loader, device)
        per_bm = per_benchmark_mape(cpi_pred, cpi_gold, benchmark_test_indices)

        print(f"\nResults:")
        print(f"  Overall MAPE: {test_mape:.2f}%")
        for name in sorted(per_bm.keys()):
            print(f"  {name:<20} {per_bm[name]:>9.2f}%")

        result = {
            "type": "pretrain_ft_type1",
            "labeled_ratio": ratio,
            "pre_finetune_mape": pre_mape,
            "overall_mape": test_mape,
            "per_benchmark": per_bm,
            "finetune_epochs": FT_EPOCHS,
            "finetune_lr": FT_LR,
            "extra": args.extra,
            "train_time_seconds": train_time,
        }

        result_path = os.path.join(out_dir, f"exp4_{ratio_key}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {result_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()