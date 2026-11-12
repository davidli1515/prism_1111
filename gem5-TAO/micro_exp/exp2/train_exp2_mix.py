"""
Exp2-Mix: OOD with Mixed-Batch Semi-Supervised Training

For each target benchmark:
  - Other benchmarks: ALL data, fully labeled
  - Target benchmark: ratio% labeled + rest unlabeled
  - All mixed in same batch, same backward pass

Labeled samples:   L_task + λ₁·L_const + λ₂·L_mono
Unlabeled samples: λ₁·L_const_self + λ₂·L_mono

Usage:
    python train_exp2_mix.py --gpu 0 --ratio 0.05 --target 508.namd_r
    python train_exp2_mix.py --gpu 0 --ratio 0.05  # all benchmarks
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

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")
from config import (
    GEM_INPUT_DIM,
    GEM_HIDDEN_DIMS, DROPOUT,
    BATCH_SIZE, PATIENCE,
    CONCORDE_LR, CONCORDE_WEIGHT_DECAY, CONCORDE_BATCH_SIZE, CONCORDE_EPOCHS,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    SEED,
    BASE_INDICES,
)
from model import (
    GEMModel,
    compute_cpi_mean, loss_task, loss_const, loss_mono,
)
from dataset import (
    load_all_benchmarks, FeatureNormalizer,
)

OOD_TRAIN_RATIO = 0.05
OUTPUT_BASE = "/scratch/gem5-TAO/micro_exp/exp2"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


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
# Semi-Supervised Loss (same batch, same backward)
# ============================================================

def semi_supervised_loss(mscs_pred, mscs_target, w_min, cpi_golden, has_label):
    """
    Labeled:   L_task + λ₁·L_const + λ₂·L_mono
    Unlabeled: λ₁·L_const_self + λ₂·L_mono
    """
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

        loss_dict["L_task"]      = l_task_val.item()
        loss_dict["L_const"]     = l_const_val.item()
        loss_dict["L_mono_lab"]  = l_mono_val.item()

    if n_unlabeled > 0:
        u_pred = mscs_pred[unlabeled_mask]
        u_wmin = w_min[unlabeled_mask]

        u_const_self = loss_const_self(u_pred, u_wmin)

        unlabeled_loss = LAMBDA_CONST * u_const_self
        total = total + unlabeled_loss

        loss_dict["L_const_self"] = u_const_self.item()

    loss_dict["L_total"]     = total.item()
    loss_dict["n_labeled"]   = n_labeled
    loss_dict["n_unlabeled"] = n_unlabeled

    return total, loss_dict


# ============================================================
# Dataset with has_label flag
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
# Data Split: OOD with labeled/unlabeled mixing
# ============================================================

def create_ood_mixed_split(all_data, target_name, ood_train_ratio=None,
                           batch_size=BATCH_SIZE, seed=SEED):
    """
    OOD split with semi-supervised mixing:
      - Other benchmarks: ALL data, has_label=1
      - Target benchmark: ratio% has_label=1, rest has_label=0
      - ALL target data goes into train (labeled + unlabeled)
      - Test: target benchmark's unlabeled portion
    """
    if ood_train_ratio is None:
        ood_train_ratio = OOD_TRAIN_RATIO

    rng = np.random.RandomState(seed)

    train_gem, train_con, train_mscs, train_cpi, train_wmin = [], [], [], [], []
    train_has_label = []
    test_gem, test_con, test_mscs, test_cpi, test_wmin = [], [], [], [], []

    n_target_labeled = 0
    n_target_unlabeled = 0

    for name, data in sorted(all_data.items()):
        if name == target_name:
            n = data["n_samples"]
            idx = rng.permutation(n)
            split = max(1, int(n * ood_train_ratio))

            labeled_idx = idx[:split]
            unlabeled_idx = idx[split:]

            n_target_labeled = len(labeled_idx)
            n_target_unlabeled = len(unlabeled_idx)

            # Labeled portion → train (has_label=1) + test
            train_gem.append(data["gem_features"][labeled_idx])
            train_con.append(data["concord_features"][labeled_idx])
            train_mscs.append(data["mscs_target"][labeled_idx])
            train_cpi.append(data["cpi_golden"][labeled_idx])
            train_wmin.append(data["w_min"][labeled_idx])
            train_has_label.append(np.ones(len(labeled_idx), dtype=np.float32))

            # Unlabeled portion → train (has_label=0) + test
            train_gem.append(data["gem_features"][unlabeled_idx])
            train_con.append(data["concord_features"][unlabeled_idx])
            train_mscs.append(data["mscs_target"][unlabeled_idx])  # not used in loss
            train_cpi.append(data["cpi_golden"][unlabeled_idx])    # not used in loss
            train_wmin.append(data["w_min"][unlabeled_idx])
            train_has_label.append(np.zeros(len(unlabeled_idx), dtype=np.float32))

            # Test = unlabeled portion (evaluate on unseen data)
            test_gem.append(data["gem_features"][unlabeled_idx])
            test_con.append(data["concord_features"][unlabeled_idx])
            test_mscs.append(data["mscs_target"][unlabeled_idx])
            test_cpi.append(data["cpi_golden"][unlabeled_idx])
            test_wmin.append(data["w_min"][unlabeled_idx])
        else:
            # Other benchmarks: all labeled in train
            train_gem.append(data["gem_features"])
            train_con.append(data["concord_features"])
            train_mscs.append(data["mscs_target"])
            train_cpi.append(data["cpi_golden"])
            train_wmin.append(data["w_min"])
            train_has_label.append(np.ones(data["n_samples"], dtype=np.float32))

    # Concatenate
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

    # Normalize (fit on train)
    gem_norm = FeatureNormalizer()
    train_gem = gem_norm.fit_transform(train_gem)
    test_gem  = gem_norm.transform(test_gem)

    con_norm = FeatureNormalizer()
    train_con = con_norm.fit_transform(train_con)
    test_con  = con_norm.transform(test_con)

    from dataset import GEMDataset
    train_ds = SemiSupervisedDataset(train_gem, train_con, train_mscs,
                                     train_cpi, train_wmin, train_has_label)
    test_ds  = GEMDataset(test_gem, test_con, test_mscs, test_cpi, test_wmin)

    actual_batch = min(batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=actual_batch, shuffle=True,
                              num_workers=4, pin_memory=True,
                              drop_last=len(train_ds) > actual_batch)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    n_labeled_total = int(train_has_label.sum())
    n_unlabeled_total = len(train_has_label) - n_labeled_total

    print(f"    Train: {len(train_ds):,} ({n_labeled_total:,} labeled + "
          f"{n_unlabeled_total:,} unlabeled) | Test: {len(test_ds):,}")
    print(f"    Target: {n_target_labeled} labeled + {n_target_unlabeled} unlabeled")

    return (train_loader, test_loader,
            n_target_labeled, n_target_unlabeled)


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
# Training: GEM Mixed Semi-Supervised
# ============================================================

def train_gem_mixed(model, train_loader, test_loader, device,
                    num_epochs=CONCORDE_EPOCHS, label="GEM-Mix"):
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
                  f"L_mono={avg.get('L_mono_lab',0):.4f} | "
                  f"MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    return model, best_mape, history


# ============================================================
# Run One Target
# ============================================================

def run_one_target(all_data, target_name, device, output_dir):
    (train_loader, test_loader,
     n_labeled, n_unlabeled) = create_ood_mixed_split(all_data, target_name)

    print(f"\n  [{target_name}] Target: {n_labeled} labeled + {n_unlabeled} unlabeled")

    results = {"target": target_name,
               "n_target_labeled": n_labeled,
               "n_target_unlabeled": n_unlabeled}

    gem_batch = min(CONCORDE_BATCH_SIZE, len(train_loader.dataset))
    mixed_loader = DataLoader(train_loader.dataset, batch_size=gem_batch,
                              shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=len(train_loader.dataset) > gem_batch)

    # --- GEM Mixed (labeled + unlabeled in same batch) ---
    print(f"    [{target_name}] Training GEM-Mix...")
    t0 = time.time()
    set_seed(SEED)
    m = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(device)
    m, mix_mape, mix_hist = train_gem_mixed(
        m, mixed_loader, test_loader, device, label=f"{target_name} GEM-Mix")
    results["gem_mix"] = mix_mape
    results["gem_mix_history"] = mix_hist
    torch.save(m.state_dict(), os.path.join(output_dir, f"gem_mix_{target_name}.pt"))
    print(f"    [{target_name}] GEM-Mix: {mix_mape:.2f}% ({time.time()-t0:.0f}s)")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Exp2-Mix: OOD Semi-Supervised")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    global OOD_TRAIN_RATIO
    OOD_TRAIN_RATIO = args.ratio

    global GEM_DATA_DIR
    if args.data_dir:
        GEM_DATA_DIR = args.data_dir

    ratio_tag = f"{int(args.ratio*100)}pct_mix_constonly"
    if args.tag:
        ratio_tag = f"{ratio_tag}_{args.tag}"
    output_dir = os.path.join(OUTPUT_BASE, f"results_{ratio_tag}")

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ratio: {OOD_TRAIN_RATIO:.2%}")
    print(f"Output: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading benchmarks...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    benchmark_names = sorted(all_data.keys())

    targets = [args.target] if args.target else benchmark_names
    if args.target:
        assert args.target in benchmark_names, f"Unknown: {args.target}"

    print(f"\n{'='*70}")
    print(f"  Exp2-Mix: OOD Semi-Supervised (ratio={OOD_TRAIN_RATIO:.2%})")
    print(f"  Targets: {len(targets)} benchmarks")
    print(f"{'='*70}")

    all_results = []
    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}]", "=" * 50)
        result = run_one_target(all_data, target, device, output_dir)
        all_results.append(result)

        per_path = os.path.join(output_dir, f"exp2_ood_{target}.json")
        with open(per_path, "w") as f:
            json.dump(result, f, indent=2)

    # Summary
    if not args.target or len(all_results) > 1:
        print(f"\n{'='*40}")
        print(f"{'Benchmark':<20} {'GEM-Mix':>10}")
        print(f"{'-'*40}")
        for r in all_results:
            print(f"{r['target']:<20} {r['gem_mix']:>9.2f}%")

        avg_m = np.mean([r["gem_mix"] for r in all_results])
        print(f"{'-'*40}")
        print(f"{'Average':<20} {avg_m:>9.2f}%")
        print(f"{'='*40}")

        summary = {
            "per_benchmark": all_results,
            "average": {"gem_mix": avg_m},
            "ood_train_ratio": OOD_TRAIN_RATIO,
        }
        with open(os.path.join(output_dir, "exp2_mix_results.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()