"""
Experiment 4 Extension: Extra Unlabeled Data from 40k dataset
Recipe: v77 (matching exp1_v3_run2)

18k train/test split unchanged. 40k extra samples added as unlabeled.
Test on 18k's original test set for direct comparison.

Usage:
    python train_exp4_extra_unlabeled.py --ratio 0.20 --gpu 0
"""

import os, sys, time, json, hashlib, argparse
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")

from config import (
    GEM_INPUT_DIM, DROPOUT, BENCHMARK_NAMES,
    SEED, TRAIN_RATIO,
    BASE_INDICES,
)
from model import GEMModel, compute_cpi_mean, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, FeatureNormalizer

# Recipe v77
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.3
BATCH_SIZE = 50000
NUM_EPOCHS = 1521
PATIENCE = 100
LR_GAMMA = 0.5
MILESTONE_FRACS = [0.42, 0.58, 0.75, 0.92]
FIXED_MILESTONES_18K = [2555, 3528, 4563, 5597]
LAMBDA_TASK = 0.1
LAMBDA_CONST = 1.0
LAMBDA_MONO = 0.01
LAMBDA_UNLAB = 1.0
RAMP_EPOCHS = 0

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


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); torch.backends.cudnn.deterministic = True

def stable_hash(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)

MIN_LR = 0.0  # default: no floor

def step_lr(optimizer, global_step, milestones, min_lr=MIN_LR):
    n_passed = sum(1 for m in milestones if global_step >= m)
    new_lr = max(LEARNING_RATE * (LR_GAMMA ** n_passed), min_lr)
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr

def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()

def loss_const_self(mscs_pred, w_min, eps=1e-8):
    """Self-consistency: variance of CPI across 4 stages (no golden needed)"""
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi_per_stage = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    cpi_mean = cpi_per_stage.mean(dim=1, keepdim=True)
    diff = cpi_per_stage - cpi_mean
    return (diff ** 2).mean()


class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, gem_features, mscs_target, cpi_golden, w_min, has_label):
        self.gem_features = torch.from_numpy(gem_features)
        self.mscs_target  = torch.from_numpy(mscs_target)
        self.cpi_golden   = torch.from_numpy(cpi_golden)
        self.w_min        = torch.from_numpy(w_min)
        self.has_label    = torch.from_numpy(has_label.astype(np.float32))

    def __len__(self): return self.gem_features.shape[0]

    def __getitem__(self, idx):
        return {
            "gem_features": self.gem_features[idx],
            "mscs_target":  self.mscs_target[idx],
            "cpi_golden":   self.cpi_golden[idx],
            "w_min":        self.w_min[idx],
            "has_label":    self.has_label[idx],
        }


def find_extra_indices(data_18k_bench, data_40k_bench):
    """Find rows in 40k NOT in 18k via byte-level hash."""
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
                                  batch_size=BATCH_SIZE, seed=SEED):
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

        test_gem.append(d18["gem_features"][te_idx])
        test_mscs.append(d18["mscs_target"][te_idx])
        test_cpi.append(d18["cpi_golden"][te_idx])
        test_wmin.append(d18["w_min"][te_idx])

        n_test = len(te_idx)
        per_benchmark_test[name] = (test_offset, test_offset + n_test)
        test_offset += n_test

        # Extra unlabeled from 40k
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

    # Normalizer: fit on ALL training data (18k train + extra)
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


def weighted_loss_task(mscs_pred, mscs_target, base_weight=1.0,
                       renorm_total=False, eps=1e-8):
    """Static Base emphasis with optional renorm to preserve L_const anchor."""
    err2 = (mscs_pred - mscs_target) ** 2
    D = mscs_pred.shape[1]

    dim_w = torch.ones(D, device=mscs_pred.device)
    dim_w[BASE_INDICES] = base_weight
    W = dim_w.unsqueeze(0).expand_as(err2)

    weighted = W * err2
    if renorm_total:
        with torch.no_grad():
            scale = err2.mean() / (weighted.mean() + eps)
            weighted = weighted * scale
    return weighted.mean()


def semi_supervised_loss(mscs_pred, mscs_target, w_min, cpi_golden, has_label,
                         unlab_scale=1.0, base_weight=1.0, renorm_total=False,
                         no_mono=False):
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

        l_task  = weighted_loss_task(l_pred, l_tgt, base_weight, renorm_total)
        l_const = loss_const(l_pred, l_wmin, l_cpi)

        labeled_loss = LAMBDA_TASK * l_task + LAMBDA_CONST * l_const
        if not no_mono:
            l_mono = loss_mono(l_pred)
            labeled_loss = labeled_loss + LAMBDA_MONO * l_mono
        loss_dict["L_task"]  = l_task.item()
        loss_dict["L_const"] = l_const.item()
        total = total + labeled_loss

    if n_unlabeled > 0:
        u_pred = mscs_pred[unlabeled_mask]
        u_wmin = w_min[unlabeled_mask]

        u_const_self = loss_const_self(u_pred, u_wmin)
        unlabeled_loss = LAMBDA_CONST * u_const_self
        if not no_mono:
            u_mono = loss_mono(u_pred)
            unlabeled_loss = unlabeled_loss + LAMBDA_MONO * u_mono

        loss_dict["L_const_self"] = u_const_self.item()
        total = total + unlab_scale * unlabeled_loss

    loss_dict["L_total"] = total.item()
    return total, loss_dict


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


def per_benchmark_mape(cpi_pred, cpi_gold, benchmark_indices):
    results = {}
    for name, (start, end) in benchmark_indices.items():
        if start == end: continue
        results[name] = compute_mape(cpi_pred[start:end], cpi_gold[start:end])
    return results


def train(model, train_loader, test_loader, device, labeled_ratio,
          num_epochs=NUM_EPOCHS, lambda_unlab=LAMBDA_UNLAB, ramp_epochs=RAMP_EPOCHS,
          base_weight=1.0, renorm_total=False, use_cosine=False,
          use_late_milestones=False, min_lr=0.0, no_mono=False):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    # LR schedule
    if use_cosine:
        import math
        def cosine_lr(optimizer, step, total):
            lr = max(LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * step / total)), min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        lr_fn = lambda step: cosine_lr(optimizer, step, total_steps)
        milestones = None
    else:
        if use_late_milestones:
            fracs = [0.42, 0.58, 0.88, 0.96]
        else:
            fracs = MILESTONE_FRACS
        milestones = [int(f * total_steps) for f in fracs]
        lr_fn = lambda step: step_lr(optimizer, step, milestones, min_lr)

    best_mape = float("inf")
    best_state = None
    global_step = 0
    t0 = time.time()

    flags = []
    if lambda_unlab != 1.0: flags.append(f"λ_unlab={lambda_unlab}")
    if ramp_epochs > 0: flags.append(f"ramp={ramp_epochs}ep")
    if use_cosine: flags.append("cosine_LR")
    if use_late_milestones: flags.append("late_milestones")

    print(f"\n{'='*60}")
    print(f"Training — labeled={labeled_ratio*100:.0f}% + extra unlabeled")
    print(f"  {num_epochs} epochs, no early stopping, steps/epoch={steps_per_epoch}")
    if use_cosine:
        print(f"  Cosine LR: {LEARNING_RATE} → 0 over {total_steps} steps")
    else:
        print(f"  Dynamic milestones={milestones}, gamma={LR_GAMMA}")
    print(f"  Loss: {LAMBDA_TASK}×L_task + {LAMBDA_CONST}×L_const(rel) + {LAMBDA_MONO}×L_mono")
    if flags: print(f"  Enabled: {', '.join(flags)}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        # Ramp-up
        if ramp_epochs > 0 and epoch <= ramp_epochs:
            unlab_scale = lambda_unlab * (epoch / ramp_epochs)
        else:
            unlab_scale = lambda_unlab

        for batch in train_loader:
            feat      = batch["gem_features"].to(device)
            mscs_tgt  = batch["mscs_target"].to(device)
            cpi_gold  = batch["cpi_golden"].to(device)
            w_min     = batch["w_min"].to(device)
            has_label = batch["has_label"].to(device)

            mscs_pred = model(feat)
            loss, loss_dict = semi_supervised_loss(
                mscs_pred, mscs_tgt, w_min, cpi_gold, has_label,
                unlab_scale=unlab_scale,
                base_weight=base_weight,
                renorm_total=renorm_total,
                no_mono=no_mono,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            lr_fn(global_step)

            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    epoch_losses[k] += v
            n_batches += 1

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
    print(f"  Best: {best_mape:.2f}% ({time.time()-t0:.0f}s)")
    return model, best_mape, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--lambda-unlab", type=float, default=LAMBDA_UNLAB,
                        help="Weight for unlabeled physics loss (default: 1.0)")
    parser.add_argument("--ramp-epochs", type=int, default=RAMP_EPOCHS,
                        help="Linear ramp-up epochs for unlabeled loss (default: 0)")
    parser.add_argument("--base-weight", type=float, default=1.0,
                        help="Extra weight for Base dims in L_task (default: 1.0)")
    parser.add_argument("--renorm-total", action="store_true",
                        help="Renormalize L_task to preserve L_const anchor share")
    parser.add_argument("--cosine", action="store_true",
                        help="Use cosine annealing LR instead of step-based")
    parser.add_argument("--late-milestones", action="store_true",
                        help="Push last two milestones later: [42,58,88,96]%% instead of [42,58,75,92]%%")
    parser.add_argument("--min-lr", type=float, default=0.0,
                        help="LR floor (default: 0 = no floor)")
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256],
                        help="Hidden dims (default: 512 256)")
    parser.add_argument("--no-mono", action="store_true",
                        help="Remove L_mono from both labeled and unlabeled loss")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed (default: from config)")
    args = parser.parse_args()

    ratios = [args.ratio] if args.ratio is not None else ALL_RATIOS
    actual_seed = args.seed if args.seed is not None else SEED
    out_dir = f"{args.results_dir}/exp4_18k_extra_unlabeled"

    set_seed(actual_seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Recipe: v77 | Seed: {actual_seed}")
    print(f"Output: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    print("\nLoading 18k...")
    data_18k = load_all_benchmarks(DATASET_PATHS["18k"]["gem"],
                                   DATASET_PATHS["18k"]["concord"], BENCHMARK_NAMES)
    print("Loading 40k...")
    data_40k = load_all_benchmarks(DATASET_PATHS["40k"]["gem"],
                                   DATASET_PATHS["40k"]["concord"], BENCHMARK_NAMES)

    for ratio in ratios:
        ratio_key = f"{int(round(ratio*100))}pct"

        print(f"\n{'#'*60}")
        print(f"# Extra Unlabeled | Labeled={ratio*100:.0f}% | GPU={args.gpu}")
        print(f"{'#'*60}")

        set_seed(actual_seed)

        (train_loader, test_loader, benchmark_test_indices,
         n_labeled, n_unlabeled_18k, n_extra) = create_dataloaders_with_extra(
            data_18k, data_40k, ratio,
        )

        model = GEMModel(input_dim=GEM_INPUT_DIM, hidden_dims=args.hidden,
                          dropout=DROPOUT).to(device)
        print(f"  GEM params: {sum(p.numel() for p in model.parameters()):,}")

        model, best_mape, train_time = train(
            model, train_loader, test_loader, device, ratio,
            lambda_unlab=args.lambda_unlab,
            ramp_epochs=args.ramp_epochs,
            base_weight=args.base_weight,
            renorm_total=args.renorm_total,
            use_cosine=args.cosine,
            use_late_milestones=args.late_milestones,
            min_lr=args.min_lr,
            no_mono=args.no_mono,
        )

        test_mape, cpi_pred, cpi_gold = evaluate(model, test_loader, device)
        per_bm = per_benchmark_mape(cpi_pred, cpi_gold, benchmark_test_indices)

        # P90: per-sample relative error, take 90th percentile
        per_sample_err = (torch.abs(cpi_pred - cpi_gold) / (torch.abs(cpi_gold) + 1e-8) * 100).numpy()
        p90_overall = float(np.percentile(per_sample_err, 90))
        p90_per_bm = {}
        for name, (s, e) in benchmark_test_indices.items():
            if s < e:
                p90_per_bm[name] = float(np.percentile(per_sample_err[s:e], 90))
        print(f"  Overall MAPE: {test_mape:.2f}%, P90: {p90_overall:.2f}%")

        # Per-dim MSCS error for diagnostic
        model.eval()
        all_mscs_pred, all_mscs_tgt = [], []
        with torch.no_grad():
            for batch in test_loader:
                feat = batch["gem_features"].to(device)
                all_mscs_pred.append(model(feat).cpu())
                all_mscs_tgt.append(batch["mscs_target"])
        all_mscs_pred = torch.cat(all_mscs_pred)
        all_mscs_tgt = torch.cat(all_mscs_tgt)
        per_dim_mse = ((all_mscs_pred - all_mscs_tgt) ** 2).mean(dim=0).tolist()
        base_mse = sum(per_dim_mse[i] for i in BASE_INDICES) / len(BASE_INDICES)
        other_mse = sum(per_dim_mse[i] for i in range(16) if i not in BASE_INDICES) / (16 - len(BASE_INDICES))

        result = {
            "dataset": "18k_extra_unlabeled",
            "labeled_ratio": ratio,
            "recipe": "v77",
            "n_labeled": n_labeled,
            "n_unlabeled_18k": n_unlabeled_18k,
            "n_extra_40k": n_extra,
            "overall_mape": test_mape,
            "p90": p90_overall,
            "per_benchmark": per_bm,
            "p90_per_benchmark": p90_per_bm,
            "per_dim_mse": per_dim_mse,
            "base_dim_mse": base_mse,
            "other_dim_mse": other_mse,
            "train_time_seconds": train_time,
        }

        result_path = os.path.join(out_dir, f"exp4_{ratio_key}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {result_path}")

        # Save checkpoint + normalizer
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        if args.seed is not None:
            ckpt_base = f"extra_{ratio_key}_s{actual_seed}"
        else:
            ckpt_base = f"extra_{ratio_key}"
        ckpt_path = os.path.join(ckpt_dir, f"{ckpt_base}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Checkpoint saved {ckpt_path}")

        # Save normalizer (mean_ and std_) for consistent inference
        import pickle
        norm_path = os.path.join(ckpt_dir, f"{ckpt_base}_normalizer.pkl")
        with open(norm_path, "wb") as nf:
            pickle.dump({"mean": gem_norm.mean_, "std": gem_norm.std_}, nf)
        print(f"  Normalizer saved {norm_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()