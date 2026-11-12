"""
GEM Vanilla (L_task only) - Foundation + Finetune
Ablation: no L_const, no L_mono

Usage:
    # Stage 1: foundation models
    python train_exp2_vanilla.py base --gpu 0 --target 508.namd_r
    python train_exp2_vanilla.py base --gpu 0  # all 17

    # Stage 2: finetune
    python train_exp2_vanilla.py finetune --gpu 0 --target 508.namd_r --n_samples 128
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
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    SEED, TRAIN_RATIO,
)
from model import GEMModel, compute_cpi_mean, loss_task
from dataset import load_all_benchmarks, FeatureNormalizer, GEMDataset

BASE_DIR   = "/scratch/gem5-TAO/micro_exp/exp2/base_models_vanilla"
OUTPUT_BASE = "/scratch/gem5-TAO/micro_exp/exp2"


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); torch.backends.cudnn.deterministic = True

def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()

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
# Training (L_task only)
# ============================================================

def train_gem_vanilla(model, train_loader, val_loader, device,
                      num_epochs=CONCORDE_EPOCHS, label="GEM-Vanilla"):
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR,
                            weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    best_mape, best_state, patience_ctr = float("inf"), None, 0
    step = 0; history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss, nb = 0, 0
        for b in train_loader:
            feat     = b["gem_features"].to(device)
            mscs_tgt = b["mscs_target"].to(device)
            mscs_pred = model(feat)
            loss = loss_task(mscs_pred, mscs_tgt)

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            step += 1; epoch_loss += loss.item(); nb += 1

        mape = evaluate_gem(model, val_loader, device)
        history.append({"epoch": epoch, "step": step, "mape": mape, "loss": epoch_loss/nb})
        if mape < best_mape:
            best_mape = mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"        [{label}] Early stop ep {epoch} (best={best_mape:.2f}%)")
                break
        if epoch % 50 == 0:
            print(f"        [{label}] epoch {epoch:3d} (step {step}) | MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    return model, best_mape, history


# ============================================================
# Stage 1: Foundation
# ============================================================

def run_base(all_data, target_name, device, output_dir):
    rng = np.random.RandomState(SEED)
    train_gem, train_con, train_mscs, train_cpi, train_wmin = [], [], [], [], []
    val_gem, val_con, val_mscs, val_cpi, val_wmin = [], [], [], [], []

    for name, data in sorted(all_data.items()):
        if name == target_name:
            continue
        n = data["n_samples"]; idx = rng.permutation(n); split = int(n * TRAIN_RATIO)
        tr, va = idx[:split], idx[split:]
        train_gem.append(data["gem_features"][tr]); train_con.append(data["concord_features"][tr])
        train_mscs.append(data["mscs_target"][tr]); train_cpi.append(data["cpi_golden"][tr])
        train_wmin.append(data["w_min"][tr])
        val_gem.append(data["gem_features"][va]); val_con.append(data["concord_features"][va])
        val_mscs.append(data["mscs_target"][va]); val_cpi.append(data["cpi_golden"][va])
        val_wmin.append(data["w_min"][va])

    train_gem = np.concatenate(train_gem); val_gem = np.concatenate(val_gem)
    train_con = np.concatenate(train_con); val_con = np.concatenate(val_con)
    train_mscs = np.concatenate(train_mscs); val_mscs = np.concatenate(val_mscs)
    train_cpi = np.concatenate(train_cpi); val_cpi = np.concatenate(val_cpi)
    train_wmin = np.concatenate(train_wmin); val_wmin = np.concatenate(val_wmin)

    gem_norm = FeatureNormalizer()
    train_gem = gem_norm.fit_transform(train_gem); val_gem = gem_norm.transform(val_gem)
    con_norm = FeatureNormalizer()
    train_con = con_norm.fit_transform(train_con); val_con = con_norm.transform(val_con)

    train_ds = GEMDataset(train_gem, train_con, train_mscs, train_cpi, train_wmin)
    val_ds = GEMDataset(val_gem, val_con, val_mscs, val_cpi, val_wmin)

    print(f"    [{target_name}] Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    set_seed(SEED)
    batch = min(CONCORDE_BATCH_SIZE, len(train_ds))
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    t0 = time.time()
    m = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(device)
    m, val_mape, hist = train_gem_vanilla(m, tl, vl, device, label=f"{target_name} Vanilla-Base")

    torch.save(m.state_dict(), os.path.join(output_dir, f"gem_base_{target_name}.pt"))
    np.savez(os.path.join(output_dir, f"gem_norm_{target_name}.npz"),
             mean=gem_norm.mean, std=gem_norm.std)

    result = {"target": target_name, "val_mape": val_mape, "history": hist}
    with open(os.path.join(output_dir, f"base_{target_name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"    [{target_name}] Val={val_mape:.2f}% ({time.time()-t0:.0f}s)")
    return result


# ============================================================
# Stage 2: Finetune
# ============================================================

def run_finetune(target_name, n_samples, device, base_dir, output_dir, data_dir):
    from dataset import load_benchmark
    target_data = load_benchmark(target_name, data_dir)

    rng = np.random.RandomState(SEED)
    n = target_data["n_samples"]; idx = rng.permutation(n)
    split = min(n_samples, n - 1); tr_idx, te_idx = idx[:split], idx[split:]

    gem_norm = FeatureNormalizer()
    nd = np.load(os.path.join(base_dir, f"gem_norm_{target_name}.npz"))
    gem_norm.mean, gem_norm.std = nd["mean"], nd["std"]

    train_gem = gem_norm.transform(target_data["gem_features"][tr_idx])
    test_gem = gem_norm.transform(target_data["gem_features"][te_idx])
    dummy_con_tr = np.zeros((len(tr_idx), 1), dtype=np.float32)
    dummy_con_te = np.zeros((len(te_idx), 1), dtype=np.float32)

    train_ds = GEMDataset(train_gem, dummy_con_tr, target_data["mscs_target"][tr_idx],
                          target_data["cpi_golden"][tr_idx], target_data["w_min"][tr_idx])
    test_ds = GEMDataset(test_gem, dummy_con_te, target_data["mscs_target"][te_idx],
                         target_data["cpi_golden"][te_idx], target_data["w_min"][te_idx])

    batch = min(BATCH_SIZE, len(train_ds))
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=len(train_ds) > batch)
    tel = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"    [{target_name}] Train: {len(train_ds)} | Test: {len(test_ds)}")

    ckpt = os.path.join(base_dir, f"gem_base_{target_name}.pt")
    m = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    zero = evaluate_gem(m, tel, device)
    print(f"    [{target_name}] Zero-shot: {zero:.2f}%")

    t0 = time.time()
    m, mape, hist = train_gem_vanilla(m, tl, tel, device, label=f"{target_name} Vanilla-FT")

    results = {"target": target_name, "n_samples": n_samples,
               "n_train": len(tr_idx), "n_test": len(te_idx),
               "gem_zero_shot": zero, "gem_vanilla": mape, "gem_vanilla_history": hist}
    torch.save(m.state_dict(), os.path.join(output_dir, f"gem_vanilla_{target_name}.pt"))

    json_path = os.path.join(output_dir, f"exp2_ood_{target_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    [{target_name}] {zero:.2f}% → {mape:.2f}% ({time.time()-t0:.0f}s)")
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["base", "finetune"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or GEM_DATA_DIR
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.mode}")

    if args.mode == "base":
        os.makedirs(BASE_DIR, exist_ok=True)
        print(f"Output: {BASE_DIR}")
        all_data = load_all_benchmarks(data_dir, CONCORD_DATA_DIR, BENCHMARK_NAMES)
        targets = [args.target] if args.target else sorted(all_data.keys())
        for i, t in enumerate(targets):
            print(f"\n[{i+1}/{len(targets)}] {'='*50}")
            run_base(all_data, t, device, BASE_DIR)

    elif args.mode == "finetune":
        assert args.n_samples, "--n_samples required for finetune"
        tag = f"{args.n_samples}shot_vanilla_ft"
        if args.tag: tag = f"{tag}_{args.tag}"
        output_dir = os.path.join(OUTPUT_BASE, f"results_{tag}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output: {output_dir}")

        if args.target:
            run_finetune(args.target, args.n_samples, device, BASE_DIR, output_dir, data_dir)
        else:
            names = sorted([f[:-4] for f in os.listdir(data_dir) if f.endswith(".npy")])
            for i, t in enumerate(names):
                print(f"\n[{i+1}/{len(names)}] {'='*50}")
                run_finetune(t, args.n_samples, device, BASE_DIR, output_dir, data_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
