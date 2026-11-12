"""
Exp2 Stage 2 (Server B): Fine-tune foundation models.
Loads per-benchmark checkpoints from /scratch/gem5-TAO/micro_exp/exp2/foundations/{benchmark}/

Usage:
    python train_exp2_finetune_serverB.py --gpu 0 --target 508.namd_r --n_samples 128
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
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM,
    GEM_HIDDEN_DIMS, DROPOUT,
    BATCH_SIZE, PATIENCE,
    CONCORDE_LR, CONCORDE_WEIGHT_DECAY, CONCORDE_BATCH_SIZE, CONCORDE_EPOCHS,
    LAMBDA_CONST, LAMBDA_MONO,
    SEED,
)
from model import (
    GEMModel, ConcordeModel,
    compute_cpi_mean, loss_task, loss_const, loss_mono,
)
from dataset import FeatureNormalizer, GEMDataset

GEM_DATA_DIR   = "/scratch2/wl730/micro_traindata_packed_all/gem"
FOUNDATION_DIR = "/scratch/gem5-TAO/micro_exp/exp2/foundations"
OUTPUT_BASE    = "/scratch/gem5-TAO/micro_exp/exp2"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


# ============================================================
# Data Split
# ============================================================

def create_target_split(target_data, target_name, n_samples, foundation_dir,
                        batch_size=BATCH_SIZE, seed=SEED):
    rng = np.random.RandomState(seed)
    n = target_data["n_samples"]
    idx = rng.permutation(n)
    split = min(n_samples, n - 1)
    tr_idx, te_idx = idx[:split], idx[split:]

    bm_dir = os.path.join(foundation_dir, target_name)
    gem_norm = FeatureNormalizer()
    nd = np.load(os.path.join(bm_dir, "gem_norm.npz"))
    gem_norm.mean, gem_norm.std = nd["mean"], nd["std"]

    con_norm = FeatureNormalizer()
    nd = np.load(os.path.join(bm_dir, "con_norm.npz"))
    con_norm.mean, con_norm.std = nd["mean"], nd["std"]

    train_gem = gem_norm.transform(target_data["gem_features"][tr_idx])
    test_gem  = gem_norm.transform(target_data["gem_features"][te_idx])
    train_con = con_norm.transform(target_data["concord_features"][tr_idx])
    test_con  = con_norm.transform(target_data["concord_features"][te_idx])

    train_ds = GEMDataset(train_gem, train_con,
                          target_data["mscs_target"][tr_idx],
                          target_data["cpi_golden"][tr_idx],
                          target_data["w_min"][tr_idx])
    test_ds  = GEMDataset(test_gem, test_con,
                          target_data["mscs_target"][te_idx],
                          target_data["cpi_golden"][te_idx],
                          target_data["w_min"][te_idx])

    actual_batch = min(batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=actual_batch, shuffle=True,
                              num_workers=4, pin_memory=True,
                              drop_last=len(train_ds) > actual_batch)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"    Train: {len(train_ds)} | Test: {len(test_ds)}")
    return train_loader, test_loader, len(tr_idx), len(te_idx)


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


@torch.no_grad()
def evaluate_concorde(model, loader, device):
    model.eval()
    all_pred, all_gold = [], []
    for batch in loader:
        feat = batch["concord_features"].to(device)
        cpi_pred = model(feat)
        all_pred.append(cpi_pred.cpu())
        all_gold.append(batch["cpi_golden"])
    return compute_mape(torch.cat(all_pred), torch.cat(all_gold))


# ============================================================
# Fine-tuning: GEM
# ============================================================

def finetune_gem(model, train_loader, test_loader, device,
                 num_epochs=CONCORDE_EPOCHS, label="GEM-FT"):
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR,
                            weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    best_mape, best_state, patience_ctr = float("inf"), None, 0
    step = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss, epoch_lt, epoch_lc, epoch_lm, nb = 0, 0, 0, 0, 0
        for batch in train_loader:
            feat     = batch["gem_features"].to(device)
            mscs_tgt = batch["mscs_target"].to(device)
            cpi_gold = batch["cpi_golden"].to(device)
            w_min    = batch["w_min"].to(device)

            mscs_pred = model(feat)
            l_t = loss_task(mscs_pred, mscs_tgt)
            l_c = loss_const(mscs_pred, w_min, cpi_gold)
            l_m = loss_mono(mscs_pred)
            loss = l_t + LAMBDA_CONST * l_c + LAMBDA_MONO * l_m

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_loss += loss.item(); epoch_lt += l_t.item()
            epoch_lc += l_c.item(); epoch_lm += l_m.item(); nb += 1

        mape = evaluate_gem(model, test_loader, device)
        history.append({"epoch": epoch, "step": step, "test_mape": mape,
                        "loss": epoch_loss/nb, "L_task": epoch_lt/nb,
                        "L_const": epoch_lc/nb, "L_mono": epoch_lm/nb})
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
# Fine-tuning: Concorde
# ============================================================

def finetune_concorde(model, train_loader, test_loader, device,
                      num_epochs=CONCORDE_EPOCHS, label="Con-FT"):
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR,
                            weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    def relative_loss(pred, target, eps=1e-8):
        return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()

    best_mape, best_state, patience_ctr = float("inf"), None, 0
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss, nb = 0, 0
        for batch in train_loader:
            feat = batch["concord_features"].to(device)
            cpi_gold = batch["cpi_golden"].to(device)
            cpi_pred = model(feat)
            loss = relative_loss(cpi_pred, cpi_gold)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item(); nb += 1

        mape = evaluate_concorde(model, test_loader, device)
        history.append({"epoch": epoch, "test_mape": mape, "loss": epoch_loss/nb})
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
            print(f"        [{label}] epoch {epoch:3d} | MAPE={mape:.2f}%")

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

    ratio_tag = f"{args.n_samples}shot_ft"
    if args.tag:
        ratio_tag = f"{ratio_tag}_{args.tag}"
    output_dir = os.path.join(OUTPUT_BASE, f"results_{ratio_tag}")

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    bm_dir = os.path.join(FOUNDATION_DIR, args.target)
    gem_ckpt = os.path.join(bm_dir, "gem_foundation.pt")
    con_ckpt = os.path.join(bm_dir, "con_foundation.pt")
    assert os.path.exists(gem_ckpt), f"Not found: {gem_ckpt}"
    assert os.path.exists(con_ckpt), f"Not found: {con_ckpt}"

    print(f"Device: {device}")
    print(f"Target: {args.target} | N samples: {args.n_samples}")
    print(f"Foundation: {bm_dir}")
    print(f"Output: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading target benchmark...")
    from dataset import load_benchmark
    target_data = load_benchmark(args.target, GEM_DATA_DIR)
    print(f"    {args.target}: {target_data['n_samples']} samples")

    train_loader, test_loader, n_train, n_test = create_target_split(
        target_data, args.target, args.n_samples, FOUNDATION_DIR)

    results = {"target": args.target, "n_samples": args.n_samples,
               "n_train": n_train, "n_test": n_test}

    # --- GEM ---
    print(f"\n    [{args.target}] GEM: load foundation → fine-tune...")
    t0 = time.time()
    gem_model = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(device)
    gem_model.load_state_dict(torch.load(gem_ckpt, map_location=device))
    gem_zero = evaluate_gem(gem_model, test_loader, device)
    print(f"    [{args.target}] GEM zero-shot: {gem_zero:.2f}%")
    results["gem_zero_shot"] = gem_zero

    gem_model, gem_mape, gem_hist = finetune_gem(
        gem_model, train_loader, test_loader, device,
        label=f"{args.target} GEM-FT")
    results["gem_full"] = gem_mape
    results["gem_full_history"] = gem_hist
    torch.save(gem_model.state_dict(), os.path.join(output_dir, f"gem_ft_{args.target}.pt"))
    print(f"    [{args.target}] GEM FT: {gem_mape:.2f}% ({time.time()-t0:.0f}s)")

    # --- Concorde ---
    print(f"\n    [{args.target}] Concorde: load foundation → fine-tune...")
    t0 = time.time()
    con_model = ConcordeModel(CONCORDE_INPUT_DIM, GEM_HIDDEN_DIMS, 0.0).to(device)
    con_model.load_state_dict(torch.load(con_ckpt, map_location=device))
    con_zero = evaluate_concorde(con_model, test_loader, device)
    print(f"    [{args.target}] Concorde zero-shot: {con_zero:.2f}%")
    results["con_zero_shot"] = con_zero

    con_model, con_mape, con_hist = finetune_concorde(
        con_model, train_loader, test_loader, device,
        label=f"{args.target} Con-FT")
    results["concorde"] = con_mape
    results["concorde_history"] = con_hist
    torch.save(con_model.state_dict(), os.path.join(output_dir, f"con_ft_{args.target}.pt"))
    print(f"    [{args.target}] Concorde FT: {con_mape:.2f}% ({time.time()-t0:.0f}s)")

    print(f"\n    [{args.target}] Summary ({args.n_samples} samples):")
    print(f"      GEM:      {gem_zero:.2f}% → {gem_mape:.2f}%")
    print(f"      Concorde: {con_zero:.2f}% → {con_mape:.2f}%")

    json_path = os.path.join(output_dir, f"exp2_ood_{args.target}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Saved: {json_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
