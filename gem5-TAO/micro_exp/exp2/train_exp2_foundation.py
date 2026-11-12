"""
Exp2 Two-Stage: Foundation Model + Fine-tune

Stage 1: Train 17 foundation models, each excluding one benchmark (0% target)
Stage 2: Fine-tune from checkpoint with X% target data (1%, 5%, 10%, 20%)

Usage:
    # Stage 1: train foundations (once)
    python train_exp2_foundation.py --gpu 0 --stage 1 --target 600.perlbench_s

    # Stage 2: fine-tune at various ratios (fast, ~50-100 epochs)
    python train_exp2_foundation.py --gpu 0 --stage 2 --target 600.perlbench_s --ratio 0.05

    # Both stages for one target
    python train_exp2_foundation.py --gpu 0 --stage both --target 600.perlbench_s --ratio 0.05
"""

import os, sys, json, time, argparse, numpy as np, torch, torch.optim as optim
sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")
from torch.utils.data import DataLoader

from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM,
    CONCORDE_HIDDEN_DIMS, GEM_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE, COSINE_T_MAX,
    CONCORDE_LR, CONCORDE_WEIGHT_DECAY, CONCORDE_BATCH_SIZE, CONCORDE_EPOCHS,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    SEED,
)
from model import (
    GEMModel, ConcordeModel,
    compute_cpi_mean, loss_task, loss_const, loss_mono,
)
from dataset import (
    load_all_benchmarks, FeatureNormalizer, GEMDataset,
)

FOUNDATION_DIR = "/scratch/gem5-TAO/micro_exp/exp2/foundations"
FT_EPOCHS = 100  # fine-tune epochs (much less than full training)


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


# ============================================================
# Data Split (same as train_exp2.py)
# ============================================================

def create_foundation_split(all_data, target_name, train_ratio=15/16,
                             batch_size=BATCH_SIZE, seed=SEED):
    """
    Foundation split: 16 seen benchmarks → 15:1 train/val.
    Target benchmark excluded from training, kept as separate test set.
    """
    rng = np.random.RandomState(seed)
    train_gem, train_con, train_mscs, train_cpi, train_wmin = [], [], [], [], []
    val_gem, val_con, val_mscs, val_cpi, val_wmin = [], [], [], [], []
    test_gem, test_con, test_mscs, test_cpi, test_wmin = [], [], [], [], []

    for name, data in sorted(all_data.items()):
        if name == target_name:
            # Target: all goes to test
            test_gem.append(data["gem_features"])
            test_con.append(data["concord_features"])
            test_mscs.append(data["mscs_target"])
            test_cpi.append(data["cpi_golden"])
            test_wmin.append(data["w_min"])
        else:
            # Seen benchmarks: 15:1 train/val
            n = data["n_samples"]
            idx = rng.permutation(n)
            split = int(n * train_ratio)
            tr_idx, va_idx = idx[:split], idx[split:]

            train_gem.append(data["gem_features"][tr_idx])
            train_con.append(data["concord_features"][tr_idx])
            train_mscs.append(data["mscs_target"][tr_idx])
            train_cpi.append(data["cpi_golden"][tr_idx])
            train_wmin.append(data["w_min"][tr_idx])

            val_gem.append(data["gem_features"][va_idx])
            val_con.append(data["concord_features"][va_idx])
            val_mscs.append(data["mscs_target"][va_idx])
            val_cpi.append(data["cpi_golden"][va_idx])
            val_wmin.append(data["w_min"][va_idx])

    train_gem_cat = np.concatenate(train_gem)
    train_con_cat = np.concatenate(train_con)
    val_gem_cat = np.concatenate(val_gem)
    val_con_cat = np.concatenate(val_con)
    test_gem_cat = np.concatenate(test_gem)
    test_con_cat = np.concatenate(test_con)

    gem_norm = FeatureNormalizer()
    gem_norm.fit(train_gem_cat)
    con_norm = FeatureNormalizer()
    con_norm.fit(train_con_cat)

    train_gem_cat = gem_norm.transform(train_gem_cat)
    train_con_cat = con_norm.transform(train_con_cat)
    val_gem_cat = gem_norm.transform(val_gem_cat)
    val_con_cat = con_norm.transform(val_con_cat)
    test_gem_cat = gem_norm.transform(test_gem_cat)
    test_con_cat = con_norm.transform(test_con_cat)

    train_ds = GEMDataset(train_gem_cat, train_con_cat,
                           np.concatenate(train_mscs), np.concatenate(train_cpi),
                           np.concatenate(train_wmin))
    val_ds = GEMDataset(val_gem_cat, val_con_cat,
                         np.concatenate(val_mscs), np.concatenate(val_cpi),
                         np.concatenate(val_wmin))
    test_ds = GEMDataset(test_gem_cat, test_con_cat,
                          np.concatenate(test_mscs), np.concatenate(test_cpi),
                          np.concatenate(test_wmin))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, gem_norm, con_norm


def create_ood_split(all_data, target_name, ood_train_ratio,
                     batch_size=BATCH_SIZE, seed=SEED):
    rng = np.random.RandomState(seed)
    train_gem, train_con, train_mscs, train_cpi, train_wmin = [], [], [], [], []
    test_gem, test_con, test_mscs, test_cpi, test_wmin = [], [], [], [], []

    for name, data in sorted(all_data.items()):
        if name == target_name:
            n = data["n_samples"]
            idx = rng.permutation(n)
            split = max(1, int(n * ood_train_ratio)) if ood_train_ratio > 0 else 0
            tr_idx, te_idx = idx[:split], idx[split:]

            if len(tr_idx) > 0:
                train_gem.append(data["gem_features"][tr_idx])
                train_con.append(data["concord_features"][tr_idx])
                train_mscs.append(data["mscs_target"][tr_idx])
                train_cpi.append(data["cpi_golden"][tr_idx])
                train_wmin.append(data["w_min"][tr_idx])

            test_gem.append(data["gem_features"][te_idx])
            test_con.append(data["concord_features"][te_idx])
            test_mscs.append(data["mscs_target"][te_idx])
            test_cpi.append(data["cpi_golden"][te_idx])
            test_wmin.append(data["w_min"][te_idx])
        else:
            train_gem.append(data["gem_features"])
            train_con.append(data["concord_features"])
            train_mscs.append(data["mscs_target"])
            train_cpi.append(data["cpi_golden"])
            train_wmin.append(data["w_min"])

    train_gem_cat = np.concatenate(train_gem)
    train_con_cat = np.concatenate(train_con)
    test_gem_cat = np.concatenate(test_gem)
    test_con_cat = np.concatenate(test_con)

    gem_norm = FeatureNormalizer()
    gem_norm.fit(train_gem_cat)
    con_norm = FeatureNormalizer()
    con_norm.fit(train_con_cat)

    train_gem_cat = gem_norm.transform(train_gem_cat)
    train_con_cat = con_norm.transform(train_con_cat)
    test_gem_cat = gem_norm.transform(test_gem_cat)
    test_con_cat = con_norm.transform(test_con_cat)

    train_ds = GEMDataset(train_gem_cat, train_con_cat,
                           np.concatenate(train_mscs), np.concatenate(train_cpi),
                           np.concatenate(train_wmin))
    test_ds  = GEMDataset(test_gem_cat, test_con_cat,
                           np.concatenate(test_mscs), np.concatenate(test_cpi),
                           np.concatenate(test_wmin))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    n_target_train = max(1, int(all_data[target_name]["n_samples"] * ood_train_ratio)) if ood_train_ratio > 0 else 0
    n_target_test  = all_data[target_name]["n_samples"] - n_target_train
    return train_loader, test_loader, n_target_train, n_target_test, gem_norm, con_norm


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_gem(model, loader, device, use_gem_features=True):
    model.eval()
    all_pred, all_gold = [], []
    feat_key = "gem_features" if use_gem_features else "concord_features"
    for batch in loader:
        mscs_pred = model(batch[feat_key].to(device))
        cpi_pred = compute_cpi_mean(mscs_pred, batch["w_min"].to(device))
        all_pred.append(cpi_pred.cpu()); all_gold.append(batch["cpi_golden"])
    return compute_mape(torch.cat(all_pred), torch.cat(all_gold))

@torch.no_grad()
def evaluate_concorde(model, loader, device):
    model.eval()
    all_pred, all_gold = [], []
    for batch in loader:
        cpi_pred = model(batch["concord_features"].to(device))
        all_pred.append(cpi_pred.cpu()); all_gold.append(batch["cpi_golden"])
    return compute_mape(torch.cat(all_pred), torch.cat(all_gold))


# ============================================================
# Training: GEM
# ============================================================

def train_gem(model, train_loader, test_loader, device, num_epochs, patience,
              label="GEM"):
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR,
                            weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    best_mape, best_state, pc = float("inf"), None, 0
    step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        ep_loss, nb = 0, 0
        for batch in train_loader:
            feat = batch["gem_features"].to(device)
            mscs_tgt = batch["mscs_target"].to(device)
            cpi_gold = batch["cpi_golden"].to(device)
            w_min = batch["w_min"].to(device)

            mscs_pred = model(feat)
            l_t = loss_task(mscs_pred, mscs_tgt)
            l_c = loss_const(mscs_pred, w_min, cpi_gold)
            l_m = loss_mono(mscs_pred)
            loss = l_t + LAMBDA_CONST * l_c + LAMBDA_MONO * l_m

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); step += 1
            ep_loss += loss.item(); nb += 1

        mape = evaluate_gem(model, test_loader, device)
        if mape < best_mape:
            best_mape, best_state, pc = mape, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= patience: break
        if epoch % 50 == 0 or epoch == 1:
            print(f"      [{label}] ep {epoch:3d} (step {step}) | Loss={ep_loss/nb:.4f} | MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    print(f"      [{label}] Best: {best_mape:.2f}% (stopped ep ~{epoch-patience})")
    return model, best_mape


# ============================================================
# Training: Concorde
# ============================================================

def train_concorde(model, train_loader, test_loader, device, num_epochs, patience,
                   label="Concorde"):
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR,
                            weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    def relative_loss(pred, target, eps=1e-8):
        return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()

    best_mape, best_state, pc = float("inf"), None, 0
    step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        ep_loss, nb = 0, 0
        for batch in train_loader:
            cpi_pred = model(batch["concord_features"].to(device))
            loss = relative_loss(cpi_pred, batch["cpi_golden"].to(device))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); step += 1
            ep_loss += loss.item(); nb += 1

        mape = evaluate_concorde(model, test_loader, device)
        if mape < best_mape:
            best_mape, best_state, pc = mape, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= patience: break
        if epoch % 50 == 0 or epoch == 1:
            print(f"      [{label}] ep {epoch:3d} (step {step}) | Loss={ep_loss/nb:.6f} | MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    print(f"      [{label}] Best: {best_mape:.2f}% (stopped ep ~{epoch-patience})")
    return model, best_mape


# ============================================================
# Stage 1: Train Foundation (0% target)
# ============================================================

def run_stage1(all_data, target_name, device):
    """Train GEM + Concorde foundation models on 16 seen benchmarks.
    Early stopping on seen-benchmark validation set (not target)."""
    print(f"\n  === Stage 1: Foundation for {target_name} ===")

    train_loader, val_loader, test_loader, gem_norm, con_norm = \
        create_foundation_split(all_data, target_name)
    print(f"    Train: {len(train_loader.dataset)} (16 seen benchmarks, 15/16 split)")
    print(f"    Val:   {len(val_loader.dataset)} (16 seen benchmarks, 1/16 split)")
    print(f"    Test:  {len(test_loader.dataset)} ({target_name}, unseen)")

    target_dir = os.path.join(FOUNDATION_DIR, target_name)
    os.makedirs(target_dir, exist_ok=True)

    # GEM: train on seen, early stop on seen-val, then evaluate on target
    set_seed(SEED)
    gem_batch = min(CONCORDE_BATCH_SIZE, len(train_loader.dataset))
    gem_loader = DataLoader(train_loader.dataset, batch_size=gem_batch,
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    gem = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(device)
    gem, gem_val_mape = train_gem(gem, gem_loader, val_loader, device,
                                   CONCORDE_EPOCHS, PATIENCE,
                                   label=f"GEM|{target_name}|foundation")
    gem_test_mape = evaluate_gem(gem, test_loader, device)
    torch.save(gem.state_dict(), os.path.join(target_dir, "gem_foundation.pt"))
    print(f"    GEM: val={gem_val_mape:.2f}% target={gem_test_mape:.2f}%")

    # Concorde: same
    set_seed(SEED)
    con_batch = min(CONCORDE_BATCH_SIZE, len(train_loader.dataset))
    con_loader = DataLoader(train_loader.dataset, batch_size=con_batch,
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    con = ConcordeModel(CONCORDE_INPUT_DIM, GEM_HIDDEN_DIMS, 0.0).to(device)
    con, con_val_mape = train_concorde(con, con_loader, val_loader, device,
                                        CONCORDE_EPOCHS, PATIENCE,
                                        label=f"Con|{target_name}|foundation")
    con_test_mape = evaluate_concorde(con, test_loader, device)
    torch.save(con.state_dict(), os.path.join(target_dir, "con_foundation.pt"))
    print(f"    Con: val={con_val_mape:.2f}% target={con_test_mape:.2f}%")

    # Save normalizers for stage 2
    import pickle
    with open(os.path.join(target_dir, "normalizers.pkl"), "wb") as f:
        pickle.dump({"gem_norm": gem_norm, "con_norm": con_norm}, f)

    result = {"target": target_name, "ratio": 0.0,
              "n_target_test": len(test_loader.dataset),
              "gem_val_mape": gem_val_mape, "gem_target_mape": gem_test_mape,
              "con_val_mape": con_val_mape, "con_target_mape": con_test_mape}
    with open(os.path.join(target_dir, "foundation_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# Stage 2: Fine-tune with X% target data
# ============================================================

def run_stage2(all_data, target_name, ratio, device):
    """Fine-tune foundation models with X% target data."""
    print(f"\n  === Stage 2: FT {target_name} @ {ratio:.0%} ===")

    target_dir = os.path.join(FOUNDATION_DIR, target_name)
    gem_ckpt = os.path.join(target_dir, "gem_foundation.pt")
    con_ckpt = os.path.join(target_dir, "con_foundation.pt")

    if not os.path.exists(gem_ckpt) or not os.path.exists(con_ckpt):
        print(f"    ERROR: Foundation not found for {target_name}. Run stage 1 first.")
        return None

    train_loader, test_loader, n_train, n_test, _, _ = \
        create_ood_split(all_data, target_name, ratio)
    print(f"    Train: {len(train_loader.dataset)} ({n_train} from {target_name})")
    print(f"    Test: {n_test} from {target_name}")

    ratio_tag = f"{int(ratio*100)}pct"
    out_dir = f"/scratch/gem5-TAO/micro_exp/exp2/results_{ratio_tag}_ft"
    os.makedirs(out_dir, exist_ok=True)

    # GEM: load foundation → fine-tune
    set_seed(SEED)
    gem = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(device)
    gem.load_state_dict(torch.load(gem_ckpt, map_location=device))
    gem_batch = min(CONCORDE_BATCH_SIZE, len(train_loader.dataset))
    gem_loader = DataLoader(train_loader.dataset, batch_size=gem_batch,
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    gem, gem_mape = train_gem(gem, gem_loader, test_loader, device,
                               FT_EPOCHS, PATIENCE,
                               label=f"GEM|{target_name}|{ratio_tag}")

    # Concorde: load foundation → fine-tune
    set_seed(SEED)
    con = ConcordeModel(CONCORDE_INPUT_DIM, GEM_HIDDEN_DIMS, 0.0).to(device)
    con.load_state_dict(torch.load(con_ckpt, map_location=device))
    con_batch = min(CONCORDE_BATCH_SIZE, len(train_loader.dataset))
    con_loader = DataLoader(train_loader.dataset, batch_size=con_batch,
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    con, con_mape = train_concorde(con, con_loader, test_loader, device,
                                    FT_EPOCHS, PATIENCE,
                                    label=f"Con|{target_name}|{ratio_tag}")

    result = {"target": target_name, "ratio": ratio,
              "n_target_train": n_train, "n_target_test": n_test,
              "gem_full": gem_mape, "concorde": con_mape}
    with open(os.path.join(out_dir, f"exp2_ood_{target_name}.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"    FT result: GEM={gem_mape:.2f}% Con={con_mape:.2f}%")
    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Exp2 Foundation + FT")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--stage", type=str, default="both",
                        choices=["1", "2", "both"])
    parser.add_argument("--ratio", type=float, default=0.05,
                        help="Target ratio for stage 2 (ignored for stage 1)")
    parser.add_argument("--small", action="store_true",
                        help="Use 183k dataset")
    args = parser.parse_args()

    global GEM_DATA_DIR, CONCORD_DATA_DIR, FOUNDATION_DIR
    if args.small:
        GEM_DATA_DIR = "/scratch2/wl730/micro_traindata_packed/gem"
        CONCORD_DATA_DIR = "/scratch2/wl730/micro_traindata_packed/concord"
        FOUNDATION_DIR = "/scratch/gem5-TAO/micro_exp/exp2/foundations_183k"

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    assert args.target in all_data, f"Unknown: {args.target}"

    if args.stage in ("1", "both"):
        run_stage1(all_data, args.target, device)

    if args.stage in ("2", "both"):
        run_stage2(all_data, args.target, args.ratio, device)

    print("\nDone!")


if __name__ == "__main__":
    main()