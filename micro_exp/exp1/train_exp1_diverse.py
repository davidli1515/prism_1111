#!/usr/bin/env python3
"""
Exp1 Diverse: Config generalization experiment.
Data: 2 configs/slice for train, 38 configs/slice for test.
Files: {benchmark}_train.npy and {benchmark}_test.npy

Usage:
    python train_exp1_diverse.py --gpu 0 --variant gem_full
    python train_exp1_diverse.py --gpu 0 --variant gem_full --hidden 256,128
    python train_exp1_diverse.py --gpu 0 --variant concorde --data_dir /path/to/diverse
"""

import os, sys, json, argparse, numpy as np, torch, torch.optim as optim
sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")
from collections import defaultdict
from torch.utils.data import DataLoader

from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM,
    CONCORDE_FEAT_END, D3_FEAT_END, MSCS_START, MSCS_END,
    WMIN_INDICES, BASE_INDICES,
    DROPOUT, LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, PATIENCE, COSINE_T_MAX,
    CONCORDE_LR, CONCORDE_WEIGHT_DECAY, CONCORDE_BATCH_SIZE, CONCORDE_EPOCHS,
    LAMBDA_CONST, LAMBDA_MONO,
    BENCHMARK_NAMES, OUTPUT_DIR, SEED,
)
from model import (
    GEMModel, ConcordeModel,
    compute_cpi_mean, loss_task, loss_const, loss_mono,
)
from dataset import FeatureNormalizer, GEMDataset

DEFAULT_DIVERSE_DIR = "/data/micro_traindata_packed_diverse/gem"


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def mape_fn(p, t, e=1e-8):
    return (torch.abs(p - t) / (torch.abs(t) + e) * 100).mean().item()


# ============================================================
# Data Loading
# ============================================================

def parse_npy(raw):
    """Parse a raw (N, 4697) array into feature dict."""
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    raw = raw.astype(np.float32)
    gem_features     = raw[:, :D3_FEAT_END]
    concord_features = raw[:, :CONCORDE_FEAT_END]
    mscs_target      = raw[:, MSCS_START:MSCS_END]
    widths           = raw[:, WMIN_INDICES]
    w_min            = widths.min(axis=1)
    base_commit      = mscs_target[:, BASE_INDICES[3]]
    cpi_golden       = 1.0 / (base_commit * w_min + 1e-8)
    return gem_features, concord_features, mscs_target, cpi_golden, w_min


def load_diverse_data(data_dir, benchmark_names=None):
    """
    Load pre-split diverse data: {benchmark}_train.npy + {benchmark}_test.npy
    Returns train/test arrays and per-benchmark test indices.
    """
    if benchmark_names is None:
        # Auto-discover from _train.npy files
        benchmark_names = sorted(set(
            f.replace("_train.npy", "")
            for f in os.listdir(data_dir)
            if f.endswith("_train.npy")
        ))

    print(f"  Found {len(benchmark_names)} benchmarks in {data_dir}")

    train_gem, train_con, train_mscs, train_cpi, train_wmin = [], [], [], [], []
    test_gem, test_con, test_mscs, test_cpi, test_wmin = [], [], [], [], []
    per_bm_test = {}
    test_offset = 0

    for name in sorted(benchmark_names):
        tr_path = os.path.join(data_dir, f"{name}_train.npy")
        te_path = os.path.join(data_dir, f"{name}_test.npy")

        if not os.path.exists(tr_path) or not os.path.exists(te_path):
            print(f"    WARNING: Missing {name}, skipping")
            continue

        tr_raw = np.load(tr_path, allow_pickle=True)
        te_raw = np.load(te_path, allow_pickle=True)

        tr_gem, tr_con, tr_mscs, tr_cpi, tr_wmin = parse_npy(tr_raw)
        te_gem, te_con, te_mscs, te_cpi, te_wmin = parse_npy(te_raw)

        train_gem.append(tr_gem); train_con.append(tr_con)
        train_mscs.append(tr_mscs); train_cpi.append(tr_cpi); train_wmin.append(tr_wmin)

        test_gem.append(te_gem); test_con.append(te_con)
        test_mscs.append(te_mscs); test_cpi.append(te_cpi); test_wmin.append(te_wmin)

        n_te = te_raw.shape[0]
        per_bm_test[name] = (test_offset, test_offset + n_te)
        test_offset += n_te

        print(f"    {name}: train={tr_raw.shape[0]}, test={te_raw.shape[0]}")

    # Concatenate
    train_gem_cat = np.concatenate(train_gem)
    train_con_cat = np.concatenate(train_con)
    test_gem_cat = np.concatenate(test_gem)
    test_con_cat = np.concatenate(test_con)

    # Normalize on train
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
    test_ds = GEMDataset(test_gem_cat, test_con_cat,
                          np.concatenate(test_mscs), np.concatenate(test_cpi),
                          np.concatenate(test_wmin))

    batch_size = min(CONCORDE_BATCH_SIZE, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"  Total: train={len(train_ds)}, test={len(test_ds)}, batch={batch_size}")
    return train_loader, test_loader, per_bm_test


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def eval_gem(model, loader, device, feat_key="gem_features"):
    model.eval(); P, G = [], []
    for b in loader:
        m = model(b[feat_key].to(device))
        P.append(compute_cpi_mean(m, b["w_min"].to(device)).cpu())
        G.append(b["cpi_golden"])
    P, G = torch.cat(P), torch.cat(G)
    return mape_fn(P, G), P, G

@torch.no_grad()
def eval_con(model, loader, device):
    model.eval(); P, G = [], []
    for b in loader:
        P.append(model(b["concord_features"].to(device)).cpu())
        G.append(b["cpi_golden"])
    P, G = torch.cat(P), torch.cat(G)
    return mape_fn(P, G), P, G

def per_bm_mape(pred, gold, bm_idx):
    return {n: mape_fn(pred[s:e], gold[s:e]) for n, (s, e) in bm_idx.items() if s < e}


# ============================================================
# Training: GEM
# ============================================================

def train_gem(model, tl, vl, device, use_const=True, use_mono=True,
              label="GEM", feat_key="gem_features"):
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR,
                            weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    best_mape, best_state, pc = float("inf"), None, 0
    history = []
    step = 0

    for epoch in range(1, CONCORDE_EPOCHS + 1):
        model.train()
        eloss = 0.0; nb = 0
        for b in tl:
            f = b[feat_key].to(device)
            mt, cg, wm = b["mscs_target"].to(device), b["cpi_golden"].to(device), b["w_min"].to(device)
            mp = model(f)
            l = loss_task(mp, mt)
            if use_const: l = l + LAMBDA_CONST * loss_const(mp, wm, cg)
            if use_mono:  l = l + LAMBDA_MONO * loss_mono(mp)
            optimizer.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); step += 1
            eloss += l.item(); nb += 1

        test_mape, _, _ = eval_gem(model, vl, device, feat_key)
        lr = optimizer.param_groups[0]['lr']
        history.append({"epoch": epoch, "loss": eloss/nb, "test_mape": test_mape,
                        "lr": lr, "step": step})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{label}] Ep {epoch:3d} (step {step}) | Loss={eloss/nb:.6f} | MAPE={test_mape:.2f}% | lr={lr:.2e}")

        if test_mape < best_mape:
            best_mape, best_state, pc = test_mape, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PATIENCE:
                print(f"  [{label}] Early stop ep {epoch} (best={best_mape:.2f}%)"); break

    model.load_state_dict(best_state)
    print(f"  [{label}] Best: {best_mape:.2f}%")
    return model, best_mape, history


# ============================================================
# Training: Concorde
# ============================================================

def train_concorde(model, tl, vl, device, label="Concorde"):
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR,
                            weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    def relative_loss(pred, target, eps=1e-8):
        return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()

    best_mape, best_state, pc = float("inf"), None, 0
    history = []
    step = 0

    for epoch in range(1, CONCORDE_EPOCHS + 1):
        model.train()
        eloss = 0.0; nb = 0
        for b in tl:
            p = model(b["concord_features"].to(device))
            l = relative_loss(p, b["cpi_golden"].to(device))
            optimizer.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); step += 1
            eloss += l.item(); nb += 1

        test_mape, _, _ = eval_con(model, vl, device)
        lr = optimizer.param_groups[0]['lr']
        history.append({"epoch": epoch, "loss": eloss/nb, "test_mape": test_mape,
                        "lr": lr, "step": step})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{label}] Ep {epoch:3d} (step {step}) | Loss={eloss/nb:.6f} | MAPE={test_mape:.2f}% | lr={lr:.2e}")

        if test_mape < best_mape:
            best_mape, best_state, pc = test_mape, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PATIENCE:
                print(f"  [{label}] Early stop ep {epoch} (best={best_mape:.2f}%)"); break

    model.load_state_dict(best_state)
    print(f"  [{label}] Best: {best_mape:.2f}%")
    return model, best_mape, history


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Exp1 Diverse: Config Generalization")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--variant", type=str, required=True,
                        choices=["gem_full", "gem_const", "gem_vanilla", "gem_nod3", "concorde"])
    parser.add_argument("--hidden", type=str, default="512,256",
                        help="Hidden dims, e.g. '512,256' or '256,128'")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DIVERSE_DIR,
                        help="Directory with {benchmark}_train.npy / _test.npy")
    args = parser.parse_args()

    HIDDEN = [int(x) for x in args.hidden.split(",")]
    hidden_tag = "_".join(str(x) for x in HIDDEN)
    save_dir = os.path.join(OUTPUT_DIR, f"exp1_diverse_{hidden_tag}")
    os.makedirs(save_dir, exist_ok=True)

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"Hidden: {HIDDEN}")
    print(f"Data: {args.data_dir}")
    print(f"Save: {save_dir}")
    print("Loading...")
    tl, vl, bm_idx = load_diverse_data(args.data_dir)

    if args.variant == "gem_full":
        set_seed(SEED)
        model = GEMModel(GEM_INPUT_DIM, HIDDEN, DROPOUT).to(device)
        n = sum(p.numel() for p in model.parameters())
        print(f"\nGEM+D3 Full {HIDDEN} ({n:,} params)")
        model, mape, hist = train_gem(model, tl, vl, device,
                                       use_const=True, use_mono=True, label="GEM+D3 Full")
        _, pred, gold = eval_gem(model, vl, device)
        pbm = per_bm_mape(pred, gold, bm_idx)

    elif args.variant == "gem_const":
        set_seed(SEED)
        model = GEMModel(GEM_INPUT_DIM, HIDDEN, DROPOUT).to(device)
        n = sum(p.numel() for p in model.parameters())
        print(f"\nGEM+D3 Const {HIDDEN} ({n:,} params)")
        model, mape, hist = train_gem(model, tl, vl, device,
                                       use_const=True, use_mono=False, label="GEM+D3 Const")
        _, pred, gold = eval_gem(model, vl, device)
        pbm = per_bm_mape(pred, gold, bm_idx)

    elif args.variant == "gem_vanilla":
        set_seed(SEED)
        model = GEMModel(GEM_INPUT_DIM, HIDDEN, DROPOUT).to(device)
        n = sum(p.numel() for p in model.parameters())
        print(f"\nGEM+D3 Vanilla {HIDDEN} ({n:,} params)")
        model, mape, hist = train_gem(model, tl, vl, device,
                                       use_const=False, use_mono=False, label="GEM+D3 Vanilla")
        _, pred, gold = eval_gem(model, vl, device)
        pbm = per_bm_mape(pred, gold, bm_idx)

    elif args.variant == "gem_nod3":
        set_seed(SEED)
        model = GEMModel(CONCORDE_INPUT_DIM, HIDDEN, DROPOUT).to(device)
        n = sum(p.numel() for p in model.parameters())
        print(f"\nGEM-noD3 {HIDDEN} ({n:,} params)")
        model, mape, hist = train_gem(model, tl, vl, device,
                                       use_const=True, use_mono=True, label="GEM-noD3",
                                       feat_key="concord_features")
        _, pred, gold = eval_gem(model, vl, device, feat_key="concord_features")
        pbm = per_bm_mape(pred, gold, bm_idx)

    elif args.variant == "concorde":
        set_seed(SEED)
        model = ConcordeModel(CONCORDE_INPUT_DIM, HIDDEN, 0.0).to(device)
        n = sum(p.numel() for p in model.parameters())
        print(f"\nConcorde {HIDDEN} ({n:,} params)")
        model, mape, hist = train_concorde(model, tl, vl, device)
        _, pred, gold = eval_con(model, vl, device)
        pbm = per_bm_mape(pred, gold, bm_idx)

    # Save
    result = {"variant": args.variant, "hidden": HIDDEN,
              "overall_mape": mape, "per_benchmark": pbm,
              "n_params": n, "loss_history": hist,
              "data_dir": args.data_dir}
    path = os.path.join(save_dir, f"{args.variant}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{args.variant}.pt"))
    print(f"\nSaved to {path}")

    # Print per-benchmark
    print(f"\n{'Benchmark':<20} {'MAPE':>8}")
    print("-" * 30)
    for bm in sorted(pbm.keys()):
        print(f"  {bm:<20} {pbm[bm]:>7.2f}%")
    print(f"  {'Overall':<20} {mape:>7.2f}%")


if __name__ == "__main__":
    main()
