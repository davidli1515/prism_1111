"""
Exp1 [512,256]: Three variants with larger model.
  1. GEM+D3 full (L_task + L_const + L_mono)
  2. GEM+D3 const-only (L_task + L_const)
  3. Concorde (scalar CPI, paper recipe)

Usage:
    python train_exp1_183k.py --gpu 0 --variant gem_full
    python train_exp1_183k.py --gpu 1 --variant gem_const
    python train_exp1_183k.py --gpu 2 --variant concorde
"""
import os, sys, json, argparse, numpy as np, torch, torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader

from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM,
    CONCORDE_HIDDEN_DIMS, GEM_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, PATIENCE, COSINE_T_MAX,
    CONCORDE_LR, CONCORDE_WEIGHT_DECAY, CONCORDE_BATCH_SIZE, CONCORDE_EPOCHS,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    OUTPUT_DIR, SEED,
)
from model import (
    GEMModel, ConcordeModel,
    compute_cpi_mean, loss_task, loss_const, loss_mono,
)
from dataset import load_all_benchmarks, create_exp1_dataloaders

# Override to 183k data
GEM_DATA_DIR = "/scratch2/wl730/micro_traindata_packed/gem"
CONCORD_DATA_DIR = "/scratch2/wl730/micro_traindata_packed/concord"

HIDDEN = GEM_HIDDEN_DIMS  # from config
SAVE_DIR = os.path.join(OUTPUT_DIR, "exp1_183k")


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True

def mape_fn(p, t, e=1e-8):
    return (torch.abs(p-t)/(torch.abs(t)+e)*100).mean().item()


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def eval_gem(model, loader, device):
    model.eval(); P, G = [], []
    for b in loader:
        m = model(b["gem_features"].to(device))
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
    return {n: mape_fn(pred[s:e], gold[s:e]) for n,(s,e) in bm_idx.items() if s < e}


# ============================================================
# Training: GEM
# ============================================================

def train_gem(model, tl, vl, device, use_const=True, use_mono=True, label="GEM"):
    # Same recipe as Concorde: AdamW + MultiStepLR + 50k batch
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR,
                            weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)

    best_mape, best_state, pc = float("inf"), None, 0
    history = []
    step = 0

    for epoch in range(1, CONCORDE_EPOCHS + 1):
        model.train()
        eloss = 0.0; nb = 0
        for b in tl:
            f = b["gem_features"].to(device)
            mt, cg, wm = b["mscs_target"].to(device), b["cpi_golden"].to(device), b["w_min"].to(device)
            mp = model(f)
            l = loss_task(mp, mt)
            if use_const: l = l + LAMBDA_CONST * loss_const(mp, wm, cg)
            if use_mono:  l = l + LAMBDA_MONO * loss_mono(mp)
            optimizer.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); step += 1
            eloss += l.item(); nb += 1

        test_mape, _, _ = eval_gem(model, vl, device)
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
# Training: Concorde (paper recipe)
# ============================================================

def train_concorde(model, tl, vl, device):
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR, weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    rl = lambda p,t,e=1e-8: (torch.abs(p-t)/(torch.abs(t)+e)).mean()
    best_mape, best_state, pc = float("inf"), None, 0
    history = []; step = 0

    for epoch in range(1, CONCORDE_EPOCHS + 1):
        model.train()
        eloss = 0; nb = 0
        for b in tl:
            p = model(b["concord_features"].to(device))
            l = rl(p, b["cpi_golden"].to(device))
            optimizer.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); step += 1
            eloss += l.item(); nb += 1

        test_mape, _, _ = eval_con(model, vl, device)
        history.append({"epoch": epoch, "loss": eloss/nb, "test_mape": test_mape, "step": step})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Concorde] Ep {epoch:3d} (step {step}) | Loss={eloss/nb:.6f} | MAPE={test_mape:.2f}%")

        if test_mape < best_mape:
            best_mape, best_state, pc = test_mape, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PATIENCE:
                print(f"  [Concorde] Early stop ep {epoch} (best={best_mape:.2f}%)"); break

    model.load_state_dict(best_state)
    print(f"  [Concorde] Best: {best_mape:.2f}%")
    return model, best_mape, history


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--variant", type=str, required=True,
                        choices=["gem_full", "gem_const", "concorde"])
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Hidden: {HIDDEN}")
    print("Loading...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    (tl, vl, _, _, bm_idx) = create_exp1_dataloaders(all_data)

    if args.variant == "gem_full":
        set_seed(SEED)
        gem_batch = min(CONCORDE_BATCH_SIZE, len(tl.dataset))
        gem_loader = DataLoader(tl.dataset, batch_size=gem_batch,
                                shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        model = GEMModel(GEM_INPUT_DIM, HIDDEN, DROPOUT).to(device)
        n = sum(p.numel() for p in model.parameters())
        print(f"\nGEM+D3 Full [512,256] ({n:,} params, batch={gem_batch})")
        model, mape, hist = train_gem(model, gem_loader, vl, device,
                                       use_const=True, use_mono=True, label="GEM+D3 Full")
        _, pred, gold = eval_gem(model, vl, device)
        pbm = per_bm_mape(pred, gold, bm_idx)

    elif args.variant == "gem_const":
        set_seed(SEED)
        gem_batch = min(CONCORDE_BATCH_SIZE, len(tl.dataset))
        gem_loader = DataLoader(tl.dataset, batch_size=gem_batch,
                                shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        model = GEMModel(GEM_INPUT_DIM, HIDDEN, DROPOUT).to(device)
        n = sum(p.numel() for p in model.parameters())
        print(f"\nGEM+D3 Const-only [512,256] ({n:,} params, batch={gem_batch})")
        model, mape, hist = train_gem(model, gem_loader, vl, device,
                                       use_const=True, use_mono=False, label="GEM+D3 Const")
        _, pred, gold = eval_gem(model, vl, device)
        pbm = per_bm_mape(pred, gold, bm_idx)

    elif args.variant == "concorde":
        set_seed(SEED)
        con_batch = min(CONCORDE_BATCH_SIZE, len(tl.dataset))
        con_loader = DataLoader(tl.dataset, batch_size=con_batch,
                                shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        model = ConcordeModel(CONCORDE_INPUT_DIM, HIDDEN, 0.0).to(device)
        n = sum(p.numel() for p in model.parameters())
        print(f"\nConcorde [512,256] ({n:,} params, batch={con_batch})")
        model, mape, hist = train_concorde(model, con_loader, vl, device)
        _, pred, gold = eval_con(model, vl, device)
        pbm = per_bm_mape(pred, gold, bm_idx)

    # Save
    result = {"variant": args.variant, "hidden": HIDDEN,
              "overall_mape": mape, "per_benchmark": pbm,
              "n_params": n, "loss_history": hist}
    path = os.path.join(SAVE_DIR, f"{args.variant}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{args.variant}.pt"))
    print(f"\nSaved to {path}")

    # Print per-benchmark
    print(f"\n{'Benchmark':<20} {'MAPE':>8}")
    print("-" * 30)
    for bm in sorted(pbm.keys()):
        print(f"  {bm:<20} {pbm[bm]:>7.2f}%")
    print(f"  {'Overall':<20} {mape:>7.2f}%")


if __name__ == "__main__":
    main()