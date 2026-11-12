"""
Parameter-matched Concorde: increase hidden to [308,128] to match GEM+D3's 1.23M params.
Compare with original Concorde [256,128] and GEM+D3.

Usage: python train_concorde_matched.py --gpu 0
"""
import os, sys, json, numpy as np, torch, torch.optim as optim, argparse
from torch.utils.data import DataLoader

from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE,
    CONCORDE_LR, CONCORDE_WEIGHT_DECAY,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    OUTPUT_DIR, SEED,
)
from model import ConcordeModel
from dataset import load_all_benchmarks, create_exp1_dataloaders


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True

def mape_fn(p, t, e=1e-8):
    return (torch.abs(p-t)/(torch.abs(t)+e)*100).mean().item()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); P, G = [], []
    for b in loader:
        P.append(model(b["concord_features"].to(device)).cpu())
        G.append(b["cpi_golden"])
    return mape_fn(torch.cat(P), torch.cat(G))

def train(model, tl, vl, device, label):
    opt = optim.AdamW(model.parameters(), lr=CONCORDE_LR, weight_decay=CONCORDE_WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    rl = lambda p,t,e=1e-8: (torch.abs(p-t)/(torch.abs(t)+e)).mean()
    best, state, pc = float("inf"), None, 0

    for ep in range(1, NUM_EPOCHS+1):
        model.train()
        for b in tl:
            p = model(b["concord_features"].to(device))
            l = rl(p, b["cpi_golden"].to(device))
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        m = evaluate(model, vl, device)
        if m < best:
            best, state, pc = m, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PATIENCE: break
        if ep % 10 == 0 or ep == 1:
            print(f"  [{label}] Ep {ep:3d} | MAPE={m:.2f}%")

    model.load_state_dict(state)
    print(f"  [{label}] Best: {best:.2f}%")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    set_seed(SEED)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Loading...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    (tl, vl, _, _, _) = create_exp1_dataloaders(all_data)

    configs = [
        ("Concorde [308,128]", [308, 128]),
    ]

    print(f"\n{'Model':<25} {'Params':>10} {'MAPE':>8}")
    print("-" * 45)

    for label, hidden in configs:
        set_seed(SEED)
        model = ConcordeModel(CONCORDE_INPUT_DIM, hidden, 0.0).to(dev)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  {label} ({n_params:,} params)")
        mape = train(model, tl, vl, dev, label)
        print(f"  {label:<25} {n_params:>10,} {mape:>7.2f}%")

    print("\n(Compare with GEM+D3 [256,128]: 1,233,552 params, 4.87%)")


if __name__ == "__main__":
    main()
