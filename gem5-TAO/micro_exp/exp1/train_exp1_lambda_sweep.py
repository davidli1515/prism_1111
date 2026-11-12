"""
Exp1: LAMBDA_MONO hyperparameter sweep
  Fix L_task + L_const (LAMBDA_CONST=1.0), sweep LAMBDA_MONO in [0, 0.01, 0.1, 0.5, 1.0, 5.0]
  All use GEM+D3 (4681 dims).

Usage:
    python train_exp1_lambda_sweep.py --gpu 0
    python train_exp1_lambda_sweep.py --gpu 0 --lambda_mono 0.1  # single value (for parallel)
"""
import os, sys, json, argparse, numpy as np, torch, torch.optim as optim
from collections import defaultdict

from config import (
    GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE,
    LAMBDA_CONST,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    OUTPUT_DIR, SEED,
)
from model import GEMModel, compute_cpi_mean, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, create_exp1_dataloaders

SWEEP_DIR = os.path.join(OUTPUT_DIR, "lambda_sweep")
LAMBDA_MONO_VALUES = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True

def mape_fn(p, t, e=1e-8):
    return (torch.abs(p-t)/(torch.abs(t)+e)*100).mean().item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); P, G = [], []
    for b in loader:
        m = model(b["gem_features"].to(device))
        P.append(compute_cpi_mean(m, b["w_min"].to(device)).cpu())
        G.append(b["cpi_golden"])
    P, G = torch.cat(P), torch.cat(G)
    return mape_fn(P, G), P, G


def per_bm_mape(pred, gold, bm_idx):
    return {n: mape_fn(pred[s:e], gold[s:e]) for n,(s,e) in bm_idx.items() if s < e}


def train_one(model, tl, vl, device, lam_mono, label):
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    best, state = float("inf"), None

    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        for b in tl:
            f = b["gem_features"].to(device)
            mt, cg, wm = b["mscs_target"].to(device), b["cpi_golden"].to(device), b["w_min"].to(device)
            mp = model(f)
            l = loss_task(mp, mt) + LAMBDA_CONST * loss_const(mp, wm, cg)
            if lam_mono > 0:
                l = l + lam_mono * loss_mono(mp)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()

        m, _, _ = evaluate(model, vl, device)
        if m < best:
            best, state = m, {k:v.cpu().clone() for k,v in model.state_dict().items()}

        if ep % 50 == 0 or ep == 1:
            print(f"  [{label}] Ep {ep:3d} | MAPE={m:.2f}% (best={best:.2f}%)")

    model.load_state_dict(state)
    print(f"  [{label}] Best: {best:.2f}% (trained full {num_epochs} epochs)")
    return model, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lambda_mono", type=float, default=None,
                        help="Single value (for parallel)")
    args = parser.parse_args()

    set_seed(SEED)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(SWEEP_DIR, exist_ok=True)

    print("Loading...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)

    # Fix data split with original seed — same for all runs
    set_seed(SEED)
    (tl, vl, _, _, bm_idx) = create_exp1_dataloaders(all_data)

    values = [args.lambda_mono] if args.lambda_mono is not None else LAMBDA_MONO_VALUES

    results = {}
    for lam in values:
        label = f"λ_mono={lam}"
        print(f"\n{'='*50}\n  {label}\n{'='*50}")

        set_seed(SEED)
        model = GEMModel(GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT).to(dev)
        model, best = train_one(model, tl, vl, dev, lam, label)

        results[str(lam)] = {"lambda_mono": lam, "overall_mape": best}

        with open(os.path.join(SWEEP_DIR, f"lambda_mono_{lam}.json"), "w") as f:
            json.dump(results[str(lam)], f, indent=2)

    # Summary
    print(f"\n{'='*50}")
    print(f"{'λ_mono':>10} {'MAPE':>10}")
    print(f"{'-'*22}")
    for lam in values:
        r = results[str(lam)]
        print(f"{lam:>10} {r['overall_mape']:>9.2f}%")
    print(f"{'='*50}")

    with open(os.path.join(SWEEP_DIR, "lambda_sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {SWEEP_DIR}")


if __name__ == "__main__":
    main()