"""
Experiment 1 Ablation: Loss Function × Feature Decomposition
  6 variants total:
    With D3 (4681 dims):                 Without D3 (3873 dims):
      d3_vanilla:   L_task                 nod3_vanilla:   L_task
      d3_mono:      L_task + L_mono        nod3_mono:      L_task + L_mono
      d3_const:     L_task + L_const       nod3_const:     L_task + L_const

  All use MSCS 16-dim output, [256,128] hidden.

Usage:
    python train_exp1_ablation.py --gpu 0                    # all 6 variants
    python train_exp1_ablation.py --gpu 0 --variant d3_const # single variant
    
    # Max parallel: 6 variants on 4 GPUs
    python train_exp1_ablation.py --gpu 0 --variant d3_vanilla &
    python train_exp1_ablation.py --gpu 1 --variant d3_mono &
    python train_exp1_ablation.py --gpu 2 --variant d3_const &
    python train_exp1_ablation.py --gpu 3 --variant nod3_vanilla &
    wait
    python train_exp1_ablation.py --gpu 0 --variant nod3_mono &
    python train_exp1_ablation.py --gpu 1 --variant nod3_const &
    wait
"""

import os, sys, json, argparse, numpy as np, torch, torch.optim as optim
from collections import defaultdict

from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE, COSINE_T_MAX,
    LAMBDA_CONST, LAMBDA_MONO, CONCORDE_WEIGHT_DECAY,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    OUTPUT_DIR, SEED,
)
from model import GEMModel, compute_cpi_mean, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, create_exp1_dataloaders

ABLATION_DIR = os.path.join(OUTPUT_DIR, "ablation")

# All 6 variants
VARIANTS = {
    "d3_vanilla":   {"input_dim": GEM_INPUT_DIM,      "use_gem": True,  "use_const": False, "use_mono": False,
                     "label": "D3 + L_task"},
    "d3_mono":      {"input_dim": GEM_INPUT_DIM,      "use_gem": True,  "use_const": False, "use_mono": True,
                     "label": "D3 + L_task + L_mono"},
    "d3_const":     {"input_dim": GEM_INPUT_DIM,      "use_gem": True,  "use_const": True,  "use_mono": False,
                     "label": "D3 + L_task + L_const"},
    "nod3_vanilla": {"input_dim": CONCORDE_INPUT_DIM, "use_gem": False, "use_const": False, "use_mono": False,
                     "label": "noD3 + L_task"},
    "nod3_mono":    {"input_dim": CONCORDE_INPUT_DIM, "use_gem": False, "use_const": False, "use_mono": True,
                     "label": "noD3 + L_task + L_mono"},
    "nod3_const":   {"input_dim": CONCORDE_INPUT_DIM, "use_gem": False, "use_const": True,  "use_mono": False,
                     "label": "noD3 + L_task + L_const"},
}


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True

def mape_fn(p, t, e=1e-8):
    return (torch.abs(p-t)/(torch.abs(t)+e)*100).mean().item()


@torch.no_grad()
def evaluate(model, loader, device, use_gem):
    model.eval(); P, G = [], []
    feat_key = "gem_features" if use_gem else "concord_features"
    for b in loader:
        m = model(b[feat_key].to(device))
        P.append(compute_cpi_mean(m, b["w_min"].to(device)).cpu())
        G.append(b["cpi_golden"])
    P, G = torch.cat(P), torch.cat(G)
    return mape_fn(P, G), P, G


def per_bm_mape(pred, gold, bm_idx):
    return {n: mape_fn(pred[s:e], gold[s:e]) for n,(s,e) in bm_idx.items() if s < e}


def train_one(model, tl, vl, device, use_gem, use_const, use_mono, label):
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=COSINE_T_MAX)
    fk = "gem_features" if use_gem else "concord_features"
    best, state, pc = float("inf"), None, 0
    history = []

    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        eloss = defaultdict(float); nb = 0
        for b in tl:
            f = b[fk].to(device)
            mt, cg, wm = b["mscs_target"].to(device), b["cpi_golden"].to(device), b["w_min"].to(device)
            mp = model(f)
            lt = loss_task(mp, mt); l = lt
            lc_v, lm_v = 0., 0.
            if use_const:
                lc = loss_const(mp, wm, cg); l = l + LAMBDA_CONST * lc; lc_v = lc.item()
            if use_mono:
                lm = loss_mono(mp); l = l + LAMBDA_MONO * lm; lm_v = lm.item()
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            eloss["L_total"] += l.item(); eloss["L_task"] += lt.item()
            eloss["L_const"] += lc_v; eloss["L_mono"] += lm_v; nb += 1

        sch.step()
        avg = {k: v/nb for k,v in eloss.items()}
        m, _, _ = evaluate(model, vl, device, use_gem)
        avg["test_mape"] = m; avg["epoch"] = ep; history.append(avg)

        if ep % 10 == 0 or ep == 1:
            print(f"  [{label}] Ep {ep:3d} | total={avg['L_total']:.6f} task={avg['L_task']:.6f} "
                  f"const={avg['L_const']:.6f} mono={avg['L_mono']:.6f} | MAPE={m:.2f}%")

        if m < best:
            best, state, pc = m, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PATIENCE:
                print(f"  [{label}] Early stop ep {ep} (best={best:.2f}%)"); break

    model.load_state_dict(state)
    print(f"  [{label}] Best: {best:.2f}%")
    return model, best, history


def run_variant(name, cfg, tl, vl, bm_idx, device):
    print(f"\n{'='*60}\n  {cfg['label']}\n{'='*60}")
    set_seed(SEED)
    model = GEMModel(cfg["input_dim"], CONCORDE_HIDDEN_DIMS, DROPOUT).to(device)
    print(f"  Input={cfg['input_dim']} Params={sum(p.numel() for p in model.parameters()):,}")

    model, best, hist = train_one(model, tl, vl, device,
                                   cfg["use_gem"], cfg["use_const"], cfg["use_mono"],
                                   cfg["label"])
    _, pred, gold = evaluate(model, vl, device, cfg["use_gem"])
    pbm = per_bm_mape(pred, gold, bm_idx)

    result = {
        "overall_mape": best, "per_benchmark": pbm,
        "loss_history": hist, "config": cfg,
    }
    # Save immediately
    path = os.path.join(ABLATION_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    torch.save(model.state_dict(), os.path.join(ABLATION_DIR, f"{name}.pt"))
    print(f"  Saved to {path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--variant", type=str, default=None,
                        choices=list(VARIANTS.keys()),
                        help="Single variant (for parallel runs)")
    args = parser.parse_args()

    set_seed(SEED)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(ABLATION_DIR, exist_ok=True)

    print("Loading..."); all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    (tl, vl, _, _, bm_idx) = create_exp1_dataloaders(all_data)

    targets = {args.variant: VARIANTS[args.variant]} if args.variant else VARIANTS

    for name, cfg in targets.items():
        run_variant(name, cfg, tl, vl, bm_idx, dev)

    # Summary (if running all)
    if not args.variant:
        print(f"\n{'='*75}")
        print(f"{'Variant':<25} {'MAPE':>8}")
        print(f"{'-'*35}")
        for name in VARIANTS:
            p = os.path.join(ABLATION_DIR, f"{name}.json")
            if os.path.exists(p):
                r = json.load(open(p))
                print(f"  {VARIANTS[name]['label']:<25} {r['overall_mape']:>7.2f}%")
        print(f"{'='*75}")


if __name__ == "__main__":
    main()