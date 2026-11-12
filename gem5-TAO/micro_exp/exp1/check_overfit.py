"""
Quick diagnostic: check train vs test MAPE to determine overfitting vs underfitting.
Usage: python check_overfit.py --gpu 0
"""
import sys, torch, os
sys.path.insert(0, ".")
from config import *
from model import *
from dataset import *

torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
dev = torch.device(f"cuda:{int(sys.argv[2]) if len(sys.argv)>2 else 0}")

all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
tl, vl, _, _, _ = create_exp1_dataloaders(all_data)

def mape(P, G):
    return ((P-G).abs()/(G.abs()+1e-8)*100).mean().item()

def eval_gem(model, loader, use_gem=True):
    model.eval(); P, G = [], []
    fk = "gem_features" if use_gem else "concord_features"
    with torch.no_grad():
        for b in loader:
            m = model(b[fk].to(dev))
            P.append(compute_cpi_mean(m, b['w_min'].to(dev)).cpu())
            G.append(b['cpi_golden'])
    return mape(torch.cat(P), torch.cat(G)), len(torch.cat(P))

def eval_con(model, loader):
    model.eval(); P, G = [], []
    with torch.no_grad():
        for b in loader:
            P.append(model(b['concord_features'].to(dev)).cpu())
            G.append(b['cpi_golden'])
    return mape(torch.cat(P), torch.cat(G)), len(torch.cat(P))

print(f"{'Model':<20} {'Train MAPE':>12} {'Test MAPE':>12} {'Gap':>8} {'Diagnosis':<15}")
print("-" * 70)

ckpts = [
    ("GEM+D3", "gem_full.pt", "gem", True),
    ("GEM-noD3", "gem_nod3.pt", "gem", False),
    ("Concorde", "concorde.pt", "con", None),
]

for name, ckpt, typ, use_gem in ckpts:
    path = os.path.join(OUTPUT_DIR, ckpt)
    if not os.path.exists(path):
        print(f"  {name:<20} (checkpoint not found: {path})")
        continue

    if typ == "gem":
        model = GEMModel(GEM_INPUT_DIM if use_gem else CONCORDE_INPUT_DIM,
                         CONCORDE_HIDDEN_DIMS, DROPOUT).to(dev)
        model.load_state_dict(torch.load(path, map_location=dev))
        tr_mape, tr_n = eval_gem(model, tl, use_gem)
        te_mape, te_n = eval_gem(model, vl, use_gem)
    else:
        model = ConcordeModel(CONCORDE_INPUT_DIM, CONCORDE_HIDDEN_DIMS, 0.0).to(dev)
        model.load_state_dict(torch.load(path, map_location=dev))
        tr_mape, tr_n = eval_con(model, tl)
        te_mape, te_n = eval_con(model, vl)

    gap = te_mape - tr_mape
    if gap > 2:
        diag = "OVERFIT"
    elif gap < 0.5:
        diag = "UNDERFIT"
    else:
        diag = "OK"

    print(f"  {name:<20} {tr_mape:>10.2f}% {te_mape:>10.2f}% {gap:>+7.2f}%  {diag}")

print(f"\n  Train samples: {tr_n}, Test samples: {te_n}")
