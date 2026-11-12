"""
Exp2 OOD Ablation: Loss decomposition at 5% target train ratio
  4 variants: Full, L_task+L_const, L_task+L_mono, Concorde
  
Usage:
    python train_exp2_ood_ablation.py --gpu 0 --target 508.namd_r
    python train_exp2_ood_ablation.py --gpu 0  # all benchmarks
"""

import os, sys, json, argparse, time, numpy as np, torch, torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")
from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE,
    CONCORDE_LR, CONCORDE_WEIGHT_DECAY,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES, SEED,
)
from model import GEMModel, ConcordeModel, compute_cpi_mean, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, FeatureNormalizer, GEMDataset

OUT_DIR = "/path/to/gem5/micro_exp/exp2/results_ood_ablation"
OOD_RATIO = 0.05


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True

def mape(p, t, e=1e-8):
    return (torch.abs(p-t)/(torch.abs(t)+e)*100).mean().item()


def build_split(all_data, target):
    rng = np.random.RandomState(SEED)
    n = all_data[target]["n_samples"]
    perm = rng.permutation(n)
    split = max(1, int(n * OOD_RATIO))
    tr_idx, te_idx = perm[:split], perm[split:]

    tg, tc, tm, tp, tw = [], [], [], [], []
    for name, d in sorted(all_data.items()):
        if name == target:
            tg.append(d["gem_features"][tr_idx]); tc.append(d["concord_features"][tr_idx])
            tm.append(d["mscs_target"][tr_idx]); tp.append(d["cpi_golden"][tr_idx])
            tw.append(d["w_min"][tr_idx])
        else:
            tg.append(d["gem_features"]); tc.append(d["concord_features"])
            tm.append(d["mscs_target"]); tp.append(d["cpi_golden"])
            tw.append(d["w_min"])

    tg,tc,tm,tp,tw = [np.concatenate(x) for x in [tg,tc,tm,tp,tw]]
    td = all_data[target]
    vg,vc,vm,vp,vw = td["gem_features"][te_idx], td["concord_features"][te_idx], \
                      td["mscs_target"][te_idx], td["cpi_golden"][te_idx], td["w_min"][te_idx]

    gn = FeatureNormalizer(); tg = gn.fit_transform(tg); vg = gn.transform(vg)
    cn = FeatureNormalizer(); tc = cn.fit_transform(tc); vc = cn.transform(vc)

    tr = GEMDataset(tg,tc,tm,tp,tw); te = GEMDataset(vg,vc,vm,vp,vw)
    tl = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return tl, vl, split


@torch.no_grad()
def eval_gem(model, loader, dev):
    model.eval(); P,G = [],[]
    for b in loader:
        m = model(b["gem_features"].to(dev))
        P.append(compute_cpi_mean(m, b["w_min"].to(dev)).cpu()); G.append(b["cpi_golden"])
    return mape(torch.cat(P), torch.cat(G))

@torch.no_grad()
def eval_con(model, loader, dev):
    model.eval(); P,G = [],[]
    for b in loader:
        P.append(model(b["concord_features"].to(dev)).cpu()); G.append(b["cpi_golden"])
    return mape(torch.cat(P), torch.cat(G))


def train_gem_v(model, tl, vl, dev, use_const, use_mono):
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    best, state, pc = float("inf"), None, 0
    for ep in range(1, NUM_EPOCHS+1):
        model.train()
        for b in tl:
            f,mt,cg,wm = b["gem_features"].to(dev), b["mscs_target"].to(dev), \
                          b["cpi_golden"].to(dev), b["w_min"].to(dev)
            mp = model(f); l = loss_task(mp, mt)
            if use_const: l = l + LAMBDA_CONST * loss_const(mp, wm, cg)
            if use_mono:  l = l + LAMBDA_MONO * loss_mono(mp)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        m = eval_gem(model, vl, dev)
        if m < best: best,state,pc = m, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PATIENCE: break
    model.load_state_dict(state); return best


def train_con(model, tl, vl, dev):
    opt = optim.AdamW(model.parameters(), lr=CONCORDE_LR, weight_decay=CONCORDE_WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    rl = lambda p,t,e=1e-8: (torch.abs(p-t)/(torch.abs(t)+e)).mean()
    best, state, pc = float("inf"), None, 0
    for ep in range(1, NUM_EPOCHS+1):
        model.train()
        for b in tl:
            p = model(b["concord_features"].to(dev))
            l = rl(p, b["cpi_golden"].to(dev))
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        m = eval_con(model, vl, dev)
        if m < best: best,state,pc = m, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PATIENCE: break
    model.load_state_dict(state); return best


def run_one(all_data, target, dev):
    tl, vl, n_tr = build_split(all_data, target)
    print(f"\n  {target}: {n_tr} target train samples")
    res = {"target": target, "n_target_train": n_tr}

    # Full
    set_seed(SEED); m = GEMModel(GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT).to(dev)
    res["full"] = train_gem_v(m, tl, vl, dev, True, True)
    print(f"    Full:          {res['full']:.2f}%")

    # L_task + L_const
    set_seed(SEED); m = GEMModel(GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT).to(dev)
    res["const_only"] = train_gem_v(m, tl, vl, dev, True, False)
    print(f"    L_task+L_const:{res['const_only']:.2f}%")

    # L_task + L_mono
    set_seed(SEED); m = GEMModel(GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT).to(dev)
    res["mono_only"] = train_gem_v(m, tl, vl, dev, False, True)
    print(f"    L_task+L_mono: {res['mono_only']:.2f}%")

    # Save GEM results immediately (don't wait for Concorde)
    with open(os.path.join(OUT_DIR, f"ood_abl_{target}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"    [GEM variants saved]")

    # Concorde (may take longer)
    set_seed(SEED); m = ConcordeModel(CONCORDE_INPUT_DIM, CONCORDE_HIDDEN_DIMS, 0.0).to(dev)
    res["concorde"] = train_con(m, tl, vl, dev)
    print(f"    Concorde:      {res['concorde']:.2f}%")

    # Save again with Concorde
    with open(os.path.join(OUT_DIR, f"ood_abl_{target}.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    set_seed(SEED)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading..."); all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    bms = sorted(all_data.keys())
    targets = [args.target] if args.target else bms

    results = []
    for t in targets:
        results.append(run_one(all_data, t, dev))

    if not args.target:
        print(f"\n{'='*75}")
        print(f"{'Benchmark':<20} {'Full':>8} {'Const':>8} {'Mono':>8} {'Concorde':>10}")
        print(f"{'-'*75}")
        for r in results:
            print(f"{r['target']:<20} {r['full']:>7.2f}% {r['const_only']:>7.2f}% {r['mono_only']:>7.2f}% {r['concorde']:>9.2f}%")
        print(f"{'-'*75}")
        print(f"{'Average':<20} {np.mean([r['full'] for r in results]):>7.2f}% "
              f"{np.mean([r['const_only'] for r in results]):>7.2f}% "
              f"{np.mean([r['mono_only'] for r in results]):>7.2f}% "
              f"{np.mean([r['concorde'] for r in results]):>9.2f}%")

        with open(os.path.join(OUT_DIR, "ood_ablation_summary.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()