"""
Exp2 Mix: Semi-supervised OOD with mixed batch training.
Two separate forward passes per step to avoid BatchNorm contamination:
  Pass 1 (labeled):   L_task + L_const + L_mono
  Pass 2 (unlabeled): L_mono only

Usage:
    python train_exp2_mix_331.py --gpu 0 --ratio 0.05 --target 508.namd_r
    python train_exp2_mix_331.py --gpu 1 --ratio 0.0  --target 602.gcc_s
"""
import os, sys, json, time, argparse, numpy as np, torch, torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")
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

OUT_BASE = "/scratch/gem5-TAO/micro_exp/exp2/results_mix"


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


# ============================================================
# Data Split: returns labeled_loader, unlabeled_loader, test_loader
# ============================================================

def create_ood_mix_split(all_data, target_name, ood_train_ratio, seed=SEED):
    """
    Returns 3 loaders:
      - labeled_loader: seen benchmarks (100%) + target labeled (ratio%)
      - unlabeled_loader: target unlabeled (1-ratio%) — features only, for L_mono
      - test_loader: target test (same as unlabeled, but for evaluation)
    """
    rng = np.random.RandomState(seed)

    # Labeled: seen benchmarks + target labeled portion
    lab_gem, lab_con, lab_mscs, lab_cpi, lab_wmin = [], [], [], [], []
    # Unlabeled: target remaining
    unl_gem = []
    # Test: same as unlabeled
    test_gem, test_con, test_mscs, test_cpi, test_wmin = [], [], [], [], []

    for name, data in sorted(all_data.items()):
        if name == target_name:
            n = data["n_samples"]
            idx = rng.permutation(n)
            split = int(n * ood_train_ratio)  # 0 when ratio=0
            tr_idx, rest_idx = idx[:split], idx[split:]

            # Labeled portion of target
            if split > 0:
                lab_gem.append(data["gem_features"][tr_idx])
                lab_con.append(data["concord_features"][tr_idx])
                lab_mscs.append(data["mscs_target"][tr_idx])
                lab_cpi.append(data["cpi_golden"][tr_idx])
                lab_wmin.append(data["w_min"][tr_idx])

            # Unlabeled portion = rest of target
            unl_gem.append(data["gem_features"][rest_idx])

            # Test = same as unlabeled
            test_gem.append(data["gem_features"][rest_idx])
            test_con.append(data["concord_features"][rest_idx])
            test_mscs.append(data["mscs_target"][rest_idx])
            test_cpi.append(data["cpi_golden"][rest_idx])
            test_wmin.append(data["w_min"][rest_idx])
        else:
            # All of seen benchmark goes to labeled
            lab_gem.append(data["gem_features"])
            lab_con.append(data["concord_features"])
            lab_mscs.append(data["mscs_target"])
            lab_cpi.append(data["cpi_golden"])
            lab_wmin.append(data["w_min"])

    # Concatenate
    lab_gem  = np.concatenate(lab_gem)
    lab_con  = np.concatenate(lab_con)
    lab_mscs = np.concatenate(lab_mscs)
    lab_cpi  = np.concatenate(lab_cpi)
    lab_wmin = np.concatenate(lab_wmin)

    unl_gem = np.concatenate(unl_gem)

    test_gem  = np.concatenate(test_gem)
    test_con  = np.concatenate(test_con)
    test_mscs = np.concatenate(test_mscs)
    test_cpi  = np.concatenate(test_cpi)
    test_wmin = np.concatenate(test_wmin)

    # Normalize (fit on labeled only)
    gem_norm = FeatureNormalizer()
    lab_gem  = gem_norm.fit_transform(lab_gem)
    unl_gem  = gem_norm.transform(unl_gem)
    test_gem = gem_norm.transform(test_gem)

    con_norm = FeatureNormalizer()
    lab_con  = con_norm.fit_transform(lab_con)
    test_con = con_norm.transform(test_con)

    # Datasets
    lab_ds  = GEMDataset(lab_gem, lab_con, lab_mscs, lab_cpi, lab_wmin)
    test_ds = GEMDataset(test_gem, test_con, test_mscs, test_cpi, test_wmin)

    # Unlabeled dataset: only gem features, dummy labels
    dummy = np.zeros(len(unl_gem))
    unl_ds = GEMDataset(unl_gem, unl_gem[:, :CONCORDE_INPUT_DIM],
                         np.zeros((len(unl_gem), 16)), dummy, dummy)

    batch = min(CONCORDE_BATCH_SIZE, len(lab_ds))
    lab_loader  = DataLoader(lab_ds, batch_size=batch, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
    unl_batch   = min(CONCORDE_BATCH_SIZE, len(unl_ds))
    unl_loader  = DataLoader(unl_ds, batch_size=unl_batch, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False,
                             num_workers=4, pin_memory=True)

    n_lab_target = int(all_data[target_name]["n_samples"] * ood_train_ratio)
    n_unlabeled  = len(unl_gem)

    return lab_loader, unl_loader, test_loader, gem_norm, n_lab_target, n_unlabeled


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_gem(model, loader, device):
    model.eval(); P, G = [], []
    for b in loader:
        m = model(b["gem_features"].to(device))
        P.append(compute_cpi_mean(m, b["w_min"].to(device)).cpu())
        G.append(b["cpi_golden"])
    P, G = torch.cat(P), torch.cat(G)
    return compute_mape(P, G)

@torch.no_grad()
def evaluate_concorde(model, loader, device):
    model.eval(); P, G = [], []
    for b in loader:
        P.append(model(b["concord_features"].to(device)).cpu())
        G.append(b["cpi_golden"])
    P, G = torch.cat(P), torch.cat(G)
    return compute_mape(P, G)


# ============================================================
# Training: GEM with mixed batch (two forward passes)
# ============================================================

def train_gem_mix(model, lab_loader, unl_loader, test_loader, device,
                  label="GEM_mix"):
    """
    Two separate forward passes per step:
      Pass 1: labeled batch  → L_task + L_const + L_mono (full supervision)
      Pass 2: unlabeled batch → L_mono only (physics only)
    Gradients from both passes are accumulated before optimizer.step().
    """
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR,
                            weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    best_mape, best_state, pc = float("inf"), None, 0
    step = 0

    for epoch in range(1, CONCORDE_EPOCHS + 1):
        model.train()
        ep_loss, ep_lt, ep_lc, ep_lm, ep_lm_unl, nb = 0, 0, 0, 0, 0, 0

        unl_iter = iter(unl_loader)

        for lab_batch in lab_loader:
            # --- Pass 1: Labeled (full loss) ---
            f_lab = lab_batch["gem_features"].to(device)
            mt = lab_batch["mscs_target"].to(device)
            cg = lab_batch["cpi_golden"].to(device)
            wm = lab_batch["w_min"].to(device)

            optimizer.zero_grad()

            mp_lab = model(f_lab)
            l_t = loss_task(mp_lab, mt)
            l_c = loss_const(mp_lab, wm, cg)
            l_m = loss_mono(mp_lab)
            loss_labeled = l_t + LAMBDA_CONST * l_c + LAMBDA_MONO * l_m
            loss_labeled.backward()

            # --- Pass 2: Unlabeled (L_mono only, separate forward) ---
            try:
                unl_batch = next(unl_iter)
            except StopIteration:
                unl_iter = iter(unl_loader)
                unl_batch = next(unl_iter)

            f_unl = unl_batch["gem_features"].to(device)
            mp_unl = model(f_unl)
            l_m_unl = LAMBDA_MONO * loss_mono(mp_unl)
            l_m_unl.backward()

            # --- Step with accumulated gradients ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            ep_loss += (loss_labeled.item() + l_m_unl.item())
            ep_lt += l_t.item(); ep_lc += l_c.item()
            ep_lm += l_m.item(); ep_lm_unl += loss_mono(mp_unl).item()
            nb += 1

        mape = evaluate_gem(model, test_loader, device)

        if mape < best_mape:
            best_mape, best_state, pc = mape, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PATIENCE * 2:
                print(f"    [{label}] Early stop ep {epoch} (best={best_mape:.2f}%)")
                break

        if epoch % 50 == 0 or epoch == 1:
            print(f"    [{label}] ep {epoch:3d} (step {step}) | "
                  f"task={ep_lt/nb:.4f} const={ep_lc/nb:.4f} "
                  f"mono_lab={ep_lm/nb:.4f} mono_unl={ep_lm_unl/nb:.4f} | "
                  f"MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    print(f"    [{label}] Best: {best_mape:.2f}%")
    return model, best_mape


# ============================================================
# Training: GEM supervised only (baseline, no unlabeled data)
# ============================================================

def train_gem_sup(model, lab_loader, test_loader, device, label="GEM_sup"):
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR,
                            weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    best_mape, best_state, pc = float("inf"), None, 0
    step = 0

    for epoch in range(1, CONCORDE_EPOCHS + 1):
        model.train()
        for b in lab_loader:
            f = b["gem_features"].to(device)
            mt, cg, wm = b["mscs_target"].to(device), b["cpi_golden"].to(device), b["w_min"].to(device)
            mp = model(f)
            loss = loss_task(mp, mt) + LAMBDA_CONST * loss_const(mp, wm, cg) + LAMBDA_MONO * loss_mono(mp)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); step += 1

        mape = evaluate_gem(model, test_loader, device)
        if mape < best_mape:
            best_mape, best_state, pc = mape, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PATIENCE * 2: break
        if epoch % 50 == 0:
            print(f"    [{label}] ep {epoch:3d} (step {step}) | MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    print(f"    [{label}] Best: {best_mape:.2f}%")
    return model, best_mape


# ============================================================
# Training: Concorde (baseline, can't use unlabeled)
# ============================================================

def train_concorde(model, lab_loader, test_loader, device, label="Concorde"):
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR,
                            weight_decay=CONCORDE_WEIGHT_DECAY)
    milestones = [10000, 14000, 18000, 22000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    rl = lambda p,t,e=1e-8: (torch.abs(p-t)/(torch.abs(t)+e)).mean()

    best_mape, best_state, pc = float("inf"), None, 0

    for epoch in range(1, CONCORDE_EPOCHS + 1):
        model.train()
        for b in lab_loader:
            p = model(b["concord_features"].to(device))
            loss = rl(p, b["cpi_golden"].to(device))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

        mape = evaluate_concorde(model, test_loader, device)
        if mape < best_mape:
            best_mape, best_state, pc = mape, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PATIENCE: break
        if epoch % 50 == 0:
            print(f"    [{label}] ep {epoch:3d} | MAPE={mape:.2f}%")

    model.load_state_dict(best_state)
    print(f"    [{label}] Best: {best_mape:.2f}%")
    return model, best_mape


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Exp2 Mix: Semi-supervised OOD")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=0.05)
    args = parser.parse_args()

    ratio_tag = f"{int(args.ratio*100)}pct"
    output_dir = os.path.join(OUT_BASE, f"{ratio_tag}")
    os.makedirs(output_dir, exist_ok=True)

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"Target: {args.target}, Ratio: {args.ratio:.0%}, GPU: {args.gpu}")
    print("Loading...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    assert args.target in all_data, f"Unknown benchmark: {args.target}"

    lab_loader, unl_loader, test_loader, _, n_lab, n_unl = \
        create_ood_mix_split(all_data, args.target, args.ratio)

    print(f"  Labeled target: {n_lab}, Unlabeled target: {n_unl}")
    print(f"  Labeled total: {len(lab_loader.dataset)}, Test: {len(test_loader.dataset)}")

    results = {"target": args.target, "ratio": args.ratio,
               "n_labeled_target": n_lab, "n_unlabeled_target": n_unl}

    # GEM + unlabeled (mixed batch, separate forward passes)
    print(f"\n--- GEM + unlabeled (mixed batch) ---")
    set_seed(SEED)
    m = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(device)
    _, mape = train_gem_mix(m, lab_loader, unl_loader, test_loader, device,
                            label=f"GEM_mix|{args.target}")
    results["gem_mix"] = mape

    # Summary
    print(f"\n{'='*60}")
    print(f"  {args.target} (ratio={args.ratio:.0%})")
    print(f"  GEM + unlabeled:   {results['gem_mix']:.2f}%")
    print(f"{'='*60}")

    path = os.path.join(output_dir, f"mix_{args.target}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
