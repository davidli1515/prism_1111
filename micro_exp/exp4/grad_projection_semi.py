"""
Gradient Projection Decomposition (PCGrad-style)

Decomposes ∇L_mono and ∇L_const into components parallel and
orthogonal to ∇L_task:

    ∇L_x = proj(∇L_x → ∇L_task) + ∇L_x_⊥

Reports novelty_ratio = ||∇L_x_⊥|| / ||∇L_x||
  - Close to 1 → L_x provides new information beyond L_task
  - Close to 0 → L_x is redundant with L_task

Runs under two settings:
  --mode full   : 100% labeled (full supervision)
  --mode semi   : 5% labeled (semi-supervised)

If L_mono novelty_ratio is low under full but rises under semi,
it confirms: L_mono is redundant when L_task has full coverage,
but becomes complementary when L_task coverage shrinks.

Usage:
    python grad_projection_semi.py --mode full --gpu 0
    python grad_projection_semi.py --mode semi --gpu 1 --ratio 0.05

    # Parallel:
    python grad_projection_semi.py --mode full --gpu 0 > logs/proj_full.log 2>&1 &
    python grad_projection_semi.py --mode semi --gpu 1 --ratio 0.05 > logs/proj_semi.log 2>&1 &

Output: grad_projection_{mode}_{ratio}pct.json

Plot:  python plot_grad_projection.py
"""
import os, sys, json, hashlib, argparse
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")

from config import (
    GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES, SEED, TRAIN_RATIO,
    BASE_INDICES,
)
from model import GEMModel, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, FeatureNormalizer, GEMDataset, create_exp1_dataloaders


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def stable_hash(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)


def loss_const_self(mscs_pred, w_min, eps=1e-8):
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi_per_stage = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    cpi_mean = cpi_per_stage.mean(dim=1, keepdim=True)
    diff = cpi_per_stage - cpi_mean
    return (diff ** 2).mean()


def get_grad_vec(loss, model):
    grads = torch.autograd.grad(loss, model.parameters(),
                                retain_graph=True, allow_unused=True)
    return torch.cat([g.flatten() if g is not None
                      else torch.zeros(p.numel(), device=p.device)
                      for g, p in zip(grads, model.parameters())])


def project_and_decompose(g_x, g_task):
    """
    Decompose g_x into component parallel to g_task and orthogonal remainder.

    proj = (g_x · g_task) / ||g_task||² × g_task
    g_x_perp = g_x - proj
    novelty_ratio = ||g_x_perp|| / ||g_x||

    Returns: novelty_ratio, ||proj||, ||g_x_perp||, ||g_x||
    """
    g_x_norm = g_x.norm().item()
    if g_x_norm < 1e-12:
        return 0.0, 0.0, 0.0, 0.0

    g_task_sq = g_task.dot(g_task)
    if g_task_sq < 1e-12:
        return 1.0, 0.0, g_x_norm, g_x_norm

    coeff = g_x.dot(g_task) / g_task_sq
    proj = coeff * g_task
    g_x_perp = g_x - proj

    proj_norm = proj.norm().item()
    perp_norm = g_x_perp.norm().item()
    novelty = perp_norm / g_x_norm

    return novelty, proj_norm, perp_norm, g_x_norm


def create_semi_dataloader(all_data, labeled_ratio, batch_size=BATCH_SIZE, seed=SEED):
    """Create dataloader with has_label flag for semi-supervised setting."""
    from torch.utils.data import Dataset, DataLoader

    class SemiDS(Dataset):
        def __init__(self, gf, cf, mt, cg, wm, hl):
            self.gf = torch.from_numpy(gf); self.cf = torch.from_numpy(cf)
            self.mt = torch.from_numpy(mt); self.cg = torch.from_numpy(cg)
            self.wm = torch.from_numpy(wm); self.hl = torch.from_numpy(hl.astype(np.float32))
        def __len__(self): return self.gf.shape[0]
        def __getitem__(self, i):
            return {"gem_features": self.gf[i], "concord_features": self.cf[i],
                    "mscs_target": self.mt[i], "cpi_golden": self.cg[i],
                    "w_min": self.wm[i], "has_label": self.hl[i]}

    rng = np.random.RandomState(seed)
    tr_gem, tr_con, tr_mscs, tr_cpi, tr_wmin, tr_hl = [], [], [], [], [], []

    for name, data in sorted(all_data.items()):
        n = data["n_samples"]
        idx = rng.permutation(n)
        sp = int(n * TRAIN_RATIO)
        tr_idx = idx[:sp]

        n_tr = len(tr_idx)
        n_lab = max(1, int(n_tr * labeled_ratio))
        lrng = np.random.RandomState(seed + stable_hash(name) + int(round(labeled_ratio*1000)))
        lp = lrng.permutation(n_tr)
        hl = np.zeros(n_tr, dtype=np.float32)
        hl[lp[:n_lab]] = 1.0

        tr_gem.append(data["gem_features"][tr_idx])
        tr_con.append(data["concord_features"][tr_idx])
        tr_mscs.append(data["mscs_target"][tr_idx])
        tr_cpi.append(data["cpi_golden"][tr_idx])
        tr_wmin.append(data["w_min"][tr_idx])
        tr_hl.append(hl)

    tr_gem = np.concatenate(tr_gem); tr_con = np.concatenate(tr_con)
    tr_mscs = np.concatenate(tr_mscs); tr_cpi = np.concatenate(tr_cpi)
    tr_wmin = np.concatenate(tr_wmin); tr_hl = np.concatenate(tr_hl)

    gn = FeatureNormalizer(); tr_gem = gn.fit_transform(tr_gem)
    cn = FeatureNormalizer(); tr_con = cn.fit_transform(tr_con)

    n_lab = int(tr_hl.sum())
    print(f"  Labeled: {n_lab:,} / {len(tr_hl):,} ({n_lab/len(tr_hl)*100:.1f}%)")

    return DataLoader(SemiDS(tr_gem, tr_con, tr_mscs, tr_cpi, tr_wmin, tr_hl),
                      batch_size=batch_size, shuffle=True, num_workers=4,
                      pin_memory=True, drop_last=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", type=str, default="semi", choices=["full", "semi"])
    parser.add_argument("--ratio", type=float, default=0.05,
                        help="Labeled ratio (only used when --mode semi)")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    set_seed(SEED)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    is_full = (args.mode == "full")
    ratio = 1.0 if is_full else args.ratio
    ratio_tag = "100pct" if is_full else f"{int(round(ratio*100))}pct"

    print(f"Gradient Projection Analysis — mode={args.mode}, labeled={ratio*100:.0f}%")
    print(f"Device: {dev}")

    print("\nLoading data...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)

    if is_full:
        # Full supervision: use exp1 dataloaders (100% labeled)
        (train_loader, _, _, _, _) = create_exp1_dataloaders(all_data)
    else:
        train_loader = create_semi_dataloader(all_data, ratio)

    model = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    print(f"\nTraining {args.epochs} epochs with projection measurement...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        metrics = defaultdict(list)

        for i, batch in enumerate(train_loader):
            feat     = batch["gem_features"].to(dev)
            mscs_tgt = batch["mscs_target"].to(dev)
            cpi_gold = batch["cpi_golden"].to(dev)
            w_min    = batch["w_min"].to(dev)

            if is_full:
                # Full mode: all samples are labeled
                mscs_pred = model(feat)
                lt = loss_task(mscs_pred, mscs_tgt)
                lc = loss_const(mscs_pred, w_min, cpi_gold)
                lm = loss_mono(mscs_pred)

                if i % 20 == 0:
                    g_task  = get_grad_vec(lt, model)
                    g_const = get_grad_vec(lc, model)
                    g_mono  = get_grad_vec(lm, model)

                    nov_const, proj_c, perp_c, norm_c = project_and_decompose(g_const, g_task)
                    nov_mono,  proj_m, perp_m, norm_m = project_and_decompose(g_mono, g_task)

                    metrics["novelty_const"].append(nov_const)
                    metrics["novelty_mono"].append(nov_mono)
                    metrics["proj_norm_const"].append(proj_c)
                    metrics["perp_norm_const"].append(perp_c)
                    metrics["proj_norm_mono"].append(proj_m)
                    metrics["perp_norm_mono"].append(perp_m)
                    metrics["norm_task"].append(g_task.norm().item())
                    metrics["norm_const"].append(norm_c)
                    metrics["norm_mono"].append(norm_m)

                # Training step
                loss = lt + LAMBDA_CONST * lc + LAMBDA_MONO * lm

            else:
                # Semi mode: split labeled/unlabeled
                has_label = batch["has_label"].to(dev)
                mscs_pred = model(feat)

                lab_mask = has_label > 0.5
                n_lab = lab_mask.sum().item()
                n_unl = (~lab_mask).sum().item()

                if i % 20 == 0 and n_lab > 1:
                    l_pred = mscs_pred[lab_mask]
                    l_tgt  = mscs_tgt[lab_mask]
                    l_wmin = w_min[lab_mask]
                    l_cpi  = cpi_gold[lab_mask]

                    lt = loss_task(l_pred, l_tgt)
                    lc = loss_const(l_pred, l_wmin, l_cpi)
                    lm = loss_mono(l_pred)

                    g_task  = get_grad_vec(lt, model)
                    g_const = get_grad_vec(lc, model)
                    g_mono  = get_grad_vec(lm, model)

                    nov_const, proj_c, perp_c, norm_c = project_and_decompose(g_const, g_task)
                    nov_mono,  proj_m, perp_m, norm_m = project_and_decompose(g_mono, g_task)

                    metrics["novelty_const"].append(nov_const)
                    metrics["novelty_mono"].append(nov_mono)
                    metrics["proj_norm_const"].append(proj_c)
                    metrics["perp_norm_const"].append(perp_c)
                    metrics["proj_norm_mono"].append(proj_m)
                    metrics["perp_norm_mono"].append(perp_m)
                    metrics["norm_task"].append(g_task.norm().item())
                    metrics["norm_const"].append(norm_c)
                    metrics["norm_mono"].append(norm_m)

                # Training step (semi-supervised loss)
                loss = torch.tensor(0.0, device=dev)
                if n_lab > 0:
                    l_pred = mscs_pred[lab_mask]
                    loss += loss_task(l_pred, mscs_tgt[lab_mask]) \
                          + LAMBDA_CONST * loss_const(l_pred, w_min[lab_mask], cpi_gold[lab_mask]) \
                          + LAMBDA_MONO * loss_mono(l_pred)
                if n_unl > 0:
                    u_pred = mscs_pred[~lab_mask]
                    loss += LAMBDA_CONST * loss_const_self(u_pred, w_min[~lab_mask]) \
                          + LAMBDA_MONO * loss_mono(u_pred)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        rec = {"epoch": epoch}
        for k, v in metrics.items():
            rec[k] = float(np.mean(v)) if v else 0.0
        history.append(rec)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"novelty_const={rec.get('novelty_const',0):.4f}  "
                  f"novelty_mono={rec.get('novelty_mono',0):.4f} | "
                  f"||∇task||={rec.get('norm_task',0):.4f}  "
                  f"||∇const||={rec.get('norm_const',0):.4f}  "
                  f"||∇mono||={rec.get('norm_mono',0):.6f}")

    out_file = f"grad_projection_{args.mode}_{ratio_tag}.json"
    with open(out_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved to {out_file}")

    # Summary
    early = history[:10]
    late  = history[-10:]
    print(f"\n{'='*60}")
    print(f"Summary — {args.mode} ({ratio*100:.0f}% labeled)")
    print(f"{'='*60}")
    print(f"  {'':20} {'Early (1-10)':>14} {'Late':>14}")
    print(f"  {'-'*50}")
    print(f"  {'novelty_const':<20} "
          f"{np.mean([r.get('novelty_const',0) for r in early]):>13.4f} "
          f"{np.mean([r.get('novelty_const',0) for r in late]):>13.4f}")
    print(f"  {'novelty_mono':<20} "
          f"{np.mean([r.get('novelty_mono',0) for r in early]):>13.4f} "
          f"{np.mean([r.get('novelty_mono',0) for r in late]):>13.4f}")
    print(f"  {'||∇mono||':<20} "
          f"{np.mean([r.get('norm_mono',0) for r in early]):>13.6f} "
          f"{np.mean([r.get('norm_mono',0) for r in late]):>13.6f}")
    print(f"{'='*60}")
    print(f"\nInterpretation:")
    print(f"  novelty → 1: provides new information beyond L_task")
    print(f"  novelty → 0: redundant with L_task")


if __name__ == "__main__":
    main()
