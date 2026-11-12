"""
Gradient Conflict Analysis for Semi-Supervised Setting (5% labeled)

Measures cosine similarity between gradients of:
  - Labeled:   ∇L_task, ∇L_const, ∇L_mono
  - Unlabeled: ∇L_const_self, ∇L_mono

This reveals how physics constraints interact with task loss
when labeled data is scarce and unlabeled data dominates.

Usage: python grad_conflict_semi.py --gpu 0
       python grad_conflict_semi.py --gpu 0 --ratio 0.20

Output: grad_conflict_semi_{ratio}pct.json
"""
import os, sys, json, hashlib, argparse
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")

from config import (
    GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES, SEED, TRAIN_RATIO,
    BASE_INDICES,
)
from model import GEMModel, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, FeatureNormalizer, GEMDataset


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
    """Get flattened gradient vector for a loss."""
    grads = torch.autograd.grad(loss, model.parameters(),
                                retain_graph=True, allow_unused=True)
    return torch.cat([g.flatten() if g is not None
                      else torch.zeros(p.numel(), device=p.device)
                      for g, p in zip(grads, model.parameters())])


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def create_semi_dataloader(all_data, labeled_ratio, batch_size=BATCH_SIZE, seed=SEED):
    """Simplified version — returns single dataloader with has_label flag."""
    from torch.utils.data import Dataset, DataLoader

    class SemiDS(Dataset):
        def __init__(self, gf, cf, mt, cg, wm, hl):
            self.gf = torch.from_numpy(gf)
            self.cf = torch.from_numpy(cf)
            self.mt = torch.from_numpy(mt)
            self.cg = torch.from_numpy(cg)
            self.wm = torch.from_numpy(wm)
            self.hl = torch.from_numpy(hl.astype(np.float32))
        def __len__(self): return self.gf.shape[0]
        def __getitem__(self, i):
            return {"gem_features": self.gf[i], "concord_features": self.cf[i],
                    "mscs_target": self.mt[i], "cpi_golden": self.cg[i],
                    "w_min": self.wm[i], "has_label": self.hl[i]}

    rng = np.random.RandomState(seed)
    tr_gem, tr_con, tr_mscs, tr_cpi, tr_wmin, tr_hl = [], [], [], [], [], []
    te_gem, te_con, te_mscs, te_cpi, te_wmin = [], [], [], [], []

    for name, data in sorted(all_data.items()):
        n = data["n_samples"]
        idx = rng.permutation(n)
        sp = int(n * TRAIN_RATIO)
        tr_idx, te_idx = idx[:sp], idx[sp:]

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

        te_gem.append(data["gem_features"][te_idx])
        te_con.append(data["concord_features"][te_idx])
        te_mscs.append(data["mscs_target"][te_idx])
        te_cpi.append(data["cpi_golden"][te_idx])
        te_wmin.append(data["w_min"][te_idx])

    tr_gem = np.concatenate(tr_gem); tr_con = np.concatenate(tr_con)
    tr_mscs = np.concatenate(tr_mscs); tr_cpi = np.concatenate(tr_cpi)
    tr_wmin = np.concatenate(tr_wmin); tr_hl = np.concatenate(tr_hl)
    te_gem = np.concatenate(te_gem); te_con = np.concatenate(te_con)
    te_mscs = np.concatenate(te_mscs); te_cpi = np.concatenate(te_cpi)
    te_wmin = np.concatenate(te_wmin)

    gn = FeatureNormalizer(); tr_gem = gn.fit_transform(tr_gem); te_gem = gn.transform(te_gem)
    cn = FeatureNormalizer(); tr_con = cn.fit_transform(tr_con); te_con = cn.transform(te_con)

    n_lab = int(tr_hl.sum())
    print(f"  Labeled: {n_lab:,} / {len(tr_hl):,} ({n_lab/len(tr_hl)*100:.1f}%)")

    train_loader = DataLoader(SemiDS(tr_gem, tr_con, tr_mscs, tr_cpi, tr_wmin, tr_hl),
                              batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    return train_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    set_seed(SEED)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ratio_tag = f"{int(round(args.ratio*100))}pct"

    print(f"Gradient conflict analysis — {args.ratio*100:.0f}% labeled")
    print("Loading data...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    train_loader = create_semi_dataloader(all_data, args.ratio)

    model = GEMModel(GEM_INPUT_DIM, GEM_HIDDEN_DIMS, DROPOUT).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    print(f"\nTraining {args.epochs} epochs with gradient measurement...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        metrics = defaultdict(list)

        for i, batch in enumerate(train_loader):
            feat     = batch["gem_features"].to(dev)
            mscs_tgt = batch["mscs_target"].to(dev)
            cpi_gold = batch["cpi_golden"].to(dev)
            w_min    = batch["w_min"].to(dev)
            has_label = batch["has_label"].to(dev)

            mscs_pred = model(feat)

            lab_mask = has_label > 0.5
            unl_mask = ~lab_mask
            n_lab = lab_mask.sum().item()
            n_unl = unl_mask.sum().item()

            # Measure every 20 batches
            if i % 20 == 0:
                # --- Labeled gradients ---
                if n_lab > 1:
                    l_pred = mscs_pred[lab_mask]
                    l_tgt  = mscs_tgt[lab_mask]
                    l_wmin = w_min[lab_mask]
                    l_cpi  = cpi_gold[lab_mask]

                    lt = loss_task(l_pred, l_tgt)
                    lc = loss_const(l_pred, l_wmin, l_cpi)
                    lm = loss_mono(l_pred)

                    g_task  = get_grad_vec(lt, model)
                    g_const = get_grad_vec(lc, model)
                    g_mono_lab = get_grad_vec(lm, model)

                    metrics["cos_task_const_lab"].append(cos_sim(g_task, g_const))
                    metrics["cos_task_mono_lab"].append(cos_sim(g_task, g_mono_lab))
                    metrics["cos_const_mono_lab"].append(cos_sim(g_const, g_mono_lab))
                    metrics["norm_task"].append(g_task.norm().item())
                    metrics["norm_const_lab"].append(g_const.norm().item())
                    metrics["norm_mono_lab"].append(g_mono_lab.norm().item())

                # --- Unlabeled gradients ---
                if n_unl > 1:
                    u_pred = mscs_pred[unl_mask]
                    u_wmin = w_min[unl_mask]

                    lcs = loss_const_self(u_pred, u_wmin)
                    lmu = loss_mono(u_pred)

                    g_cself = get_grad_vec(lcs, model)
                    g_mono_unl = get_grad_vec(lmu, model)

                    metrics["cos_cself_mono_unl"].append(cos_sim(g_cself, g_mono_unl))
                    metrics["norm_const_self"].append(g_cself.norm().item())
                    metrics["norm_mono_unl"].append(g_mono_unl.norm().item())

                # --- Cross: labeled task vs unlabeled physics ---
                if n_lab > 1 and n_unl > 1:
                    metrics["cos_task_cself"].append(cos_sim(g_task, g_cself))
                    metrics["cos_task_mono_unl"].append(cos_sim(g_task, g_mono_unl))

            # Normal training step (full semi-supervised loss)
            total_loss = torch.tensor(0.0, device=dev)
            if n_lab > 0:
                l_pred = mscs_pred[lab_mask]
                total_loss += loss_task(l_pred, mscs_tgt[lab_mask]) \
                            + LAMBDA_CONST * loss_const(l_pred, w_min[lab_mask], cpi_gold[lab_mask]) \
                            + LAMBDA_MONO * loss_mono(l_pred)
            if n_unl > 0:
                u_pred = mscs_pred[unl_mask]
                total_loss += LAMBDA_CONST * loss_const_self(u_pred, w_min[unl_mask]) \
                            + LAMBDA_MONO * loss_mono(u_pred)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        rec = {"epoch": epoch}
        for k, v in metrics.items():
            rec[k] = float(np.mean(v)) if v else 0.0
        history.append(rec)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"cos(task,const_lab)={rec.get('cos_task_const_lab',0):+.4f}  "
                  f"cos(task,mono_lab)={rec.get('cos_task_mono_lab',0):+.4f}  "
                  f"cos(task,cself)={rec.get('cos_task_cself',0):+.4f}  "
                  f"cos(task,mono_unl)={rec.get('cos_task_mono_unl',0):+.4f} | "
                  f"||∇task||={rec.get('norm_task',0):.4f}  "
                  f"||∇cself||={rec.get('norm_const_self',0):.4f}  "
                  f"||∇mono_unl||={rec.get('norm_mono_unl',0):.6f}")

    out_file = f"grad_conflict_semi_{ratio_tag}.json"
    with open(out_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved to {out_file}")

    # Summary
    early = history[:10]
    late  = history[-10:]
    print(f"\n--- Summary ({args.ratio*100:.0f}% labeled) ---")
    print(f"Early (ep 1-10):")
    print(f"  cos(task, const_lab)  = {np.mean([r.get('cos_task_const_lab',0) for r in early]):+.4f}")
    print(f"  cos(task, mono_lab)   = {np.mean([r.get('cos_task_mono_lab',0) for r in early]):+.4f}")
    print(f"  cos(const, mono_lab)  = {np.mean([r.get('cos_const_mono_lab',0) for r in early]):+.4f}")
    print(f"  cos(cself, mono_unl)  = {np.mean([r.get('cos_cself_mono_unl',0) for r in early]):+.4f}")
    print(f"Late (ep {args.epochs-9}-{args.epochs}):")
    print(f"  cos(task, const_lab)  = {np.mean([r.get('cos_task_const_lab',0) for r in late]):+.4f}")
    print(f"  cos(task, mono_lab)   = {np.mean([r.get('cos_task_mono_lab',0) for r in late]):+.4f}")
    print(f"  cos(const, mono_lab)  = {np.mean([r.get('cos_const_mono_lab',0) for r in late]):+.4f}")
    print(f"  cos(cself, mono_unl)  = {np.mean([r.get('cos_cself_mono_unl',0) for r in late]):+.4f}")
    print(f"\n||∇mono_unl|| early={np.mean([r.get('norm_mono_unl',0) for r in early]):.6f}  "
          f"late={np.mean([r.get('norm_mono_unl',0) for r in late]):.6f}")


if __name__ == "__main__":
    main()