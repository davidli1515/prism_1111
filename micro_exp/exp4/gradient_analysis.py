"""
Gradient Projection Analysis: Novelty of Physics Constraints vs L_task

For each physics loss (L_const, L_mono), compute:
  novelty_ratio = ||g_perp|| / ||g||
where g_perp is the component of the physics gradient orthogonal to L_task's gradient.

novelty ~1.0: physics loss provides independent/competing signal
novelty ~0.0: physics loss aligns with and amplifies L_task

Outputs a multi-panel figure comparing novelty across label ratios.

Usage:
    python gradient_analysis.py --gpu 0
"""

import os, sys, time, json, hashlib
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")

from config import (
    GEM_INPUT_DIM, DROPOUT, BENCHMARK_NAMES,
    SEED, TRAIN_RATIO, BASE_INDICES,
    BATCH_SIZE,
)
from model import GEMModel, compute_cpi_mean, loss_const, loss_mono
from dataset import load_all_benchmarks, FeatureNormalizer

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.3
LAMBDA_TASK = 0.1
LAMBDA_CONST = 1.0
LAMBDA_MONO = 0.01
NUM_EPOCHS = 50          # Only need ~50 epochs to see trends
LOG_EVERY = 1            # Log every epoch

RATIOS = [0.05, 0.15, 0.50, 1.0]

DATASET_PATHS = {
    "gem":     "/data/micro_traindata_packed/gem",
    "concord": "/data/micro_traindata_packed/concord",
}

SAVE_DIR = "/path/to/gem5/micro_exp/exp4/figures_final"

# ============================================================
# Helpers
# ============================================================

def set_seed(seed=SEED):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); torch.backends.cudnn.deterministic = True

def stable_hash(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)

def loss_task(mscs_pred, mscs_target):
    return ((mscs_pred - mscs_target) ** 2).mean()

def loss_const_self(mscs_pred, w_min, eps=1e-8):
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi_per_stage = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    cpi_mean = cpi_per_stage.mean(dim=1, keepdim=True)
    diff = cpi_per_stage - cpi_mean
    return (diff ** 2).mean()


# ============================================================
# Dataloader (reuse from train_exp4_all)
# ============================================================

def create_dataloaders(all_data, labeled_ratio, batch_size=BATCH_SIZE, seed=SEED):
    rng_split = np.random.RandomState(seed)
    train_gem, train_mscs, train_cpi, train_wmin, train_has_label = [], [], [], [], []
    test_gem, test_mscs, test_cpi, test_wmin = [], [], [], []
    per_bm_test = {}
    offset = 0

    for name in sorted(all_data.keys()):
        data = all_data[name]
        n = data["n_samples"]
        indices = rng_split.permutation(n)
        split = int(n * TRAIN_RATIO)
        tr_idx, te_idx = indices[:split], indices[split:]

        train_gem.append(data["gem_features"][tr_idx])
        train_mscs.append(data["mscs_target"][tr_idx])
        train_cpi.append(data["cpi_golden"][tr_idx])
        train_wmin.append(data["w_min"][tr_idx])

        # Labeled/unlabeled split
        n_train = len(tr_idx)
        n_labeled = max(1, int(n_train * labeled_ratio))
        label_rng = np.random.RandomState(
            seed + stable_hash(name) + int(round(labeled_ratio * 1000)))
        label_perm = label_rng.permutation(n_train)
        has_label = np.zeros(n_train, dtype=np.float32)
        has_label[label_perm[:n_labeled]] = 1.0
        train_has_label.append(has_label)

        test_gem.append(data["gem_features"][te_idx])
        test_mscs.append(data["mscs_target"][te_idx])
        test_cpi.append(data["cpi_golden"][te_idx])
        test_wmin.append(data["w_min"][te_idx])
        n_te = len(te_idx)
        per_bm_test[name] = (offset, offset + n_te)
        offset += n_te

    train_gem = np.concatenate(train_gem)
    train_mscs = np.concatenate(train_mscs)
    train_cpi = np.concatenate(train_cpi)
    train_wmin = np.concatenate(train_wmin)
    train_has_label = np.concatenate(train_has_label)

    gem_norm = FeatureNormalizer()
    train_gem = gem_norm.fit_transform(train_gem)
    test_gem = gem_norm.transform(np.concatenate(test_gem))

    class SemiDataset(torch.utils.data.Dataset):
        def __init__(self, gem, mscs, cpi, wmin, has_label):
            self.gem = torch.from_numpy(gem)
            self.mscs = torch.from_numpy(mscs)
            self.cpi = torch.from_numpy(cpi)
            self.wmin = torch.from_numpy(wmin)
            self.has_label = torch.from_numpy(has_label)
        def __len__(self):
            return len(self.gem)
        def __getitem__(self, idx):
            return {"gem_features": self.gem[idx], "mscs_target": self.mscs[idx],
                    "cpi_golden": self.cpi[idx], "w_min": self.wmin[idx],
                    "has_label": self.has_label[idx]}

    train_ds = SemiDataset(train_gem, train_mscs, train_cpi, train_wmin, train_has_label)
    actual_bs = min(batch_size, len(train_ds))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=actual_bs, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    return train_loader


# ============================================================
# Gradient novelty computation
# ============================================================

def compute_novelty(g_phys_list, g_task_list):
    """Compute novelty ratio: ||g_perp|| / ||g|| for physics gradient relative to g_task."""
    # Flatten all params into single vectors
    g_phys = torch.cat([g.flatten() for g in g_phys_list])
    g_task = torch.cat([g.flatten() for g in g_task_list])

    # Project g_phys onto g_task
    dot = (g_phys * g_task).sum()
    task_norm_sq = (g_task * g_task).sum() + 1e-12
    g_parallel = (dot / task_norm_sq) * g_task
    g_perp = g_phys - g_parallel

    phys_norm = g_phys.norm() + 1e-12
    novelty = g_perp.norm() / phys_norm

    return novelty.item()


def get_grads(model, loss):
    """Compute gradients of loss w.r.t. model params, return as list."""
    model.zero_grad()
    loss.backward(retain_graph=True)
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.clone())
        else:
            grads.append(torch.zeros_like(p))
    return grads


# ============================================================
# Main analysis loop
# ============================================================

def run_analysis(train_loader, device, labeled_ratio, num_epochs=NUM_EPOCHS):
    """Train model and log gradient novelty per epoch."""
    set_seed(SEED)
    model = GEMModel(GEM_INPUT_DIM, [512, 256], DROPOUT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    const_novelties = []
    mono_novelties = []
    mono_vs_const_unlabeled = []
    mono_vs_const_labeled = []  # NEW: L_mono vs L_const on labeled

    print(f"\n  Running {labeled_ratio*100:.0f}% labeled, {num_epochs} epochs...")
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_const_nov = []
        epoch_mono_nov = []
        epoch_mono_vs_const_unl = []
        epoch_mono_vs_const_lab = []

        for batch in train_loader:
            gem_feat  = batch["gem_features"].to(device)
            mscs_tgt  = batch["mscs_target"].to(device)
            cpi_gold  = batch["cpi_golden"].to(device)
            w_min     = batch["w_min"].to(device)
            has_label = batch["has_label"].to(device)

            mscs_pred = model(gem_feat)

            # === Labeled: L_mono vs L_task (as before) ===
            labeled_mask = has_label > 0.5
            n_lab = labeled_mask.sum().item()

            if n_lab >= 2:
                l_pred = mscs_pred[labeled_mask]
                l_tgt  = mscs_tgt[labeled_mask]
                l_wmin = w_min[labeled_mask]
                l_cpi  = cpi_gold[labeled_mask]

                l_task_val  = loss_task(l_pred, l_tgt)
                l_const_val = loss_const(l_pred, l_wmin, l_cpi)
                l_mono_val  = loss_mono(l_pred)

                g_task  = get_grads(model, l_task_val)
                g_const = get_grads(model, l_const_val)
                g_mono  = get_grads(model, l_mono_val)

                epoch_const_nov.append(compute_novelty(g_const, g_task))
                epoch_mono_nov.append(compute_novelty(g_mono, g_task))
                epoch_mono_vs_const_lab.append(compute_novelty(g_mono, g_const))

            # === Unlabeled: L_mono vs L_const_self ===
            unlabeled_mask = ~labeled_mask
            n_unl = unlabeled_mask.sum().item()

            if n_unl >= 2:
                u_pred = mscs_pred[unlabeled_mask]
                u_wmin = w_min[unlabeled_mask]

                u_const_self_val = loss_const_self(u_pred, u_wmin)
                u_mono_val = loss_mono(u_pred)

                g_const_self = get_grads(model, u_const_self_val)
                g_mono_unl   = get_grads(model, u_mono_val)

                epoch_mono_vs_const_unl.append(
                    compute_novelty(g_mono_unl, g_const_self))

            # Normal training step (combined loss)
            total_loss = torch.tensor(0.0, device=gem_feat.device)
            if n_lab >= 2:
                total_loss = (LAMBDA_TASK * l_task_val
                             + LAMBDA_CONST * l_const_val
                             + LAMBDA_MONO * l_mono_val)

            if n_unl > 0:
                u_pred2 = mscs_pred[unlabeled_mask]
                u_wmin2 = w_min[unlabeled_mask]
                u_cs = loss_const_self(u_pred2, u_wmin2)
                u_mo = loss_mono(u_pred2)
                total_loss = total_loss + LAMBDA_CONST * u_cs + LAMBDA_MONO * u_mo

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Epoch averages
        if epoch_const_nov:
            const_novelties.append(np.mean(epoch_const_nov))
            mono_novelties.append(np.mean(epoch_mono_nov))
            mono_vs_const_labeled.append(np.mean(epoch_mono_vs_const_lab))
        else:
            const_novelties.append(float('nan'))
            mono_novelties.append(float('nan'))
            mono_vs_const_labeled.append(float('nan'))

        if epoch_mono_vs_const_unl:
            mono_vs_const_unlabeled.append(np.mean(epoch_mono_vs_const_unl))
        else:
            mono_vs_const_unlabeled.append(float('nan'))

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            unl_str = f" | mono/const(U)={mono_vs_const_unlabeled[-1]:.3f}" if epoch_mono_vs_const_unl else ""
            lab_str = f" | mono/const(L)={mono_vs_const_labeled[-1]:.3f}" if epoch_mono_vs_const_lab else ""
            print(f"    Epoch {epoch:3d} | "
                  f"const/task={const_novelties[-1]:.3f} | "
                  f"mono/task={mono_novelties[-1]:.3f}"
                  f"{lab_str}{unl_str} | {elapsed:.0f}s")

    return const_novelties, mono_novelties, mono_vs_const_unlabeled, mono_vs_const_labeled


# ============================================================
# Plotting
# ============================================================

def plot_results(all_results, save_dir):
    """Single panel: L_const and L_mono novelty vs L_task."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 600,
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    # Pick any ratio (all look the same) — prefer 5% or lowest available
    ratio = min(all_results.keys())
    data = all_results[ratio]

    fig, ax = plt.subplots(figsize=(6, 4))

    epochs = np.arange(1, len(data["const"]) + 1)

    ax.plot(epochs, data["const"], color="#1565C0", marker="o",
            markersize=3, linewidth=1.5, label=r"$\mathcal{L}_{const}$ novelty")
    ax.plot(epochs, data["mono"], color="#C62828", marker="o",
            markersize=3, linewidth=1.5, label=r"$\mathcal{L}_{mono}$ novelty")

    ax.set_title("Gradient Projection: Novelty vs $\\mathcal{L}_{task}$",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Novelty Ratio  $\||\nabla L_\perp\|| / \||\nabla L\||$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, len(epochs) + 1)

    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.6)
    ax.xaxis.grid(False)

    legend = ax.legend(loc="lower right", frameon=True, edgecolor='black',
                      facecolor='white', fancybox=False, framealpha=1.0)
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    path = os.path.join(save_dir, "gradient_novelty_analysis.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    pdf_path = path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"\n  Saved {path}")
    print(f"  Saved {pdf_path}")


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--ratios", type=float, nargs="+", default=RATIOS)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading data...")
    all_data = load_all_benchmarks(
        DATASET_PATHS["gem"], DATASET_PATHS["concord"], BENCHMARK_NAMES)

    all_results = {}

    for ratio in args.ratios:
        print(f"\n{'='*60}")
        print(f"Ratio: {ratio*100:.0f}% labeled")
        print(f"{'='*60}")

        train_loader = create_dataloaders(all_data, ratio)
        const_nov, mono_nov, mono_vs_const_unl, mono_vs_const_lab = run_analysis(
            train_loader, device, ratio, num_epochs=args.epochs)

        all_results[ratio] = {
            "const": const_nov,
            "mono": mono_nov,
            "mono_vs_const_unl": mono_vs_const_unl,
            "mono_vs_const_lab": mono_vs_const_lab,
        }

        # Save raw data
        json_path = os.path.join(SAVE_DIR, f"gradient_novelty_{int(ratio*100)}pct.json")
        with open(json_path, "w") as f:
            json.dump({"ratio": ratio, "const": const_nov, "mono": mono_nov,
                       "mono_vs_const_unl": mono_vs_const_unl,
                       "mono_vs_const_lab": mono_vs_const_lab}, f)
        print(f"  Raw data saved: {json_path}")

    # Plot
    print("\nPlotting...")
    plot_results(all_results, SAVE_DIR)
    print("\nDone!")


if __name__ == "__main__":
    main()