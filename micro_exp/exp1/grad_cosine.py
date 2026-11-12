#!/usr/bin/env python3
"""
Measure cosine similarity between gradient of L_task and L_mono (and L_const).
The SIGN is decisive:
  cos < 0  →  gradients point in opposing directions (conflict)
  cos > 0  →  gradients aligned (cooperative)
  cos ≈ 0  →  orthogonal (independent)

Measured across training epochs at different label ratios.

Usage: python3 grad_cosine.py
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")

from config import (GEM_INPUT_DIM, DROPOUT, SEED, CONCORDE_LR, CONCORDE_WEIGHT_DECAY,
                    CONCORDE_BATCH_SIZE, CONCORDE_EPOCHS, LAMBDA_CONST, LAMBDA_MONO,
                    BENCHMARK_NAMES)
from model import GEMModel, compute_cpi_mean, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, create_exp1_dataloaders
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 11,
    'figure.dpi': 600, 'axes.linewidth': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
})

# ============================================================
# Config
# ============================================================
GEM_DIR = "/data/micro_traindata_packed/gem"
CON_DIR = "/data/micro_traindata_packed/concord"
HIDDEN = [512, 256]
N_EPOCHS = 100          # train for this many epochs
N_MEASURE_BATCHES = 5   # average cos over this many batches

# MODE: "full" = train with L_task+L_const+L_mono (lambda_task=0.1)
#       "mono" = train with L_task+L_mono only (lambda_task=1.0) — the gem_mono scenario
MODE = "mono"

OUTPUT_JSON = f"/path/to/gem5/micro_exp/exp1/grad_cosine_{MODE}.json"
OUTPUT_FIG = f"/path/to/gem5/micro_exp/exp1/grad_cosine_{MODE}.png"
# ============================================================


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def get_flat_grad(model, loss_val):
    """Compute flattened gradient vector of loss w.r.t. all model params."""
    model.zero_grad()
    loss_val.backward(retain_graph=True)
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten().clone())
        else:
            grads.append(torch.zeros_like(p).flatten())
    return torch.cat(grads)


def cosine(g1, g2):
    return (torch.dot(g1, g2) / (g1.norm() * g2.norm() + 1e-12)).item()


@torch.no_grad()
def eval_mape(model, loader, device):
    model.eval(); P, G = [], []
    for b in loader:
        mp = model(b["gem_features"].to(device))
        P.append(compute_cpi_mean(mp, b["w_min"].to(device)).cpu())
        G.append(b["cpi_golden"])
    P, G = torch.cat(P), torch.cat(G)
    return (torch.abs(P - G) / (G.abs() + 1e-8) * 100).mean().item()


def measure_cosines(model, loader, device, n_batches=5):
    """Measure cos + gradient norms of L_task, L_mono, L_const."""
    model.train()
    cos_task_mono, cos_task_const, cos_const_mono = [], [], []
    norm_task, norm_const, norm_mono = [], [], []

    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        f = b["gem_features"].to(device)
        mt = b["mscs_target"].to(device)
        cg = b["cpi_golden"].to(device)
        wm = b["w_min"].to(device)

        mp = model(f)
        lt = loss_task(mp, mt)
        lc = loss_const(mp, wm, cg)
        lm = loss_mono(mp)

        g_task = get_flat_grad(model, lt)
        g_const = get_flat_grad(model, lc)
        g_mono = get_flat_grad(model, lm)

        cos_task_mono.append(cosine(g_task, g_mono))
        cos_task_const.append(cosine(g_task, g_const))
        cos_const_mono.append(cosine(g_const, g_mono))

        norm_task.append(g_task.norm().item())
        norm_const.append(g_const.norm().item())
        norm_mono.append(g_mono.norm().item())

    model.zero_grad()
    return {
        "cos_task_mono": np.mean(cos_task_mono),
        "cos_task_const": np.mean(cos_task_const),
        "cos_const_mono": np.mean(cos_const_mono),
        "norm_task": np.mean(norm_task),
        "norm_const": np.mean(norm_const),
        "norm_mono": np.mean(norm_mono),
    }


def main():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    all_data = load_all_benchmarks(GEM_DIR, CON_DIR, BENCHMARK_NAMES)
    tl, vl, _, _, _ = create_exp1_dataloaders(all_data)
    gem_batch = min(CONCORDE_BATCH_SIZE, len(tl.dataset))
    gl = DataLoader(tl.dataset, batch_size=gem_batch, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)

    model = GEMModel(GEM_INPUT_DIM, HIDDEN, DROPOUT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONCORDE_LR, weight_decay=CONCORDE_WEIGHT_DECAY)
    steps_per_epoch = max(1, len(gl))
    total_steps = CONCORDE_EPOCHS * steps_per_epoch
    milestones = [int(total_steps * f) for f in [0.42, 0.58, 0.75, 0.92]]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    # Set weights based on MODE
    if MODE == "mono":
        lambda_task = 1.0   # gem_mono uses lambda_task=1.0 (no L_const)
        use_const_in_training = False
    else:  # full
        lambda_task = 0.1
        use_const_in_training = True

    history = []
    print(f"\nMODE={MODE}, lambda_task={lambda_task}, use_const={use_const_in_training}")
    print(f"\n{'Epoch':>6} {'MAPE':>8} {'cos(task,mono)':>16} {'cos(task,const)':>16}")
    print("-" * 54)

    for epoch in range(1, N_EPOCHS + 1):
        # Measure every epoch (always measure all 3 cosines regardless of training mode)
        m = measure_cosines(model, gl, device, N_MEASURE_BATCHES)
        mape = eval_mape(model, vl, device)
        history.append({"epoch": epoch, "mape": mape, **m})
        if epoch % 10 == 0 or epoch == 1:
            sign_tm = "CONFLICT" if m["cos_task_mono"] < -0.05 else ("ALIGN" if m["cos_task_mono"] > 0.05 else "ORTHOG")
            print(f"{epoch:>6} {mape:>7.2f}% {m['cos_task_mono']:>+15.3f}  {m['cos_task_const']:>+15.3f}   [task-mono: {sign_tm}]")

        # Training step
        model.train()
        for b in gl:
            f = b["gem_features"].to(device)
            mt, cg, wm = b["mscs_target"].to(device), b["cpi_golden"].to(device), b["w_min"].to(device)
            mp = model(f)
            l = lambda_task * loss_task(mp, mt) + LAMBDA_MONO * loss_mono(mp)
            if use_const_in_training:
                l = l + LAMBDA_CONST * loss_const(mp, wm, cg)
            optimizer.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    # ============================================================
    # Plot
    # ============================================================
    epochs = [h["epoch"] for h in history]
    ctm = [h["cos_task_mono"] for h in history]
    ctc = [h["cos_task_const"] for h in history]
    ccm = [h["cos_const_mono"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, ctm, '-', color='#CD6155', linewidth=2,
            label=r'$\cos(\nabla \mathcal{L}_{task}, \nabla \mathcal{L}_{mono})$')
    ax.plot(epochs, ctc, '-', color='#2E86C1', linewidth=2,
            label=r'$\cos(\nabla \mathcal{L}_{task}, \nabla \mathcal{L}_{const})$')
    ax.plot(epochs, ccm, '-', color='#27AE60', linewidth=2,
            label=r'$\cos(\nabla \mathcal{L}_{const}, \nabla \mathcal{L}_{mono})$')

    ax.axhline(y=0, color='black', linewidth=1.0, linestyle='-', alpha=0.5, zorder=1)

    # Shade conflict region (below zero)
    ax.axhspan(-1, 0, alpha=0.08, color='red', zorder=0)
    ax.text(0.98, 0.04, 'conflict region (cos < 0)', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='#CD6155', style='italic')

    ax.set_xlabel("Epoch", fontweight='bold')
    ax.set_ylabel("Gradient Cosine Similarity", fontweight='bold')
    ax.set_title("Gradient Direction Analysis", fontweight='bold', pad=10)
    ax.set_ylim(-0.5, 1.0)
    ax.set_xlim(1, N_EPOCHS)

    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color('black'); s.set_linewidth(1.0)

    legend = ax.legend(frameon=True, edgecolor='black', facecolor='white',
                       fancybox=False, framealpha=1.0, fontsize=12, loc='center right')
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=600, bbox_inches='tight')
    fig.savefig(OUTPUT_FIG.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_FIG}")
    print(f"Saved: {OUTPUT_FIG.replace('.png', '.pdf')}")

    # ============================================================
    # Plot 2: Gradient norm comparison
    # ============================================================
    nt = [h["norm_task"] * lambda_task for h in history]     # effective (weighted) norm
    nc = [h["norm_const"] * LAMBDA_CONST for h in history]
    nm = [h["norm_mono"] * LAMBDA_MONO for h in history]

    nt_raw = [h["norm_task"] for h in history]                # raw norm
    nc_raw = [h["norm_const"] for h in history]
    nm_raw = [h["norm_mono"] for h in history]

    fig2, (ax_raw, ax_eff) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw gradient norms
    ax_raw.plot(epochs, nt_raw, '-', color='#8E44AD', linewidth=2, label=r'$\|\nabla \mathcal{L}_{task}\|$')
    ax_raw.plot(epochs, nc_raw, '-', color='#2E86C1', linewidth=2, label=r'$\|\nabla \mathcal{L}_{const}\|$')
    ax_raw.plot(epochs, nm_raw, '-', color='#CD6155', linewidth=2, label=r'$\|\nabla \mathcal{L}_{mono}\|$')
    ax_raw.set_xlabel("Epoch", fontweight='bold')
    ax_raw.set_ylabel("Raw Gradient Norm", fontweight='bold')
    ax_raw.set_title("Raw Gradient Norms", fontweight='bold', pad=10)
    ax_raw.set_yscale('log')
    ax_raw.set_xlim(1, N_EPOCHS)
    ax_raw.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8)
    for s in ax_raw.spines.values(): s.set_visible(True); s.set_color('black'); s.set_linewidth(1.0)
    ax_raw.legend(frameon=True, edgecolor='black', facecolor='white', fancybox=False, framealpha=1.0, fontsize=12).get_frame().set_linewidth(1.0)

    # Effective (weighted) gradient norms
    ax_eff.plot(epochs, nt, '-', color='#8E44AD', linewidth=2, label=r'$\lambda_{task}\|\nabla \mathcal{L}_{task}\|$')
    ax_eff.plot(epochs, nc, '-', color='#2E86C1', linewidth=2, label=r'$\lambda_{const}\|\nabla \mathcal{L}_{const}\|$')
    ax_eff.plot(epochs, nm, '-', color='#CD6155', linewidth=2, label=r'$\lambda_{mono}\|\nabla \mathcal{L}_{mono}\|$')
    ax_eff.set_xlabel("Epoch", fontweight='bold')
    ax_eff.set_ylabel("Weighted Gradient Norm", fontweight='bold')
    ax_eff.set_title(f"Effective Gradient Norms\n($\\lambda_{{task}}$={lambda_task}, $\\lambda_{{const}}$={LAMBDA_CONST}, $\\lambda_{{mono}}$={LAMBDA_MONO})", fontweight='bold', pad=10)
    ax_eff.set_yscale('log')
    ax_eff.set_xlim(1, N_EPOCHS)
    ax_eff.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8)
    for s in ax_eff.spines.values(): s.set_visible(True); s.set_color('black'); s.set_linewidth(1.0)
    ax_eff.legend(frameon=True, edgecolor='black', facecolor='white', fancybox=False, framealpha=1.0, fontsize=12).get_frame().set_linewidth(1.0)

    plt.tight_layout()
    norm_fig = OUTPUT_FIG.replace(".png", "_norms.png")
    fig2.savefig(norm_fig, dpi=600, bbox_inches='tight')
    fig2.savefig(norm_fig.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    print(f"Saved: {norm_fig}")
    print(f"Saved: {norm_fig.replace('.png', '.pdf')}")

    # Summary
    print("\n" + "=" * 60)
    conflict_count = sum(1 for h in history if h["cos_task_mono"] < -0.05)
    print(f"  Epochs with cos(task,mono) < -0.05 (CONFLICT): {conflict_count}/{len(history)}")
    print(f"  Mean cos(task, mono):  {np.mean(ctm):+.3f}")
    print(f"  Mean cos(task, const): {np.mean(ctc):+.3f}")
    print(f"  Mean cos(const, mono): {np.mean(ccm):+.3f}")
    print(f"\n  Raw norm ratio (task:const:mono): {np.mean(nt_raw):.3f} : {np.mean(nc_raw):.3f} : {np.mean(nm_raw):.3f}")
    print(f"  Effective norm ratio: {np.mean(nt):.4f} : {np.mean(nc):.4f} : {np.mean(nm):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()