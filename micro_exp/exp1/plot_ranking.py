#!/usr/bin/env python3
"""
Plot per-component MAE + bottleneck ranking accuracy heatmaps.
Requires PRISM checkpoint + matching data for inference.
Usage: python3 plot_ranking.py
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.dpi': 600, 'axes.linewidth': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
})

# ============================================================
# Configure paths here
# ============================================================
PRISM_CKPT = "/path/to/gem5/micro_exp/exp1/results/exp1_v3_run2/exp1_512_256_183k/gem_full.pt"
DATA_SIZE = "183k"
HIDDEN = [512, 256]
OUTPUT = "/path/to/gem5/micro_exp/exp1/results/exp1_v3_run2/ranking_accuracy.png"
# ============================================================

# Custom colormaps
CMAP_RANKING = LinearSegmentedColormap.from_list("RedGreen",
    ["#FADBD8", "#FDEBD0", "#D5F5E3"])  # light red → light yellow → light green

CMAP_MAE = LinearSegmentedColormap.from_list("GreenRed",
    ["#D5F5E3", "#FDEBD0", "#FADBD8"])  # light green → light yellow → light red (low=good)

def _style_ax(ax):
    ax.yaxis.grid(False); ax.xaxis.grid(False)
    for s in ax.spines.values(): s.set_visible(True); s.set_color('black'); s.set_linewidth(1.0)

def _save(fig, path):
    fig.savefig(path, dpi=600, bbox_inches='tight')
    pdf = path.replace(".png", ".pdf")
    fig.savefig(pdf, format='pdf', bbox_inches='tight', dpi=600)
    print(f"  Saved: {path}\n  Saved: {pdf}")

def _save_pdf(fig, path):
    """Save PDF only."""
    fig.savefig(path, format='pdf', bbox_inches='tight', dpi=600)
    print(f"  Saved: {path}")

def main():
    from config import (GEM_INPUT_DIM, CONCORDE_INPUT_DIM, DROPOUT, SEED,
                        BENCHMARK_NAMES, BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES)
    from model import GEMModel, ConcordeModel, compute_cpi_mean
    from dataset import load_all_benchmarks, create_exp1_dataloaders

    gem_dir = "/data/micro_traindata_packed/gem" if DATA_SIZE == "183k" else None
    con_dir = "/data/micro_traindata_packed/concord" if DATA_SIZE == "183k" else None
    if gem_dir is None:
        from config import GEM_DATA_DIR as gem_dir, CONCORD_DATA_DIR as con_dir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED)
    all_data = load_all_benchmarks(gem_dir, con_dir, BENCHMARK_NAMES)
    tl, vl, _, _, bm_idx = create_exp1_dataloaders(all_data)

    prism = GEMModel(GEM_INPUT_DIM, HIDDEN, DROPOUT).to(device)
    prism.load_state_dict(torch.load(PRISM_CKPT, map_location=device))
    prism.eval()

    mscs_pred_all, mscs_true_all = [], []
    with torch.no_grad():
        for b in vl:
            mscs_pred_all.append(prism(b["gem_features"].to(device)).cpu())
            mscs_true_all.append(b["mscs_target"])

    pred_np = torch.cat(mscs_pred_all).numpy()
    true_np = torch.cat(mscs_true_all).numpy()

    stage_names = ['Decode', 'Dispatch', 'Issue', 'Commit']
    comp_keys = ['FP', 'BP', 'SP']
    component_names = {'FP': FP_INDICES, 'BP': BP_INDICES, 'SP': SP_INDICES}

    # MAE matrix (4 stages × 4 components)
    full_mae = np.zeros((4, 4))
    for s in range(4):
        full_mae[s, 0] = np.abs(pred_np[:, BASE_INDICES[s]] - true_np[:, BASE_INDICES[s]]).mean()
        for j, comp in enumerate(comp_keys):
            full_mae[s, j+1] = np.abs(pred_np[:, component_names[comp][s]] - true_np[:, component_names[comp][s]]).mean()

    # MAPE matrix (masked for near-zero true values)
    MAPE_EPS = 0.05
    full_mape = np.zeros((4, 4))
    for s in range(4):
        for j, idx in enumerate([BASE_INDICES[s]] + [component_names[c][s] for c in comp_keys]):
            p, t = pred_np[:, idx], true_np[:, idx]
            mask = t > MAPE_EPS
            full_mape[s, j] = (np.abs(p[mask] - t[mask]) / t[mask] * 100).mean() if mask.sum() > 0 else 0.0

    # Ranking accuracy (only on samples where components are clearly different)
    EPS = 0.01  # threshold for "clearly different"
    dominant_acc = np.zeros(4)
    pairwise_acc = np.zeros((4, 3))
    for s in range(4):
        fp_p, fp_t = pred_np[:, FP_INDICES[s]], true_np[:, FP_INDICES[s]]
        bp_p, bp_t = pred_np[:, BP_INDICES[s]], true_np[:, BP_INDICES[s]]
        sp_p, sp_t = pred_np[:, SP_INDICES[s]], true_np[:, SP_INDICES[s]]

        pred_dom = np.argmax(np.stack([fp_p, bp_p, sp_p], 1), 1)
        true_dom = np.argmax(np.stack([fp_t, bp_t, sp_t], 1), 1)
        dominant_acc[s] = (pred_dom == true_dom).mean() * 100

        # Pairwise: only count samples where the true difference exceeds EPS
        for k, (a_t, a_p, b_t, b_p) in enumerate([
            (fp_t, fp_p, bp_t, bp_p),   # FP vs BP
            (fp_t, fp_p, sp_t, sp_p),   # FP vs SP
            (bp_t, bp_p, sp_t, sp_p),   # BP vs SP
        ]):
            mask = np.abs(a_t - b_t) > EPS
            if mask.sum() > 0:
                pairwise_acc[s, k] = ((a_p[mask] > b_p[mask]) == (a_t[mask] > b_t[mask])).mean() * 100
            else:
                pairwise_acc[s, k] = 100.0  # no distinguishable pairs → trivially correct

    # ============================================================
    # Fig 1: Per-Component MAE heatmap
    # ============================================================
    full_labels = ['Base', 'FP', 'BP', 'SP']
    fig1, ax_mae = plt.subplots(figsize=(7, 4.0))
    im1 = ax_mae.imshow(full_mae, cmap=CMAP_MAE, aspect='auto')
    ax_mae.set_xticks(range(4)); ax_mae.set_xticklabels(full_labels, fontweight='bold')
    ax_mae.set_yticks(range(4)); ax_mae.set_yticklabels(stage_names, fontweight='bold')
    ax_mae.set_title("Per-Component MAE", fontweight='bold', pad=10)
    for i in range(4):
        for j in range(4):
            ax_mae.text(j, i, f"{full_mae[i,j]:.3f}", ha='center', va='center',
                       fontsize=13, fontweight='bold', color='black')
    fig1.colorbar(im1, ax=ax_mae, shrink=0.8).ax.set_ylabel("MAE", fontsize=10)
    _style_ax(ax_mae)
    plt.tight_layout()
    _save_pdf(fig1, OUTPUT.replace(".png", "_mae.pdf"))

    # ============================================================
    # Fig 1b: Per-Component MAPE heatmap
    # ============================================================
    fig_mape, ax_mape = plt.subplots(figsize=(7, 4.0))
    im_mape = ax_mape.imshow(full_mape, cmap=CMAP_MAE, aspect='auto')
    ax_mape.set_xticks(range(4)); ax_mape.set_xticklabels(full_labels, fontweight='bold')
    ax_mape.set_yticks(range(4)); ax_mape.set_yticklabels(stage_names, fontweight='bold')
    ax_mape.set_title("Per-Component MAPE (%)\n(samples with true > 0.05)", fontweight='bold', pad=10)
    for i in range(4):
        for j in range(4):
            ax_mape.text(j, i, f"{full_mape[i,j]:.1f}%", ha='center', va='center',
                        fontsize=13, fontweight='bold', color='black')
    fig_mape.colorbar(im_mape, ax=ax_mape, shrink=0.8).ax.set_ylabel("MAPE (%)", fontsize=10)
    _style_ax(ax_mape)
    plt.tight_layout()
    _save_pdf(fig_mape, OUTPUT.replace(".png", "_mape.pdf"))

    # ============================================================
    # Fig 2: Ranking accuracy heatmap
    # ============================================================
    fig2, ax_rank = plt.subplots(figsize=(7, 4.0))
    rank_matrix = np.column_stack([dominant_acc, pairwise_acc])
    rank_labels = ['Dominant', 'FP vs BP', 'FP vs SP', 'BP vs SP']
    im2 = ax_rank.imshow(rank_matrix, cmap=CMAP_RANKING, aspect='auto', vmin=50, vmax=100)
    ax_rank.set_xticks(range(4)); ax_rank.set_xticklabels(rank_labels, fontweight='bold', fontsize=9)
    ax_rank.set_yticks(range(4)); ax_rank.set_yticklabels(stage_names, fontweight='bold')
    ax_rank.set_title("Bottleneck Ranking Accuracy (%)\n(pairs with true diff > 0.01)", fontweight='bold', pad=10)
    for i in range(4):
        for j in range(4):
            ax_rank.text(j, i, f"{rank_matrix[i,j]:.1f}%", ha='center', va='center',
                        fontsize=13, fontweight='bold', color='black')
    fig2.colorbar(im2, ax=ax_rank, shrink=0.8).ax.set_ylabel("Accuracy (%)", fontsize=10)
    _style_ax(ax_rank)
    plt.tight_layout()
    _save_pdf(fig2, OUTPUT.replace(".png", "_ranking.pdf"))

    # Print summary
    print(f"\n  {'Stage':<12} {'Dominant':>10} {'FP>BP':>8} {'FP>SP':>8} {'BP>SP':>8}")
    print("  " + "-" * 48)
    for s, stage in enumerate(stage_names):
        print(f"  {stage:<12} {dominant_acc[s]:>9.1f}% {pairwise_acc[s,0]:>7.1f}% {pairwise_acc[s,1]:>7.1f}% {pairwise_acc[s,2]:>7.1f}%")
    print(f"  {'Average':<12} {dominant_acc.mean():>9.1f}% {pairwise_acc.mean(0)[0]:>7.1f}% {pairwise_acc.mean(0)[1]:>7.1f}% {pairwise_acc.mean(0)[2]:>7.1f}%")

    # Diagnostic: why is a cell low?
    print("\n  Diagnostic — pairwise difference distribution:")
    for s, stage in enumerate(stage_names):
        bp_t, sp_t = true_np[:, BP_INDICES[s]], true_np[:, SP_INDICES[s]]
        diff = np.abs(bp_t - sp_t)
        n_clear = (diff > EPS).sum()
        print(f"    {stage} BP vs SP: {n_clear}/{len(diff)} samples with |diff|>{EPS} "
              f"(mean diff={diff.mean():.4f})")

    plt.close("all")
    print("\nDone!")

if __name__ == "__main__":
    main()