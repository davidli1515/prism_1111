#!/usr/bin/env python3
"""
Plot per-benchmark error breakdown.
- PRISM: average + P90 both from JSON (different normalizer)
- Concorde & Vanilla: inference from checkpoints
Usage: python3 plot_error_breakdown.py
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 600, 'axes.linewidth': 1.0,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 4, 'ytick.major.size': 4,
})

COLOR_PRISM = '#E67E22';       COLOR_PRISM_LIGHT = '#F5CBA7'
COLOR_CONCORDE = '#2E86C1';    COLOR_CONCORDE_LIGHT = '#AED6F1'
COLOR_VANILLA = '#CD6155';     COLOR_VANILLA_LIGHT = '#E6B0AA'

# ============================================================
# Configure paths here
# ============================================================

# --- PRISM: all from JSON (avg + P90) ---
PRISM_JSON    = "/path/to/gem5/micro_exp/exp4/results_p90/exp4_18k_extra_unlabeled/exp4_100pct.json"

# --- Concorde & Vanilla: avg from JSON, P90 from checkpoint inference ---
CONCORDE_JSON = "/path/to/gem5/micro_exp/exp1/results/exp1_v2_run2/exp1_512_256_183k/concorde.json"
CONCORDE_CKPT = "/path/to/gem5/micro_exp/exp1/results/exp1_v2_run2/exp1_512_256_183k/concorde.pt"

VANILLA_JSON  = "/path/to/gem5/micro_exp/exp1/results/exp1_v2_run2/exp1_512_256_183k/gem_vanilla.json"
VANILLA_CKPT  = "/path/to/gem5/micro_exp/exp1/results/exp1_v2_run2/exp1_512_256_183k/gem_vanilla.pt"

# --- Data config (for Concorde/Vanilla inference) ---
DATA_SIZE = "183k"
HIDDEN = [512, 256]

# --- Output ---
OUTPUT = "/path/to/gem5/micro_exp/exp1/results/exp1_v2_run2/error_breakdown_183k.png"

# ============================================================

def _style_ax(ax):
    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0)
    ax.xaxis.grid(False)
    for s in ax.spines.values(): s.set_visible(True); s.set_color('black'); s.set_linewidth(1.0)

def _save(fig, path):
    fig.savefig(path, dpi=600, bbox_inches='tight')
    pdf = path.replace(".png", ".pdf")
    fig.savefig(pdf, format='pdf', bbox_inches='tight', dpi=600)
    print(f"  Saved: {path}\n  Saved: {pdf}")

def load_json(path):
    with open(path) as f: return json.load(f)

def main():
    # ============================================================
    # Load PRISM from JSON (avg + P90)
    # ============================================================
    print("=== Loading PRISM from JSON ===")
    prism_data = load_json(PRISM_JSON)
    p_pbm = prism_data["per_benchmark"]
    p_p90_pbm = prism_data["p90_per_benchmark"]
    p_overall = prism_data["overall_mape"]
    p_overall_p90 = prism_data["p90"]
    print(f"  PRISM avg={p_overall:.2f}%, P90={p_overall_p90:.2f}%")

    # ============================================================
    # Load Concorde & Vanilla avg from JSON
    # ============================================================
    c_data = load_json(CONCORDE_JSON)
    if "overall_mape" in c_data:
        c_overall, c_pbm = c_data["overall_mape"], c_data["per_benchmark"]
    else:
        for k in c_data["overall"]: c_overall, c_pbm = c_data["overall"][k], c_data["per_benchmark"][k]
    print(f"  Concorde avg={c_overall:.2f}%")

    has_vanilla = VANILLA_JSON and os.path.exists(VANILLA_JSON)
    if has_vanilla:
        v_data = load_json(VANILLA_JSON)
        if "overall_mape" in v_data:
            v_overall, v_pbm = v_data["overall_mape"], v_data["per_benchmark"]
        else:
            for k in v_data["overall"]: v_overall, v_pbm = v_data["overall"][k], v_data["per_benchmark"][k]
        print(f"  Vanilla avg={v_overall:.2f}%")

    # Benchmark list and sort
    benchmarks = sorted(p_pbm.keys())
    short_names = [b.replace("_r","").replace("_s","").split(".")[-1] for b in benchmarks]
    prism_avg = [p_pbm[b] for b in benchmarks]
    con_avg = [c_pbm.get(b, 0) for b in benchmarks]
    van_avg = [v_pbm.get(b, 0) for b in benchmarks] if has_vanilla else []

    sort_idx = np.argsort(con_avg)
    bm_sorted = [benchmarks[i] for i in sort_idx]
    short_sorted = [short_names[i] for i in sort_idx]
    prism_avg_s = [prism_avg[i] for i in sort_idx]
    con_avg_s = [con_avg[i] for i in sort_idx]
    van_avg_s = [van_avg[i] for i in sort_idx] if has_vanilla else []

    # Geomean
    con_geo = np.exp(np.mean(np.log(np.array(con_avg_s)+1e-8)))
    prism_geo = np.exp(np.mean(np.log(np.array(prism_avg_s)+1e-8)))
    van_geo = np.exp(np.mean(np.log(np.array(van_avg_s)+1e-8))) if has_vanilla else 0
    print(f"  GeoMean — PRISM: {prism_geo:.2f}%, Concorde: {con_geo:.2f}%", end="")
    if has_vanilla: print(f", Vanilla: {van_geo:.2f}%", end="")
    print()

    all_labels = ['GeoMean'] + short_sorted
    all_con_avg = [con_geo] + con_avg_s
    all_prism_avg = [prism_geo] + prism_avg_s
    all_van_avg = ([van_geo] + van_avg_s) if has_vanilla else []

    n_total = len(all_labels)
    x = np.arange(n_total)
    n_bars = 3 if has_vanilla else 2
    w = 0.8 / n_bars

    # ============================================================
    # Plot 1: Average only
    # ============================================================
    fig, ax = plt.subplots(figsize=(max(12, n_total*0.7), 4.0))
    if has_vanilla:
        ax.bar(x-w, all_con_avg, w, color=COLOR_CONCORDE, edgecolor='black', linewidth=0.3, zorder=3, label='Concorde')
        ax.bar(x, all_prism_avg, w, color=COLOR_PRISM, edgecolor='black', linewidth=0.3, zorder=3, label='PRISM')
        ax.bar(x+w, all_van_avg, w, color=COLOR_VANILLA, edgecolor='black', linewidth=0.3, zorder=3, label='PRISM (vanilla)')
    else:
        ax.bar(x-w/2, all_con_avg, w, color=COLOR_CONCORDE, edgecolor='black', linewidth=0.3, zorder=3, label='Concorde')
        ax.bar(x+w/2, all_prism_avg, w, color=COLOR_PRISM, edgecolor='black', linewidth=0.3, zorder=3, label='PRISM')
    ax.axvline(x=0.5, color='black', linewidth=0.8, linestyle='--', zorder=4, alpha=0.5)
    ax.set_ylabel("Average CPI Error (%)", fontweight='bold')
    ax.set_title("Per-Benchmark Average Error", fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=11, fontweight='bold')
    ax.get_xticklabels()[0].set_fontsize(13); ax.get_xticklabels()[0].set_color('red')
    _style_ax(ax)
    ax.legend(frameon=True, edgecolor='black', facecolor='white', fancybox=False, framealpha=1.0, fontsize=13, borderpad=0.8, handletextpad=0.5).get_frame().set_linewidth(1.0)
    ax.set_xlim(-0.5, n_total-0.5)
    plt.tight_layout(); _save(fig, OUTPUT)

    # Print table
    print(f"\n  {'Benchmark':<20} {'Concorde':>10} {'PRISM':>10}", end="")
    if has_vanilla: print(f" {'Vanilla':>10}", end="")
    print()
    print("  "+"-"*(42 + (12 if has_vanilla else 0)))
    pw = 0
    for i in range(len(bm_sorted)):
        win = "PRISM" if prism_avg_s[i] <= con_avg_s[i] else "Concorde"
        if prism_avg_s[i] <= con_avg_s[i]: pw += 1
        print(f"  {bm_sorted[i]:<20} {con_avg_s[i]:>9.2f}% {prism_avg_s[i]:>9.2f}%", end="")
        if has_vanilla: print(f" {van_avg_s[i]:>9.2f}%", end="")
        print(f"  {win}")
    print(f"  PRISM wins: {pw}/{len(bm_sorted)}")

    # ============================================================
    # Plot 2: Average + P90 (PRISM P90 from JSON, others from inference)
    # ============================================================
    if not os.path.exists(CONCORDE_CKPT):
        print("\n  Skipping P90 plot (Concorde checkpoint not found)")
        plt.close("all"); return

    has_vanilla_ckpt = VANILLA_CKPT and os.path.exists(VANILLA_CKPT)
    print(f"\n=== P90 plot (PRISM from JSON, Concorde/Vanilla from inference) ===")

    from config import (GEM_INPUT_DIM, CONCORDE_INPUT_DIM, DROPOUT, SEED,
                        BENCHMARK_NAMES, BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES)
    from model import GEMModel, ConcordeModel, compute_cpi_mean
    from dataset import load_all_benchmarks, create_exp1_dataloaders

    gem_dir = "/data/micro_traindata_packed/gem" if DATA_SIZE=="183k" else None
    con_dir = "/data/micro_traindata_packed/concord" if DATA_SIZE=="183k" else None
    if gem_dir is None:
        from config import GEM_DATA_DIR as gem_dir, CONCORD_DATA_DIR as con_dir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED)
    all_data = load_all_benchmarks(gem_dir, con_dir, BENCHMARK_NAMES)
    tl, vl, _, _, bm_idx = create_exp1_dataloaders(all_data)

    # Load Concorde
    concorde = ConcordeModel(CONCORDE_INPUT_DIM, HIDDEN, 0.0).to(device)
    concorde.load_state_dict(torch.load(CONCORDE_CKPT, map_location=device)); concorde.eval()

    # Load Vanilla
    vanilla = None
    if has_vanilla_ckpt:
        vanilla = GEMModel(GEM_INPUT_DIM, HIDDEN, DROPOUT).to(device)
        vanilla.load_state_dict(torch.load(VANILLA_CKPT, map_location=device)); vanilla.eval()

    # Inference (Concorde + Vanilla only)
    con_pred, van_pred, all_gold = [], [], []
    with torch.no_grad():
        for b in vl:
            con_pred.append(concorde(b["concord_features"].to(device)).cpu())
            if vanilla is not None:
                mscs_v = vanilla(b["gem_features"].to(device))
                van_pred.append(compute_cpi_mean(mscs_v, b["w_min"].to(device)).cpu())
            all_gold.append(b["cpi_golden"])
    con_pred = torch.cat(con_pred); all_gold = torch.cat(all_gold)
    con_pct_err = (torch.abs(con_pred - all_gold) / (all_gold+1e-8) * 100)
    if vanilla is not None:
        van_pred = torch.cat(van_pred)
        van_pct_err = (torch.abs(van_pred - all_gold) / (all_gold+1e-8) * 100)
    else:
        van_pct_err = None

    # Build P90 arrays
    prism_p90, con_p90, van_p90 = [], [], []
    for bm in bm_sorted:
        prism_p90.append(p_p90_pbm.get(bm, 0))  # PRISM from JSON
        s, e = bm_idx[bm]
        con_p90.append(con_pct_err[s:e].quantile(0.9).item())
        if van_pct_err is not None:
            van_p90.append(van_pct_err[s:e].quantile(0.9).item())

    con_p90_geo = np.exp(np.mean(np.log(np.array(con_p90)+1e-8)))
    prism_p90_geo = np.exp(np.mean(np.log(np.array(prism_p90)+1e-8)))
    van_p90_geo = np.exp(np.mean(np.log(np.array(van_p90)+1e-8))) if van_p90 else 0

    all_con_p90 = [con_p90_geo] + con_p90
    all_prism_p90 = [prism_p90_geo] + prism_p90
    all_van_p90 = ([van_p90_geo] + van_p90) if van_p90 else []

    # --- Average + P90 plot ---
    fig2, ax2 = plt.subplots(figsize=(max(12, n_total*0.7), 4.0))
    if has_vanilla_ckpt:
        ax2.bar(x-w, all_con_p90, w, color=COLOR_CONCORDE_LIGHT, edgecolor='black', linewidth=0.3, zorder=2, label='Concorde P90')
        ax2.bar(x, all_prism_p90, w, color=COLOR_PRISM_LIGHT, edgecolor='black', linewidth=0.3, zorder=2, label='PRISM P90')
        ax2.bar(x+w, all_van_p90, w, color=COLOR_VANILLA_LIGHT, edgecolor='black', linewidth=0.3, zorder=2, label='V1 P90')
        ax2.bar(x-w, all_con_avg, w, color=COLOR_CONCORDE, edgecolor='black', linewidth=0.3, zorder=3, label='Concorde avg')
        ax2.bar(x, all_prism_avg, w, color=COLOR_PRISM, edgecolor='black', linewidth=0.3, zorder=3, label='PRISM avg')
        ax2.bar(x+w, all_van_avg, w, color=COLOR_VANILLA, edgecolor='black', linewidth=0.3, zorder=3, label='V1 avg')
    else:
        ax2.bar(x-w/2, all_con_p90, w, color=COLOR_CONCORDE_LIGHT, edgecolor='black', linewidth=0.3, zorder=2, label='Concorde P90')
        ax2.bar(x-w/2, all_con_avg, w, color=COLOR_CONCORDE, edgecolor='black', linewidth=0.3, zorder=3, label='Concorde avg')
        ax2.bar(x+w/2, all_prism_p90, w, color=COLOR_PRISM_LIGHT, edgecolor='black', linewidth=0.3, zorder=2, label='PRISM P90')
        ax2.bar(x+w/2, all_prism_avg, w, color=COLOR_PRISM, edgecolor='black', linewidth=0.3, zorder=3, label='PRISM avg')

    ax2.axvline(x=0.5, color='black', linewidth=0.8, linestyle='--', zorder=4, alpha=0.5)
    ax2.set_ylabel("CPI Error (%)", fontweight='bold')
    ax2.set_title("Per-Benchmark Error (Average + 90th Percentile)", fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=11, fontweight='bold')
    ax2.get_xticklabels()[0].set_fontsize(13); ax2.get_xticklabels()[0].set_color('red')
    _style_ax(ax2)
    handles, labels = ax2.get_legend_handles_labels()
    order = [3,0,4,1,5,2] if has_vanilla_ckpt else [1,0,3,2]
    ax2.legend([handles[i] for i in order], [labels[i] for i in order],
               loc='upper left', frameon=True, edgecolor='black', facecolor='white',
               fancybox=False, framealpha=1.0, ncol=3 if has_vanilla_ckpt else 2,
               columnspacing=1.0, handletextpad=0.5, borderpad=0.8, fontsize=10).get_frame().set_linewidth(1.0)
    ax2.set_xlim(-0.5, n_total-0.5)
    plt.tight_layout(); _save(fig2, OUTPUT.replace(".png","_with_p90.png"))

    print(f"\n  P90 GeoMean — PRISM: {prism_p90_geo:.2f}%, Concorde: {con_p90_geo:.2f}%", end="")
    if van_p90: print(f", Vanilla: {van_p90_geo:.2f}%", end="")
    print()

    # ============================================================
    # Plot 3: Ranking accuracy (uses Concorde normalizer inference)
    # ============================================================
    # Skip if no PRISM checkpoint (can't do component analysis from JSON)
    print("\n  Note: Component/ranking analysis requires PRISM checkpoint inference.")
    print("  Skipping component plot (PRISM uses different normalizer).")

    plt.close("all")
    print("\nDone!")

if __name__ == "__main__":
    main()