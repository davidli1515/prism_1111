"""
Shapley Value Pruning via MSCS Cosine Similarity

Theory:
  - Full Shapley for d params requires O(d!) evaluations (Monte Carlo approximation)
  - MSCS provides 12-dim bottleneck vector per param (FP/BP/SP × 4 stages)
  - Params with high |cos(v_i, v_j)| interact (same pipeline mechanism)
  - Params with low |cos| are independent → Shapley factorizes across groups
  - Pruned Shapley: O(n1! + n2! + ... + nk!) << O(d!)

Usage:
    python shapley_pruning.py --benchmark 605.mcf_s
    python shapley_pruning.py --benchmark 605.mcf_s --n-permutations 200
"""

import argparse, os, sys, time, itertools, math
import numpy as np
import torch
from collections import OrderedDict
from scipy.stats import spearmanr

np.seterr(divide='ignore', invalid='ignore')

EXP1_DIR = "/scratch/gem5-TAO/micro_exp/exp1"
DSE_DIR  = "/scratch/gem5-TAO/micro_exp/dse_exp"
sys.path.insert(0, EXP1_DIR)
sys.path.insert(0, DSE_DIR)

from model import GEMModel
from dataset import FeatureNormalizer
from dse_infer import DseEngine
from config_dse import (
    GRADIENT_PARAMS, PARAM_TO_GD_INDEX, ALL_PARAMS,
    compute_w_min, get_param_values,
)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

GEM_INPUT_DIM   = 4681
GEM_HIDDEN_DIMS = [512, 256]
GEM_WEIGHT_PATH = "/scratch/gem5-TAO/micro_exp/exp1/results/exp1_v2/exp1_512_256/gem_full.pt"
NORMALIZER_CACHE = os.path.join(DSE_DIR, "normalizer_cache")
OUTPUT_DIR = os.path.join(DSE_DIR, "result_shapley_520")

PRECOMPUTE_BASE = os.path.join(DSE_DIR, "precompute")


def load_model(benchmark=None):
    """Load GEM model, normalizer, and DSE engine. Returns (model, normalizer, dse_engine)."""
    os.chdir(DSE_DIR)
    m = GEMModel(input_dim=GEM_INPUT_DIM, hidden_dims=GEM_HIDDEN_DIMS, dropout=0.1)
    m.load_state_dict(torch.load(GEM_WEIGHT_PATH, map_location="cpu"))
    m.eval()

    norm = FeatureNormalizer()
    norm.mean = np.load(os.path.join(NORMALIZER_CACHE, "gem_mean.npy"))
    norm.std  = np.load(os.path.join(NORMALIZER_CACHE, "gem_std.npy"))

    dse = DseEngine(PRECOMPUTE_BASE)
    return m, norm, dse


# MSCS indices: FP/BP/SP × 4 stages (exclude Base)
FP_INDICES = [1, 5, 9, 13]
BP_INDICES = [2, 6, 10, 14]
SP_INDICES = [3, 7, 11, 15]
MSCS_12_INDICES = FP_INDICES + BP_INDICES + SP_INDICES  # 12-dim bottleneck vector

DEFAULT_CONFIG = {
    "fetchWidth": 8, "decodeWidth": 8, "dispatchWidth": 8,
    "commitWidth": 8, "wbWidth": 8,
    "numROBEntries": 192, "numIQEntries": 64,
    "LQEntries": 32, "SQEntries": 32, "fetchBufferSize": 32,
    "IntALU": 2, "IntMultDiv": 1, "FP_ALU": 2, "FP_MultDiv": 1, "RdWrPort": 2,
    "l1_size": 64, "l2_size": 1024,
    "l1d_assoc": 2, "l1i_assoc": 2, "l2_assoc": 8, "bp_type": 0,
}

PARAM_STRIDE = {
    "fetchWidth": 1, "decodeWidth": 1, "dispatchWidth": 1, "commitWidth": 1,
    "wbWidth": 1,
    "numROBEntries": 16, "numIQEntries": 8, "LQEntries": 8, "SQEntries": 8,
    "fetchBufferSize": 8,
    "IntALU": 1, "IntMultDiv": 1, "FP_ALU": 1, "FP_MultDiv": 1, "RdWrPort": 1,
}

MAX_SLICES = 200

# All tunable architectural parameters for Shapley attribution
# Hardcoded here to avoid dependency on config_dse.GRADIENT_PARAMS
SHAPLEY_PARAMS = [
    # Pipeline widths
    "fetchWidth", "decodeWidth", "dispatchWidth", "commitWidth", "wbWidth",
    # Queue/buffer sizes
    "numROBEntries", "numIQEntries", "LQEntries", "SQEntries", "fetchBufferSize",
    # Functional units
    "IntALU", "IntMultDiv", "FP_ALU", "FP_MultDiv", "RdWrPort",
    # Cache hierarchy
    "l1_size", "l1d_assoc", "l1i_assoc",
]

# Concorde-style color mapping for parameter categories
PARAM_COLORS = {
    # Pipeline widths - purples/pinks
    "fetchWidth":      "#9b59b6",
    "decodeWidth":     "#c39bd3",
    "dispatchWidth":   "#AF7AC5",
    "commitWidth":     "#1a1a2e",
    "wbWidth":         "#d5a6bd",
    # Queue/buffer sizes - reds/oranges
    "numROBEntries":   "#e74c3c",
    "numIQEntries":    "#f1948a",
    "LQEntries":       "#e67e22",
    "SQEntries":       "#f0b27a",
    "fetchBufferSize": "#fad7a0",
    # Functional units - greens/teals
    "IntALU":          "#16a085",
    "IntMultDiv":      "#1abc9c",
    "FP_ALU":          "#2ecc71",
    "FP_MultDiv":      "#82e0aa",
    "RdWrPort":        "#abebc6",
    # Cache - blues/cyans
    "l1_size":         "#00bcd4",
    "l1d_assoc":       "#3498db",
    "l1i_assoc":       "#85c1e9",
}



def predict_cpi(model, normalizer, dse_engine, workload, config):
    all_feats = dse_engine.get_features(config, workload=workload)[:MAX_SLICES]
    batch = np.stack([f[:GEM_INPUT_DIM] for f in all_feats])
    x = torch.from_numpy(normalizer.transform(batch)).float()
    with torch.no_grad():
        mscs = model(x)
        w_min = float(compute_w_min(config))
        cpis = 1.0 / (mscs[:, 12] * w_min + 1e-8)
        return cpis.mean().item()


def predict_mscs(model, normalizer, dse_engine, workload, config):
    all_feats = dse_engine.get_features(config, workload=workload)[:MAX_SLICES]
    batch = np.stack([f[:GEM_INPUT_DIM] for f in all_feats])
    x = torch.from_numpy(normalizer.transform(batch)).float()
    with torch.no_grad():
        return model(x).mean(dim=0).cpu().numpy()


def increase_param(config, name):
    c = config.copy()
    vals = get_param_values(name)
    current = c[name]
    if isinstance(ALL_PARAMS.get(name), list):
        idx = vals.index(current) if current in vals else 0
        if idx >= len(vals) - 1:
            return None, None
        c[name] = vals[idx + 1]
    else:
        stride = PARAM_STRIDE.get(name, 1)
        hi = max(vals)
        new_val = min(hi, current + stride)
        if new_val == current:
            return None, None
        c[name] = new_val
    return c, c[name]


# Override min values for params where precompute isn't available at true min
PARAM_MIN_OVERRIDE = {
    "l1_size": 32,  # precompute for l1_size=16 not yet computed
}


def get_param_min(name):
    """Get minimum value for a parameter."""
    if name in PARAM_MIN_OVERRIDE:
        return PARAM_MIN_OVERRIDE[name]
    vals = get_param_values(name)
    return min(vals)


def get_param_max(name):
    """Get maximum value for a parameter."""
    vals = get_param_values(name)
    return max(vals)


def make_min_config(base_config, param_list):
    """Config with all tunable params at their minimum values."""
    config = base_config.copy()
    for name in param_list:
        config[name] = get_param_min(name)
    return config


def make_max_config(base_config, param_list):
    """Config with all tunable params at their maximum values."""
    config = base_config.copy()
    for name in param_list:
        config[name] = get_param_max(name)
    return config


# ============================================================
# Step 1: MSCS finite-diff vectors (12-dim, DEFAULT→min)
# ============================================================
def compute_mscs_vectors(model, normalizer, dse_engine, workload, config, params):
    """For each param, compute 12-dim MSCS change vector.
    
    Reference = DEFAULT_CONFIG. Perturb each param to min.
    Consistent context with Shapley evaluation (both use default→min).
    
    For params already at min in DEFAULT, use increase direction as fallback.
    """
    mscs_default = predict_mscs(model, normalizer, dse_engine, workload, config)
    
    vectors = {}
    target_vals = {}
    vector_info = {}
    
    for name in params:
        vals = get_param_values(name)
        current = config[name]
        param_min = min(vals)
        
        if current > param_min:
            config_removed = config.copy()
            config_removed[name] = param_min
            mscs_removed = predict_mscs(model, normalizer, dse_engine, workload, config_removed)
            delta = mscs_removed[MSCS_12_INDICES] - mscs_default[MSCS_12_INDICES]
            vectors[name] = delta
            target_vals[name] = current
            vector_info[name] = f"default→min ({current}→{param_min})"
        else:
            c_inc, new_val = increase_param(config, name)
            if c_inc is None:
                continue
            mscs_inc = predict_mscs(model, normalizer, dse_engine, workload, c_inc)
            delta = mscs_default[MSCS_12_INDICES] - mscs_inc[MSCS_12_INDICES]
            vectors[name] = delta
            target_vals[name] = new_val
            vector_info[name] = f"default→inc ({current}→{new_val})"
    
    return vectors, target_vals, vector_info


# ============================================================
# Step 2: Pairwise cosine similarity
# ============================================================
def compute_cosine_matrix(vectors, param_list):
    """Compute |cos(v_i, v_j)| for all pairs."""
    n = len(param_list)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            vi = vectors[param_list[i]]
            vj = vectors[param_list[j]]
            norm_i = np.linalg.norm(vi)
            norm_j = np.linalg.norm(vj)
            if norm_i < 1e-10 or norm_j < 1e-10:
                sim[i][j] = 0
            else:
                sim[i][j] = abs(np.dot(vi, vj) / (norm_i * norm_j))
    return sim


# ============================================================
# Step 3: Union-Find grouping
# ============================================================
def group_params(sim_matrix, param_list, threshold=0.5, max_group_size=999):
    """
    Group params by cosine similarity (clique-based).
    A param joins a group only if |cos| > threshold with ALL existing members.
    No transitive closure — prevents chain-linking unrelated params.
    """
    n = len(param_list)
    groups = []
    assigned = [False] * n
    
    # Sort params by average similarity (most connected first)
    avg_sim = [np.mean([sim_matrix[i][j] for j in range(n) if j != i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: avg_sim[i], reverse=True)
    
    for i in order:
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in order:
            if assigned[j] or j == i:
                continue
            if len(group) >= max_group_size:
                break
            # Must have high similarity to ALL current group members
            if all(sim_matrix[j][k] >= threshold for k in group):
                group.append(j)
                assigned[j] = True
        groups.append(group)
    
    named_groups = [[param_list[i] for i in g] for g in groups]
    named_groups.sort(key=len, reverse=True)
    return named_groups


# ============================================================
# Step 4a: Full Monte Carlo Shapley (ground truth, no cache)
# ============================================================
def shapley_full_mc(model, normalizer, dse_engine, workload, base_config,
                     param_list, target_vals, n_permutations=200, seed=42):
    """
    Monte Carlo Shapley — no caching, every eval is real inference.
    Aligned with Concorde's 200-permutation protocol.
    Eval count = n_permutations × (d + 1).
    """
    min_config = make_min_config(base_config, param_list)
    
    rng = np.random.RandomState(seed)
    d = len(param_list)
    shapley = {name: 0.0 for name in param_list}
    n_evals = 0
    
    for perm_idx in range(n_permutations):
        perm = rng.permutation(d)
        config = min_config.copy()
        prev_cpi = predict_cpi(model, normalizer, dse_engine, workload, config)
        n_evals += 1
        
        for pos in range(d):
            name = param_list[perm[pos]]
            config[name] = target_vals[name]
            curr_cpi = predict_cpi(model, normalizer, dse_engine, workload, config)
            n_evals += 1
            
            marginal = prev_cpi - curr_cpi
            shapley[name] += marginal / n_permutations
            prev_cpi = curr_cpi
        
        if (perm_idx + 1) % 50 == 0:
            print(f"      MC Shapley: {perm_idx + 1}/{n_permutations} permutations, {n_evals} evals")
    
    return shapley, n_evals


# ============================================================
# Step 4b: Pruned Shapley (within groups only)
# ============================================================
def shapley_exact_small(model, normalizer, dse_engine, workload, base_config,
                         group_params_list, target_vals, param_list_all):
    """
    Exact Shapley for a small group.
    Outside-group params fixed at DEFAULT_CONFIG (realistic working context).
    Group params: empty set = all at min, "adding" = set to DEFAULT/target.
    This way, even if independence assumption is imperfect, the context is reasonable.
    """
    # Start from DEFAULT_CONFIG, only set GROUP params to min
    context_config = base_config.copy()  # outside params stay at DEFAULT
    for name in group_params_list:
        context_config[name] = get_param_min(name)
    
    d = len(group_params_list)
    shapley = {name: 0.0 for name in group_params_list}
    n_evals = 0
    
    cpi_cache = {}
    
    def get_cpi(subset_key):
        nonlocal n_evals
        if subset_key not in cpi_cache:
            config = context_config.copy()
            for name in subset_key:
                config[name] = target_vals[name]
            cpi_cache[subset_key] = predict_cpi(model, normalizer, dse_engine, workload, config)
            n_evals += 1
        return cpi_cache[subset_key]
    
    # Enumerate all permutations
    for perm in itertools.permutations(range(d)):
        current_set = []
        prev_cpi = get_cpi(())
        
        for pos in range(d):
            name = group_params_list[perm[pos]]
            current_set.append(name)
            current_key = tuple(sorted(current_set))
            curr_cpi = get_cpi(current_key)
            marginal = prev_cpi - curr_cpi
            shapley[name] += marginal
            prev_cpi = curr_cpi
    
    # Average over d! permutations
    n_perms = math.factorial(d)
    for name in shapley:
        shapley[name] /= n_perms
    
    return shapley, n_evals


def shapley_pruned(model, normalizer, dse_engine, workload, base_config,
                    groups, target_vals, param_list_all, n_permutations=10):
    """
    Pruned Shapley: exact within each group, assume independence across groups.
      ≤8: exact enumeration with cpi_cache
      >8: MC with n_permutations (default 10, groups this large are rare with clique)
    Outside-group params fixed at DEFAULT_CONFIG.
    """
    shapley = {}
    total_evals = 0
    
    for gi, group in enumerate(groups):
        gs = len(group)
        
        if gs <= 8:
            sv, ne = shapley_exact_small(model, normalizer, dse_engine, workload,
                                          base_config, group, target_vals, param_list_all)
            print(f"    Group {gi} ({gs} params): exact, {ne} evals ({gs}!={math.factorial(gs)})")
        else:
            # MC Shapley within group, no cache
            context_config = base_config.copy()
            for name in group:
                context_config[name] = get_param_min(name)
            
            rng = np.random.RandomState(42 + gi)
            d = gs
            sv = {name: 0.0 for name in group}
            ne = 0
            
            for perm_idx in range(n_permutations):
                perm = rng.permutation(d)
                config = context_config.copy()
                prev_cpi = predict_cpi(model, normalizer, dse_engine, workload, config)
                ne += 1
                for pos in range(d):
                    name = group[perm[pos]]
                    config[name] = target_vals[name]
                    curr_cpi = predict_cpi(model, normalizer, dse_engine, workload, config)
                    ne += 1
                    marginal = prev_cpi - curr_cpi
                    sv[name] += marginal / n_permutations
                    prev_cpi = curr_cpi
            
            print(f"    Group {gi} ({gs} params): MC {n_permutations} perms, {ne} evals")
        
        shapley.update(sv)
        total_evals += ne
    
    return shapley, total_evals




# ============================================================
# Step 4c: One-at-a-time (naive baseline)
# ============================================================
def shapley_one_at_a_time(model, normalizer, dse_engine, workload, base_config,
                           param_list, target_vals):
    """Simple ΔCPI attribution: one param at a time, from min baseline.
    All configs = min with one param at DEFAULT — within training distribution."""
    min_config = make_min_config(base_config, param_list)
    cpi_base = predict_cpi(model, normalizer, dse_engine, workload, min_config)
    shapley = {}
    for name in param_list:
        config = min_config.copy()
        config[name] = target_vals[name]
        cpi_inc = predict_cpi(model, normalizer, dse_engine, workload, config)
        shapley[name] = cpi_base - cpi_inc  # positive = improvement
    return shapley, len(param_list)


# ============================================================
# Plotting
# ============================================================
def plot_cosine_heatmap(sim_matrix, param_list, groups, benchmark, output_dir):
    """Cosine similarity heatmap with group boundaries."""
    # Reorder params by group
    ordered_params = []
    group_boundaries = [0]
    for g in groups:
        ordered_params.extend(g)
        group_boundaries.append(len(ordered_params))
    
    # Reindex similarity matrix
    idx_map = [param_list.index(p) for p in ordered_params]
    reordered = sim_matrix[np.ix_(idx_map, idx_map)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(reordered, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
    
    # Group boundaries
    for b in group_boundaries[1:-1]:
        ax.axhline(b - 0.5, color='blue', linewidth=2)
        ax.axvline(b - 0.5, color='blue', linewidth=2)
    
    ax.set_xticks(range(len(ordered_params)))
    ax.set_xticklabels(ordered_params, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(ordered_params)))
    ax.set_yticklabels(ordered_params, fontsize=8)
    ax.set_title(f'MSCS Cosine Similarity — {benchmark}', fontsize=14)
    plt.colorbar(im, ax=ax, label='|cos(v_i, v_j)|')
    
    # Annotate group labels
    for gi, g in enumerate(groups):
        start = group_boundaries[gi]
        end = group_boundaries[gi + 1]
        mid = (start + end - 1) / 2
        ax.annotate(f'G{gi+1}', xy=(-0.5, mid), fontsize=9, color='blue',
                    fontweight='bold', ha='right', va='center',
                    xycoords=('axes fraction', 'data'),
                    annotation_clip=False)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"{benchmark}_cosine_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_shapley_comparison(sv_full, sv_pruned, sv_oaat, param_list, benchmark, output_dir):
    """Side-by-side horizontal bar chart comparing three Shapley methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Sort by full Shapley value
    sorted_params = sorted(param_list, key=lambda p: sv_full.get(p, 0), reverse=True)
    x = np.arange(len(sorted_params))
    
    vals_full = [sv_full.get(p, 0) for p in sorted_params]
    vals_pruned = [sv_pruned.get(p, 0) for p in sorted_params]
    vals_oaat = [sv_oaat.get(p, 0) for p in sorted_params]
    
    for ax, vals, title, color in [
        (axes[0], vals_full, 'Full MC Shapley (ground truth)', '#3498db'),
        (axes[1], vals_pruned, 'Pruned Shapley (ours)', '#2ecc71'),
        (axes[2], vals_oaat, 'One-at-a-time (ΔCPI)', '#e74c3c'),
    ]:
        ax.barh(x, vals, color=color, alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(sorted_params, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Shapley Value (CPI improvement)')
        ax.invert_yaxis()
    
    plt.suptitle(f'Shapley Value Comparison — {benchmark}', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{benchmark}_shapley_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_concorde_style(sv_full, sv_pruned, sv_oaat, param_list, benchmark, output_dir, base_cpi=None):
    """
    Concorde-style stacked bar chart: one bar per method, stacked by parameter contribution.
    Bottom gray = Base CPI (irreducible). Colored = parameter contributions.
    """
    methods = ['Full MC\nShapley', 'MSCS-Pruned\nShapley', 'One-at-a-time\n(ΔCPI)']
    sv_dicts = [sv_full, sv_pruned, sv_oaat]
    
    # Sort params by Full Shapley value (largest positive first)
    sorted_params = sorted(param_list, key=lambda p: sv_full.get(p, 0), reverse=True)
    
    # Separate positive and negative contributors
    pos_params = [p for p in sorted_params if sv_full.get(p, 0) >= 0]
    neg_params = [p for p in sorted_params if sv_full.get(p, 0) < 0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(methods))
    bar_width = 0.5
    
    for mi, sv in enumerate(sv_dicts):
        bottom_pos = 0.0
        
        # Draw Base CPI at bottom (gray)
        if base_cpi is not None:
            ax.bar(x[mi], base_cpi, bar_width, bottom=0,
                   color='#808080', edgecolor='white', linewidth=0.3,
                   label='Base' if mi == 0 else "")
            if base_cpi > 0.05:
                ax.text(x[mi], base_cpi / 2, 'Base',
                        ha='center', va='center', fontsize=6,
                        fontweight='bold', color='white')
            bottom_pos = base_cpi
        
        bottom_neg = 0.0
        
        for pname in pos_params:
            val = sv.get(pname, 0)
            if val >= 0:
                color = PARAM_COLORS.get(pname, '#cccccc')
                ax.bar(x[mi], val, bar_width, bottom=bottom_pos, color=color,
                       edgecolor='white', linewidth=0.3,
                       label=pname if mi == 0 else "")
                if abs(val) > 0.03:
                    ax.text(x[mi], bottom_pos + val/2, pname,
                            ha='center', va='center', fontsize=5.5,
                            fontweight='bold', color='white')
                bottom_pos += val
            else:
                color = PARAM_COLORS.get(pname, '#cccccc')
                ax.bar(x[mi], val, bar_width, bottom=bottom_neg, color=color,
                       edgecolor='white', linewidth=0.3)
                if abs(val) > 0.03:
                    ax.text(x[mi], bottom_neg + val/2, pname,
                            ha='center', va='center', fontsize=5.5,
                            fontweight='bold', color='white')
                bottom_neg += val
        
        for pname in neg_params:
            val = sv.get(pname, 0)
            color = PARAM_COLORS.get(pname, '#cccccc')
            ax.bar(x[mi], val, bar_width, bottom=bottom_neg, color=color,
                   edgecolor='white', linewidth=0.3,
                   label=pname if mi == 0 else "")
            if abs(val) > 0.03:
                ax.text(x[mi], bottom_neg + val/2, pname,
                        ha='center', va='center', fontsize=5.5,
                        fontweight='bold', color='white')
            bottom_neg += val
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('CPI', fontsize=12)
    ax.set_title(f'CPI Attribution by Parameter — {benchmark}', fontsize=13)
    
    # Legend outside
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=7, ncol=1, frameon=True)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"{benchmark}_shapley_concorde.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="605.mcf_s")
    parser.add_argument("--n-permutations", type=int, default=200,
                        help="MC Shapley permutations (Concorde uses 200)")
    parser.add_argument("--cos-threshold", type=float, default=0.3,
                        help="Cosine similarity threshold for grouping")
    parser.add_argument("--max-group-size", type=int, default=999,
                        help="Max group size (default: no cap)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    benchmark = args.benchmark
    # Extract short name: "605.mcf_s" -> "mcf", "508.namd_r" -> "namd"
    bname = benchmark.split(".")[-1]  # "mcf_s" or "namd_r"
    for suffix in ("_s", "_r"):
        bname = bname.replace(suffix, "")  # "mcf" or "namd"
    output_dir = os.path.join(args.output_dir, benchmark)
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(DSE_DIR)

    # Load model
    model, gem_norm, dse_engine = load_model(benchmark)

    print(f"\n{'='*70}")
    print(f"Shapley Value Pruning — {benchmark}")
    print(f"{'='*70}")

    # Get list of tunable params (hardcoded, independent of config_dse)
    param_list = [name for name in SHAPLEY_PARAMS if name in DEFAULT_CONFIG]
    d = len(param_list)
    print(f"\n  Parameters: {d}")
    print(f"  Full Shapley: {d}! = {math.factorial(d):.2e} permutations")

    # ---- Step 1: MSCS finite-diff vectors ----
    print(f"\n  Step 1: Computing MSCS finite-diff vectors (12-dim, DEFAULT→min)...")
    t0 = time.time()
    vectors, target_vals, vector_info = compute_mscs_vectors(
        model, gem_norm, dse_engine, benchmark, DEFAULT_CONFIG, param_list)
    t1 = time.time()
    
    # Filter to params that have vectors
    param_list = [p for p in param_list if p in vectors]
    d = len(param_list)
    print(f"    {d} params with valid range, {t1-t0:.1f}s")
    print(f"    Vectors (12-dim, DEFAULT→min perturbation):")
    for name in param_list:
        v = vectors[name]
        norm = np.linalg.norm(v)
        info = vector_info.get(name, "")
        print(f"      {name:<20} norm={norm:.4f}  {info:<28} "
              f"FP=[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f},{v[3]:+.3f}] "
              f"BP=[{v[4]:+.3f},{v[5]:+.3f},{v[6]:+.3f},{v[7]:+.3f}] "
              f"SP=[{v[8]:+.3f},{v[9]:+.3f},{v[10]:+.3f},{v[11]:+.3f}]")

    # ---- Step 2: Cosine similarity matrix ----
    print(f"\n  Step 2: Computing pairwise cosine similarity...")
    sim_matrix = compute_cosine_matrix(vectors, param_list)
    n_pairs = d * (d - 1) // 2
    print(f"    {n_pairs} pairs computed")

    # ---- Step 3: Grouping ----
    print(f"\n  Step 3: Grouping (threshold={args.cos_threshold})...")
    groups = group_params(sim_matrix, param_list, threshold=args.cos_threshold,
                          max_group_size=args.max_group_size)
    
    print(f"    {len(groups)} groups:")
    total_pruned_cost = 0
    for gi, g in enumerate(groups):
        cost = math.factorial(len(g))
        total_pruned_cost += cost
        print(f"      G{gi+1} ({len(g)} params, {len(g)}!={cost}): {g}")
    
    max_group = max(len(g) for g in groups)
    print(f"    Max group size: {max_group}")
    print(f"    Pruned cost: {total_pruned_cost:.2e}  (vs full {math.factorial(d):.2e})")
    print(f"    Reduction: {math.factorial(d) / total_pruned_cost:.1f}x")

    # Plot heatmap
    plot_cosine_heatmap(sim_matrix, param_list, groups, bname, output_dir)

    # ---- Step 4a: Full MC Shapley (ground truth) ----
    json_path = os.path.join(output_dir, f"{bname}_shapley_results.json")
    sv_full = None
    ne_full = 0
    t_full = 0
    if os.path.exists(json_path):
        import json
        with open(json_path) as f:
            prev = json.load(f)
        if ("shapley_full" in prev
                and set(prev["shapley_full"].keys()) == set(param_list)):
            sv_full = prev["shapley_full"]
            ne_full = prev["metrics"]["full"]["evals"]
            t_full = prev["metrics"]["full"]["time_s"]
            print(f"\n  Step 4a: Loaded Full MC Shapley from cache")
            print(f"    {ne_full} evals, {t_full}s (cached)")

    if sv_full is None:
        print(f"\n  Step 4a: Full MC Shapley ({args.n_permutations} permutations, no cache)...")
        t0 = time.time()
        sv_full, ne_full = shapley_full_mc(
            model, gem_norm, dse_engine, benchmark, DEFAULT_CONFIG,
            param_list, target_vals, n_permutations=args.n_permutations)
        t_full = time.time() - t0
        print(f"    {ne_full} evals, {t_full:.1f}s")

    # ---- Step 4b: Pruned Shapley ----
    print(f"\n  Step 4b: Pruned Shapley (exact within groups)...")
    t0 = time.time()
    sv_pruned, ne_pruned = shapley_pruned(
        model, gem_norm, dse_engine, benchmark, DEFAULT_CONFIG,
        groups, target_vals, param_list)
    t_pruned = time.time() - t0
    print(f"    {ne_pruned} unique evals, {t_pruned:.1f}s")

    # ---- Step 4b2: Matched-budget MC (same eval count as pruned) ----
    n_perms_matched = max(1, ne_pruned // (d + 1))
    ne_matched_actual = n_perms_matched * (d + 1)
    print(f"\n  Step 4b2: Matched MC ({n_perms_matched} perms = {ne_matched_actual} evals, matching pruned {ne_pruned})...")
    t0 = time.time()
    sv_matched, ne_matched = shapley_full_mc(
        model, gem_norm, dse_engine, benchmark, DEFAULT_CONFIG,
        param_list, target_vals, n_permutations=n_perms_matched, seed=42)
    t_matched = time.time() - t0
    print(f"    {ne_matched} evals, {t_matched:.1f}s")

    # ---- Step 4c: One-at-a-time ----
    print(f"\n  Step 4c: One-at-a-time ΔCPI...")
    t0 = time.time()
    sv_oaat, ne_oaat = shapley_one_at_a_time(
        model, gem_norm, dse_engine, benchmark, DEFAULT_CONFIG,
        param_list, target_vals)
    t_oaat = time.time() - t0
    print(f"    {ne_oaat} evals, {t_oaat:.1f}s")

    # ---- Step 5: Comparison ----
    print(f"\n  {'='*70}")
    print(f"  Results:")
    print(f"  {'='*70}")

    # Print Shapley values side by side
    print(f"\n    {'Param':<20} {'Full MC':>10} {'Pruned':>10} {'One-at-a':>10}")
    print(f"    {'─'*50}")
    sorted_params = sorted(param_list, key=lambda p: sv_full.get(p, 0), reverse=True)
    for name in sorted_params:
        print(f"    {name:<20} {sv_full.get(name,0):>+10.4f} "
              f"{sv_pruned.get(name,0):>+10.4f} {sv_oaat.get(name,0):>+10.4f}")

    # Spearman rank correlation
    full_vals = [sv_full[p] for p in param_list]
    pruned_vals = [sv_pruned[p] for p in param_list]
    matched_vals = [sv_matched[p] for p in param_list]
    oaat_vals = [sv_oaat[p] for p in param_list]

    rho_pruned, p_pruned = spearmanr(full_vals, pruned_vals)
    rho_matched, p_matched = spearmanr(full_vals, matched_vals)
    rho_oaat, p_oaat = spearmanr(full_vals, oaat_vals)

    # MSE
    full_arr = np.array(full_vals)
    pruned_arr = np.array(pruned_vals)
    matched_arr = np.array(matched_vals)
    oaat_arr = np.array(oaat_vals)
    mse_pruned = np.mean((full_arr - pruned_arr) ** 2)
    mse_matched = np.mean((full_arr - matched_arr) ** 2)
    mse_oaat = np.mean((full_arr - oaat_arr) ** 2)

    pruned_wins = rho_pruned > rho_matched
    print(f"\n    {'Method':<25} {'Spearman ρ':>12} {'MSE':>12} {'Evals':>10} {'Time':>10}")
    print(f"    {'─'*70}")
    print(f"    {'Full MC Shapley':<25} {'1.000':>12} {'0':>12} {ne_full:>10} {t_full:>9.1f}s")
    print(f"    {'Pruned Shapley (ours)':<25} {rho_pruned:>12.3f} {mse_pruned:>12.6f} {ne_pruned:>10} {t_pruned:>9.1f}s")
    print(f"    {'Matched MC':<25} {rho_matched:>12.3f} {mse_matched:>12.6f} {ne_matched:>10} {t_matched:>9.1f}s")
    print(f"    {'One-at-a-time (ΔCPI)':<25} {rho_oaat:>12.3f} {mse_oaat:>12.6f} {ne_oaat:>10} {t_oaat:>9.1f}s")
    print(f"\n    >>> Pruned {'WINS' if pruned_wins else 'LOSES'} vs Matched MC (ρ {rho_pruned:.3f} vs {rho_matched:.3f})")

    # Compute base CPI values for plotting and JSON
    cpi_default = predict_cpi(model, gem_norm, dse_engine, benchmark, DEFAULT_CONFIG)
    min_config = make_min_config(DEFAULT_CONFIG, param_list)
    cpi_all_min = predict_cpi(model, gem_norm, dse_engine, benchmark, min_config)

    # Plot comparison
    plot_shapley_comparison(sv_full, sv_pruned, sv_oaat, param_list, bname, output_dir)
    plot_concorde_style(sv_full, sv_pruned, sv_oaat, param_list, bname, output_dir, base_cpi=float(cpi_default))

    # Save all results to JSON
    import json
    results = {
        "benchmark": benchmark,
        "config": {k: v for k, v in DEFAULT_CONFIG.items()},
        "params": param_list,
        "n_params": d,
        "cpi_default": round(float(cpi_default), 6),
        "cpi_all_min": round(float(cpi_all_min), 6),
        "cos_threshold": args.cos_threshold,
        "n_permutations": args.n_permutations,
        "groups": [[str(p) for p in g] for g in groups],
        "n_groups": len(groups),
        "max_group_size": max(len(g) for g in groups),
        "group_sizes": [len(g) for g in groups],
        "full_shapley_cost": math.factorial(d),
        "pruned_cost": sum(math.factorial(len(g)) for g in groups),
        "shapley_full": {k: float(v) for k, v in sv_full.items()},
        "shapley_pruned": {k: float(v) for k, v in sv_pruned.items()},
        "shapley_matched": {k: float(v) for k, v in sv_matched.items()},
        "shapley_oaat": {k: float(v) for k, v in sv_oaat.items()},
        "metrics": {
            "full":    {"evals": ne_full,    "time_s": round(t_full, 1),    "rho": 1.0,                          "mse": 0.0},
            "pruned":  {"evals": ne_pruned,  "time_s": round(t_pruned, 1),  "rho": round(float(rho_pruned), 4),  "mse": round(float(mse_pruned), 8)},
            "matched": {"evals": ne_matched, "time_s": round(t_matched, 1), "rho": round(float(rho_matched), 4), "mse": round(float(mse_matched), 8)},
            "oaat":    {"evals": ne_oaat,    "time_s": round(t_oaat, 1),    "rho": round(float(rho_oaat), 4),     "mse": round(float(mse_oaat), 8)},
        },
        "pruned_wins": bool(pruned_wins),
        "cosine_matrix": sim_matrix.tolist(),
        "shapley_mode": "default_to_min_v2",
        "note": "pruned uses DEFAULT context for outside-group params",
        "reference": "DEFAULT_CONFIG (ARM N1)",
        "min_values": {name: get_param_min(name) for name in param_list},
        "target_values": {name: target_vals[name] for name in param_list},
    }
    json_path = os.path.join(output_dir, f"{bname}_shapley_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()