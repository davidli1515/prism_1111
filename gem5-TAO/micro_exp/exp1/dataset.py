"""
GEM Dataset Loading

Loads per-benchmark .npy files, extracts features/labels/CPI/W_min,
and builds PyTorch DataLoaders for training and evaluation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import (
    CONCORDE_FEAT_END, D3_FEAT_END, MSCS_START, MSCS_END,
    WMIN_INDICES, GEM_INPUT_DIM, CONCORDE_INPUT_DIM,
    BASE_INDICES,
    TRAIN_RATIO, BATCH_SIZE, SEED,
)


# ============================================================
# Data Loading
# ============================================================

def load_benchmark(benchmark_name, gem_dir, concord_dir=None):
    """
    Load one benchmark's data.
    
    All data comes from gem_data.npy (CPI, features, MSCS are all in gem_data).
    concord_dir is kept for compatibility but not required.
    
    Files:
        {gem_dir}/{benchmark_name}.npy — shape (N, 4697)
    
    Returns dict with numpy arrays:
        gem_features:    (N, 4680)  — Concorde + D3 features
        concord_features:(N, 3873)  — Concorde features only (subset of gem)
        mscs_target:     (N, 16)    — ground-truth MSCS
        cpi_golden:      (N,)       — ground-truth CPI
        w_min:           (N,)       — minimum pipeline width
    """
    gem_path = os.path.join(gem_dir, f"{benchmark_name}.npy")
    gem_raw = np.load(gem_path).astype(np.float32)

    # Handle 1D case (single sample)
    if gem_raw.ndim == 1:
        gem_raw = gem_raw[np.newaxis, :]

    # Extract fields — all from gem_data
    gem_features     = gem_raw[:, :D3_FEAT_END]        # (N, 4681)
    concord_features = gem_raw[:, :CONCORDE_FEAT_END]  # (N, 3873)
    mscs_target      = gem_raw[:, MSCS_START:MSCS_END] # (N, 16)

    # Compute W_min = min(fetchW, decodeW, renameW, dispatchW, issueW, commitW)
    widths = gem_raw[:, WMIN_INDICES]                   # (N, 6)
    w_min  = widths.min(axis=1)                         # (N,)

    # Compute golden CPI from MSCS: CPI = 1 / (Base_commit × W_min)
    # Base_commit is at MSCS index 12 (BASE_INDICES[3])
    base_commit = mscs_target[:, BASE_INDICES[3]]       # (N,)
    cpi_golden  = 1.0 / (base_commit * w_min + 1e-8)   # (N,)

    return {
        "gem_features":     gem_features,
        "concord_features": concord_features,
        "mscs_target":      mscs_target,
        "cpi_golden":       cpi_golden,
        "w_min":            w_min,
        "n_samples":        gem_raw.shape[0],
    }

    return {
        "gem_features":     gem_features,
        "concord_features": concord_features,
        "mscs_target":      mscs_target,
        "cpi_golden":       cpi_golden,
        "w_min":            w_min,
        "n_samples":        gem_raw.shape[0],
    }


def discover_benchmarks(gem_dir):
    """Auto-discover benchmark names from .npy files in gem_dir."""
    names = []
    for f in sorted(os.listdir(gem_dir)):
        if f.endswith(".npy"):
            names.append(f[:-4])  # strip .npy
    return names


def load_all_benchmarks(gem_dir, concord_dir, benchmark_names=None):
    """
    Load all benchmarks.
    
    Args:
        gem_dir:         path to gem .npy files
        concord_dir:     path to concord .npy files
        benchmark_names: list of names, or None to auto-discover
        
    Returns:
        dict: {"name": loaded_data_dict, ...}
    """
    if benchmark_names is None:
        benchmark_names = discover_benchmarks(gem_dir)
    
    print(f"  Found {len(benchmark_names)} benchmarks")
    all_data = {}
    total_samples = 0
    for name in benchmark_names:
        print(f"    Loading {name} ...", end=" ")
        data = load_benchmark(name, gem_dir, concord_dir)
        print(f"{data['n_samples']} samples")
        all_data[name] = data
        total_samples += data["n_samples"]
    
    print(f"  Total: {total_samples:,} samples across {len(all_data)} benchmarks")
    return all_data


# ============================================================
# Normalization
# ============================================================

class FeatureNormalizer:
    """Per-feature standardization (zero mean, unit variance)."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, features):
        """Compute mean/std from training data. features: (N, D)"""
        self.mean = features.mean(axis=0)
        self.std  = features.std(axis=0)
        self.std[self.std < 1e-8] = 1.0  # avoid division by zero

    def transform(self, features):
        """Apply normalization."""
        return (features - self.mean) / self.std

    def fit_transform(self, features):
        self.fit(features)
        return self.transform(features)


# ============================================================
# PyTorch Dataset
# ============================================================

class GEMDataset(Dataset):
    """
    Dataset for GEM training.
    Each sample: (gem_features, concord_features, mscs_target, cpi_golden, w_min)
    """
    def __init__(self, gem_features, concord_features, mscs_target,
                 cpi_golden, w_min):
        self.gem_features     = torch.from_numpy(gem_features)
        self.concord_features = torch.from_numpy(concord_features)
        self.mscs_target      = torch.from_numpy(mscs_target)
        self.cpi_golden       = torch.from_numpy(cpi_golden)
        self.w_min            = torch.from_numpy(w_min)

    def __len__(self):
        return self.gem_features.shape[0]

    def __getitem__(self, idx):
        return {
            "gem_features":     self.gem_features[idx],
            "concord_features": self.concord_features[idx],
            "mscs_target":      self.mscs_target[idx],
            "cpi_golden":       self.cpi_golden[idx],
            "w_min":            self.w_min[idx],
        }


# ============================================================
# Experiment 1: In-Distribution Split
# ============================================================

def create_exp1_dataloaders(all_benchmark_data, train_ratio=TRAIN_RATIO,
                            batch_size=BATCH_SIZE, seed=SEED):
    """
    Exp 1: In-distribution accuracy.
    For EVERY benchmark, split samples into train (15/16) and test (1/16).
    Combine all train splits → train set, all test splits → test set.
    
    Args:
        all_benchmark_data: dict from load_all_benchmarks()
        
    Returns:
        train_loader, test_loader, gem_normalizer, concord_normalizer,
        per_benchmark_test_indices (for per-benchmark evaluation)
    """
    rng = np.random.RandomState(seed)

    train_gem, train_con, train_mscs, train_cpi, train_wmin = [], [], [], [], []
    test_gem,  test_con,  test_mscs,  test_cpi,  test_wmin  = [], [], [], [], []
    per_benchmark_test = {}  # {name: (start_idx, end_idx)} in test set

    test_offset = 0
    for name, data in sorted(all_benchmark_data.items()):
        n = data["n_samples"]
        indices = rng.permutation(n)
        split = int(n * train_ratio)

        tr_idx = indices[:split]
        te_idx = indices[split:]

        train_gem.append(data["gem_features"][tr_idx])
        train_con.append(data["concord_features"][tr_idx])
        train_mscs.append(data["mscs_target"][tr_idx])
        train_cpi.append(data["cpi_golden"][tr_idx])
        train_wmin.append(data["w_min"][tr_idx])

        test_gem.append(data["gem_features"][te_idx])
        test_con.append(data["concord_features"][te_idx])
        test_mscs.append(data["mscs_target"][te_idx])
        test_cpi.append(data["cpi_golden"][te_idx])
        test_wmin.append(data["w_min"][te_idx])

        n_test = len(te_idx)
        per_benchmark_test[name] = (test_offset, test_offset + n_test)
        test_offset += n_test

    # Concatenate
    train_gem  = np.concatenate(train_gem)
    train_con  = np.concatenate(train_con)
    train_mscs = np.concatenate(train_mscs)
    train_cpi  = np.concatenate(train_cpi)
    train_wmin = np.concatenate(train_wmin)

    test_gem  = np.concatenate(test_gem)
    test_con  = np.concatenate(test_con)
    test_mscs = np.concatenate(test_mscs)
    test_cpi  = np.concatenate(test_cpi)
    test_wmin = np.concatenate(test_wmin)

    # Normalize features (fit on train only)
    gem_normalizer = FeatureNormalizer()
    train_gem = gem_normalizer.fit_transform(train_gem)
    test_gem  = gem_normalizer.transform(test_gem)

    concord_normalizer = FeatureNormalizer()
    train_con = concord_normalizer.fit_transform(train_con)
    test_con  = concord_normalizer.transform(test_con)

    # Build datasets & loaders
    train_ds = GEMDataset(train_gem, train_con, train_mscs, train_cpi, train_wmin)
    test_ds  = GEMDataset(test_gem,  test_con,  test_mscs,  test_cpi,  test_wmin)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"\n  Train samples: {len(train_ds)}")
    print(f"  Test samples:  {len(test_ds)}")
    print(f"  Benchmarks in test: {list(per_benchmark_test.keys())}")

    return (train_loader, test_loader,
            gem_normalizer, concord_normalizer,
            per_benchmark_test)