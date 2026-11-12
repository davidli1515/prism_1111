"""
GEM Training Configuration
All data indices, hyperparameters, and paths in one place.
"""
import os

# ============================================================
# DATA LAYOUT CONSTANTS
# ============================================================

# --- gem_data.npy per-sample layout (4697 dims) ---
CONCORDE_FEAT_START = 0
CONCORDE_FEAT_END   = 3873      # gem_data[0:3873]   → Concorde features
D3_FEAT_START       = 3873
D3_FEAT_END         = 4681      # gem_data[3873:4681] → D3 pressure features (808)
MSCS_START          = 4681
MSCS_END            = 4697      # gem_data[4681:4697] → MSCS (16)

# GEM model uses all features: Concorde + D3
GEM_INPUT_DIM       = D3_FEAT_END            # 4681
CONCORDE_INPUT_DIM  = CONCORDE_FEAT_END      # 3873
MSCS_DIM            = MSCS_END - MSCS_START  # 16

# --- CPI is NOT stored in the npy file ---
# Compute from MSCS: CPI = 1 / (Base_commit × W_min)
# Base_commit is at MSCS index 12 (gd[4693])

# --- concord_data.npy ---
# ⚠️ Has +1 offset bug. Do NOT use cd[3873] as CPI. Use gem_data only.

# --- Pipeline width indices within gem_data ---
# gd[3850]=fetchW, gd[3851]=decodeW, gd[3852]=renameW,
# gd[3853]=dispatchW, gd[3854]=issueW, gd[3855]=wbW(excluded), gd[3856]=commitW
WMIN_INDICES = [3850, 3851, 3852, 3853, 3854, 3856]  # exclude wbWidth

# --- MSCS index mapping (within the 16-dim vector) ---
# Layout: [Base, FP, BP, SP] × [Decode, Dispatch, Issue, Commit]
#   0: Base_dec   1: FP_dec   2: BP_dec   3: SP_dec
#   4: Base_disp  5: FP_disp  6: BP_disp  7: SP_disp
#   8: Base_iss   9: FP_iss  10: BP_iss  11: SP_iss
#  12: Base_com  13: FP_com  14: BP_com  15: SP_com
BASE_INDICES = [0, 4, 8, 12]
FP_INDICES   = [1, 5, 9, 13]
BP_INDICES   = [2, 6, 10, 14]
SP_INDICES   = [3, 7, 11, 15]
NUM_STAGES   = 4
NUM_STATES   = 4

# ============================================================
# HYPERPARAMETERS
# ============================================================

# Model architecture — same backbone for fair comparison
GEM_HIDDEN_DIMS     = [256, 128]    # same as Concorde for fair comparison
CONCORDE_HIDDEN_DIMS = [256, 128]   # Concorde paper: 2 hidden layers, 256 and 128
DROPOUT             = 0.1

# Training — GEM
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-5
BATCH_SIZE     = 256
NUM_EPOCHS     = 300
PATIENCE       = 30       # early stopping patience

# Training — Concorde (from Concorde-ISCA'25 paper)
CONCORDE_LR           = 1e-3
CONCORDE_WEIGHT_DECAY = 0.3        # AdamW weight decay
CONCORDE_BATCH_SIZE   = 50000      # paper uses 50k; adjust down if memory limited
CONCORDE_EPOCHS       = 1521       # paper value; early stopping will likely trigger first

# Loss weights
LAMBDA_CONST = 1.0    # weight for L_const (throughput conservation)
LAMBDA_MONO  = 1.0    # weight for L_mono  (bottleneck monotonicity)

# Data split
TRAIN_RATIO = 15 / 16  # within each benchmark, 15:1 split

# ============================================================
# PATHS
# ============================================================

CONCORD_DATA_DIR = "/data/micro_traindata_packed/concord"
GEM_DATA_DIR     = "/data/micro_traindata_packed/gem"

# Benchmarks: auto-discovered from gem data directory.
# Each benchmark has {benchmark_name}.npy in both concord and gem dirs.
# Example: 508.namd_r.npy
#
# To use a fixed list, uncomment and populate:
# BENCHMARK_NAMES = ["508.namd_r", "519.lbm_r", ...]
#
# To auto-discover, set to None (dataset.py will scan the directory):
BENCHMARK_NAMES = None  # auto-discover from GEM_DATA_DIR

# Output
OUTPUT_DIR = "./results/exp1"
SEED = 42