"""
GEM Model and Loss Functions

GEMModel:  MLP that predicts 16-dim MSCS vector with per-stage softmax.
ConcordeModel: Baseline MLP that predicts scalar CPI.
GEMLoss: Architecture-constrained loss = L_task + λ1·L_const + λ2·L_mono.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES,
    NUM_STAGES, NUM_STATES,
)


# ============================================================
# MLP Building Block
# ============================================================

class MLPBlock(nn.Module):
    """Linear → BatchNorm → ReLU → Dropout"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# GEM Model
# ============================================================

class GEMModel(nn.Module):
    """
    MLP that outputs a 16-dim MSCS vector.
    Per-stage softmax ensures [Base, FP, BP, SP] sums to 1 for each stage.
    
    Input:  (batch, input_dim)   — Concorde features + D3 features
    Output: (batch, 16)          — MSCS proportions
    """
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(MLPBlock(prev_dim, h_dim, dropout))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, NUM_STAGES * NUM_STATES)  # 16

    def forward(self, x):
        h = self.backbone(x)
        logits = self.head(h)                          # (batch, 16)
        logits = logits.view(-1, NUM_STAGES, NUM_STATES)  # (batch, 4, 4)
        probs = F.softmax(logits, dim=-1)              # per-stage softmax
        return probs.view(-1, NUM_STAGES * NUM_STATES)    # (batch, 16)


# ============================================================
# Concorde Baseline Model
# ============================================================

class ConcordeModel(nn.Module):
    """
    Baseline MLP that predicts scalar CPI (Concorde-style).
    
    Input:  (batch, input_dim)   — Concorde features only
    Output: (batch, 1)           — predicted CPI
    """
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(MLPBlock(prev_dim, h_dim, dropout))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h).squeeze(-1)  # (batch,)


# ============================================================
# CPI Computation from MSCS
# ============================================================

def compute_cpi_from_mscs(mscs_pred, w_min, n_window, eps=1e-8):
    """
    Compute per-stage CPI from predicted MSCS proportions.
    
    Standard formula (Equation 3 in GEM paper):
        CPI_s = 1/W_min + (FP_s + BP_s + SP_s) / (N_window × W_min)
    
    Since the model predicts normalized proportions (sum to 1),
    we first recover raw stall slots:
        FP_raw = fp_s × N_window × W_min   (and similarly for BP, SP)
    
    Substituting:
        CPI_s = 1/W_min + (fp_s + bp_s + sp_s) × N_window × W_min
                / (N_window × W_min)
        CPI_s = 1/W_min + (fp_s + bp_s + sp_s)
    
    Wait — that only works if N_window in the formula equals
    N_cycles (not N_instructions). Let's derive properly:
    
    Equivalently: CPI_s = 1 / (base_s × W_min)
    (since base_s = 1 - fp_s - bp_s - sp_s, and base_s × W_min = IPC_s)
    
    Args:
        mscs_pred: (batch, 16) predicted MSCS proportions (normalized, sum=1)
        w_min:     (batch,)    minimum pipeline width
        n_window:  (batch,)    number of instructions in the window
        eps:       numerical stability
        
    Returns:
        cpi_per_stage: (batch, 4) CPI for [Decode, Dispatch, Issue, Commit]
    """
    fp_vals   = mscs_pred[:, FP_INDICES]    # (batch, 4)
    bp_vals   = mscs_pred[:, BP_INDICES]    # (batch, 4)
    sp_vals   = mscs_pred[:, SP_INDICES]    # (batch, 4)

    w = w_min.unsqueeze(1)                  # (batch, 1)
    n = n_window.unsqueeze(1)               # (batch, 1)

    # Recover raw stall slots from proportions:
    #   total_slots_per_stage = N_cycles × W_min
    #   But we don't know N_cycles directly. Instead use the identity:
    #   total_slots = N_instructions / base_s  (since base_s = N_instr / total_slots)
    #
    # Standard formula: CPI_s = 1/W_min + (FP_raw + BP_raw + SP_raw) / (N_instr × W_min)
    # With FP_raw = fp_s × total_slots = fp_s × n / base_s ... gets circular.
    #
    # Simplest equivalent: CPI_s = 1 / (base_s × W_min)
    # base_vals = mscs_pred[:, BASE_INDICES]  # (batch, 4)
    # cpi_per_stage = 1.0 / (base_vals * w + eps)

    base_vals = mscs_pred[:, BASE_INDICES]
    base_vals = torch.clamp(base_vals, min=0.01)
    cpi_per_stage = 1.0 / (base_vals * w + eps)
    cpi_per_stage = torch.clamp(cpi_per_stage, max=100.0)

    return cpi_per_stage


def compute_cpi_mean(mscs_pred, w_min, n_window=None, eps=1e-8):
    """
    Average CPI across stages — used for final CPI prediction.
    
    Note: Due to throughput conservation, all stages should yield
    the same CPI. Averaging reduces noise from prediction errors.
    """
    if n_window is None:
        n_window = torch.ones_like(w_min)  # placeholder, not used in computation
    cpi_per_stage = compute_cpi_from_mscs(mscs_pred, w_min, n_window, eps)
    return cpi_per_stage.mean(dim=1)  # (batch,)


# ============================================================
# Loss Functions
# ============================================================

def loss_task(mscs_pred, mscs_target):
    """
    L_task: MSE between predicted and ground-truth MSCS vectors.
    
    Args:
        mscs_pred:   (batch, 16)
        mscs_target: (batch, 16)
    """
    return F.mse_loss(mscs_pred, mscs_target)


def loss_const(mscs_pred, w_min, cpi_golden, eps=1e-8):
    """
    L_const: Pipeline throughput conservation.
    Penalizes relative deviation of per-stage CPI from golden CPI.
    
    L_const = (1/|S|) × Σ_s |CPI_s − CPI_golden| / |CPI_golden|
    
    Args:
        mscs_pred:  (batch, 16)
        w_min:      (batch,)
        cpi_golden: (batch,)
    """
    base_vals = mscs_pred[:, BASE_INDICES]                        # (batch, 4)
    cpi_per_stage = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)  # (batch, 4)
    rel_err = torch.abs(cpi_per_stage - cpi_golden.unsqueeze(1)) / (torch.abs(cpi_golden.unsqueeze(1)) + eps)
    return rel_err.mean()


def loss_mono(mscs_pred):
    """
    L_mono: Bottleneck visibility monotonicity.
    From Decode → Dispatch → Issue → Commit:
      - FP (Frontend Pressure) must decrease
      - SP (Squash Pressure) must decrease
      - BP (Backend Pressure) must increase
    
    Uses softplus instead of hinge loss:
      softplus(x) = log(1 + exp(x))
    Unlike hinge loss (gradient=0 when satisfied), softplus always provides
    gradient signal, encouraging larger monotonicity margins. This avoids
    the barrier effect of hinge loss while maintaining smooth optimization.
    
    Args:
        mscs_pred: (batch, 16)
    """
    fp = mscs_pred[:, FP_INDICES]  # (batch, 4) across stages
    sp = mscs_pred[:, SP_INDICES]  # (batch, 4)
    bp = mscs_pred[:, BP_INDICES]  # (batch, 4)

    # FP should decrease: penalize FP[i+1] >= FP[i]
    loss_fp = F.softplus(fp[:, 1:] - fp[:, :-1]).sum(dim=1).mean()

    # SP should decrease: penalize SP[i+1] >= SP[i]
    loss_sp = F.softplus(sp[:, 1:] - sp[:, :-1]).sum(dim=1).mean()

    # BP should increase: penalize BP[i] >= BP[i+1]
    loss_bp = F.softplus(bp[:, :-1] - bp[:, 1:]).sum(dim=1).mean()

    return loss_fp + loss_sp + loss_bp


def gem_total_loss(mscs_pred, mscs_target, w_min, cpi_golden,
                   lambda_const=1.0, lambda_mono=1.0):
    """
    L_total = L_task + λ1·L_const + λ2·L_mono
    
    Returns:
        total_loss, dict of individual losses for logging
    """
    l_task  = loss_task(mscs_pred, mscs_target)
    l_const = loss_const(mscs_pred, w_min, cpi_golden)
    l_mono  = loss_mono(mscs_pred)

    total = l_task + lambda_const * l_const + lambda_mono * l_mono

    return total, {
        "L_task":  l_task.item(),
        "L_const": l_const.item(),
        "L_mono":  l_mono.item(),
        "L_total": total.item(),
    }