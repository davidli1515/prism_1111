import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES,
    NUM_STAGES, NUM_STATES,
)


class MLPBlock(nn.Module):
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


class GEMModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(MLPBlock(prev_dim, h_dim, dropout))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, NUM_STAGES * NUM_STATES)

    def forward(self, x):
        h = self.backbone(x)
        logits = self.head(h)
        logits = logits.view(-1, NUM_STAGES, NUM_STATES)
        probs = F.softmax(logits, dim=-1)
        return probs.view(-1, NUM_STAGES * NUM_STATES)


class ConcordeModel(nn.Module):
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
        return self.head(h).squeeze(-1)


def compute_cpi_from_mscs(mscs_pred, w_min, n_window, eps=1e-8):
    fp_vals   = mscs_pred[:, FP_INDICES]
    bp_vals   = mscs_pred[:, BP_INDICES]
    sp_vals   = mscs_pred[:, SP_INDICES]

    w = w_min.unsqueeze(1)
    n = n_window.unsqueeze(1)


    base_vals = mscs_pred[:, BASE_INDICES]
    base_vals = torch.clamp(base_vals, min=0.01)
    cpi_per_stage = 1.0 / (base_vals * w + eps)
    cpi_per_stage = torch.clamp(cpi_per_stage, max=100.0)

    return cpi_per_stage


def compute_cpi_mean(mscs_pred, w_min, n_window=None, eps=1e-8):
    if n_window is None:
        n_window = torch.ones_like(w_min)
    cpi_per_stage = compute_cpi_from_mscs(mscs_pred, w_min, n_window, eps)
    return cpi_per_stage.mean(dim=1)


def loss_task(mscs_pred, mscs_target):
    return F.mse_loss(mscs_pred, mscs_target)


def loss_const(mscs_pred, w_min, cpi_golden, eps=1e-8):
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi_per_stage = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    rel_err = torch.abs(cpi_per_stage - cpi_golden.unsqueeze(1)) / (torch.abs(cpi_golden.unsqueeze(1)) + eps)
    return rel_err.mean()


def loss_mono(mscs_pred):
    fp = mscs_pred[:, FP_INDICES]
    sp = mscs_pred[:, SP_INDICES]
    bp = mscs_pred[:, BP_INDICES]


    loss_fp = F.relu(fp[:, 1:] - fp[:, :-1]).sum(dim=1).mean()


    loss_sp = F.relu(sp[:, 1:] - sp[:, :-1]).sum(dim=1).mean()


    loss_bp = F.relu(bp[:, :-1] - bp[:, 1:]).sum(dim=1).mean()

    return loss_fp + loss_sp + loss_bp


def gem_total_loss(mscs_pred, mscs_target, w_min, cpi_golden,
                   lambda_const=1.0, lambda_mono=1.0):
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