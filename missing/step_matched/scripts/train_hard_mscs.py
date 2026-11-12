"""
Hard-constraint MSCS ablation (H): one shared Base for all 4 stages, per-stage FP/BP/SP =
(1 - Base) * softmax over 3 logits. Cross-stage Base equality holds by construction, so
L_const_self is identically 0 and unlabeled rows give no gradient.

Everything else is train_exp4_lambda_sens.py unchanged (same data split, recipe, milestones,
losses, selection); only GEMModel is replaced. Backbone init matches GEMModel for the same
seed (the head is built after the backbone).
  --variant v3 -> H-U (labeled + 18k unlabeled rows, same steps/batches as V3)
  --variant v2 -> H-L (labeled rows only, as V2)

Usage (same flags as train_exp4_lambda_sens.py, --out required to keep results here):
    python train_hard_mscs.py --variant v3 --ratio 0.05 --lam-task 10 --lam-const 1 --lam-mono 0 \
        --input concorde --seed 42 --gpu 0 --out DIR
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp4")
sys.path.insert(1, "/scratch/gem5-TAO/micro_exp/exp1")
import train_exp4_lambda_sens as T
from model import MLPBlock
from config import NUM_STAGES


class HardMSCSModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers.append(MLPBlock(prev, h, dropout)); prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1 + NUM_STAGES * 3)          # shared Base + 4 x (FP, BP, SP)

    def forward(self, x):
        z = self.head(self.backbone(x))
        base = torch.sigmoid(z[:, :1])                             # (batch, 1)
        stall = F.softmax(z[:, 1:].view(-1, NUM_STAGES, 3), dim=-1) * (1 - base).unsqueeze(-1)
        out = torch.cat([base.unsqueeze(1).expand(-1, NUM_STAGES, 1), stall], dim=-1)
        return out.reshape(-1, NUM_STAGES * 4)                     # [Base, FP, BP, SP] per stage


if __name__ == "__main__":
    if "--out" not in sys.argv:
        sys.exit("--out is required")
    T.GEMModel = HardMSCSModel
    T.main()
