"""
Experiment 1 Ablation: Loss Function Decomposition
  All use GEM+D3 (4681 input → MSCS 16-dim output, [256,128] hidden)
  
  Variant 1: Vanilla MLP       — L_task only
  Variant 2: MLP + L_mono      — L_task + L_mono
  Variant 3: MLP + L_const     — L_task + L_const

  Compare with existing exp1 results (GEM+D3 full = L_task + L_const + L_mono)

Usage:
    python train_exp1_ablation.py --gpu 0
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader

from config import (
    GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES,
    OUTPUT_DIR, SEED,
)
from model import (
    GEMModel, compute_cpi_mean, loss_task, loss_const, loss_mono,
)
from dataset import load_all_benchmarks, create_exp1_dataloaders


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_gold = [], []
    for batch in loader:
        feat  = batch["gem_features"].to(device)
        w_min = batch["w_min"].to(device)
        mscs_pred = model(feat)
        cpi_pred = compute_cpi_mean(mscs_pred, w_min)
        all_pred.append(cpi_pred.cpu())
        all_gold.append(batch["cpi_golden"])
    all_pred = torch.cat(all_pred)
    all_gold = torch.cat(all_gold)
    return compute_mape(all_pred, all_gold), all_pred, all_gold


def per_benchmark_mape(cpi_pred, cpi_gold, benchmark_indices):
    results = {}
    for name, (start, end) in benchmark_indices.items():
        if start == end:
            continue
        results[name] = compute_mape(cpi_pred[start:end], cpi_gold[start:end])
    return results


# ============================================================
# Generic trainer with configurable loss
# ============================================================

def train_ablation(model, train_loader, test_loader, device,
                   use_const=False, use_mono=False,
                   num_epochs=NUM_EPOCHS, label=""):
    """
    Train GEM+D3 model with configurable loss components.
    Always includes L_task. Optionally adds L_const and/or L_mono.
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_mape = float("inf")
    best_state = None
    patience_counter = 0
    loss_history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch in train_loader:
            feat     = batch["gem_features"].to(device)
            mscs_tgt = batch["mscs_target"].to(device)
            cpi_gold = batch["cpi_golden"].to(device)
            w_min    = batch["w_min"].to(device)

            mscs_pred = model(feat)
            l_t = loss_task(mscs_pred, mscs_tgt)
            loss = l_t

            l_c_val, l_m_val = 0.0, 0.0
            if use_const:
                l_c = loss_const(mscs_pred, w_min, cpi_gold)
                loss = loss + LAMBDA_CONST * l_c
                l_c_val = l_c.item()
            if use_mono:
                l_m = loss_mono(mscs_pred)
                loss = loss + LAMBDA_MONO * l_m
                l_m_val = l_m.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses["L_total"] += loss.item()
            epoch_losses["L_task"]  += l_t.item()
            epoch_losses["L_const"] += l_c_val
            epoch_losses["L_mono"]  += l_m_val
            n_batches += 1

        scheduler.step()
        avg = {k: v / n_batches for k, v in epoch_losses.items()}
        test_mape, _, _ = evaluate(model, test_loader, device)

        avg["test_mape"] = test_mape
        avg["epoch"] = epoch
        loss_history.append(avg)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{label}] Epoch {epoch:3d} | "
                  f"L_total={avg['L_total']:.6f} L_task={avg['L_task']:.6f} "
                  f"L_const={avg['L_const']:.6f} L_mono={avg['L_mono']:.6f} | "
                  f"MAPE={test_mape:.2f}%")

        if test_mape < best_mape:
            best_mape = test_mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [{label}] Early stopping at epoch {epoch} "
                      f"(best MAPE={best_mape:.2f}%)")
                break

    model.load_state_dict(best_state)
    print(f"  [{label}] Best MAPE: {best_mape:.2f}%")
    return model, best_mape, loss_history


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Exp1 Ablation")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ablation_dir = os.path.join(OUTPUT_DIR, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    # Load data
    print("\nLoading benchmarks...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)

    print("\nBuilding dataloaders (15:1 split)...")
    (train_loader, test_loader, gem_norm, con_norm,
     benchmark_test_indices) = create_exp1_dataloaders(all_data)

    # Define ablation variants
    variants = [
        {"name": "vanilla",    "label": "Vanilla (L_task only)",  "use_const": False, "use_mono": False},
        {"name": "mono_only",  "label": "L_task + L_mono",        "use_const": False, "use_mono": True},
        {"name": "const_only", "label": "L_task + L_const",       "use_const": True,  "use_mono": False},
    ]

    all_results = {}

    for v in variants:
        print(f"\n{'='*60}")
        print(f"  {v['label']}")
        print(f"{'='*60}")

        set_seed(SEED)
        model = GEMModel(GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT).to(device)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        model, best_mape, history = train_ablation(
            model, train_loader, test_loader, device,
            use_const=v["use_const"], use_mono=v["use_mono"],
            label=v["name"],
        )

        # Per-benchmark evaluation
        _, pred, gold = evaluate(model, test_loader, device)
        per_bm = per_benchmark_mape(pred, gold, benchmark_test_indices)

        all_results[v["name"]] = {
            "overall_mape": best_mape,
            "per_benchmark": per_bm,
            "loss_history": history,
            "config": {
                "use_const": v["use_const"],
                "use_mono": v["use_mono"],
                "label": v["label"],
            },
        }

        # Save checkpoint
        torch.save(model.state_dict(),
                   os.path.join(ablation_dir, f"{v['name']}.pt"))

    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*70}")
    print("ABLATION RESULTS")
    print(f"{'='*70}")

    benchmarks = sorted(benchmark_test_indices.keys())
    header = f"{'Benchmark':<20}"
    for v in variants:
        header += f" {v['name']:>12}"
    print(header)
    print("-" * (20 + 13 * len(variants)))

    for bm in benchmarks:
        row = f"{bm:<20}"
        for v in variants:
            val = all_results[v["name"]]["per_benchmark"].get(bm, 0)
            row += f" {val:>11.2f}%"
        print(row)

    print("-" * (20 + 13 * len(variants)))
    row = f"{'Overall':<20}"
    for v in variants:
        row += f" {all_results[v['name']]['overall_mape']:>11.2f}%"
    print(row)

    # Save
    result_path = os.path.join(ablation_dir, "exp1_ablation_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {result_path}")


if __name__ == "__main__":
    main()