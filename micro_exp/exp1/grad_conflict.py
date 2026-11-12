"""
Gradient Conflict Analysis: measure cosine similarity between
∇L_task, ∇L_const, ∇L_mono during training.

Usage: python grad_conflict.py --gpu 0

Output: grad_conflict_results.json
"""
import os, sys, json, numpy as np, torch, torch.optim as optim
from collections import defaultdict

from config import (
    GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, PATIENCE,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES, SEED,
)
from model import GEMModel, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, create_exp1_dataloaders


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def get_grad_vec(loss, model):
    """Get flattened gradient vector for a loss."""
    grads = torch.autograd.grad(loss, model.parameters(),
                                 retain_graph=True, allow_unused=True)
    return torch.cat([g.flatten() if g is not None
                      else torch.zeros(p.numel(), device=p.device)
                      for g, p in zip(grads, model.parameters())])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    set_seed(SEED)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    (train_loader, test_loader, _, _, _) = create_exp1_dataloaders(all_data)

    model = GEMModel(GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    print(f"Training {args.epochs} epochs with gradient conflict measurement...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        cos_tm, cos_tc, cos_cm = [], [], []
        norm_t, norm_c, norm_m = [], [], []

        for i, batch in enumerate(train_loader):
            feat     = batch["gem_features"].to(dev)
            mscs_tgt = batch["mscs_target"].to(dev)
            cpi_gold = batch["cpi_golden"].to(dev)
            w_min    = batch["w_min"].to(dev)

            mscs_pred = model(feat)
            l_t = loss_task(mscs_pred, mscs_tgt)
            l_c = loss_const(mscs_pred, w_min, cpi_gold)
            l_m = loss_mono(mscs_pred)

            # Measure gradient conflict every 20 batches
            if i % 20 == 0:
                g_task  = get_grad_vec(l_t, model)
                g_const = get_grad_vec(l_c, model)
                g_mono  = get_grad_vec(l_m, model)

                cos_tm.append(torch.nn.functional.cosine_similarity(
                    g_task, g_mono, dim=0).item())
                cos_tc.append(torch.nn.functional.cosine_similarity(
                    g_task, g_const, dim=0).item())
                cos_cm.append(torch.nn.functional.cosine_similarity(
                    g_const, g_mono, dim=0).item())

                norm_t.append(g_task.norm().item())
                norm_c.append(g_const.norm().item())
                norm_m.append(g_mono.norm().item())

            # Normal training step
            loss = l_t + LAMBDA_CONST * l_c + LAMBDA_MONO * l_m
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        rec = {
            "epoch": epoch,
            "cos_task_mono":  np.mean(cos_tm),
            "cos_task_const": np.mean(cos_tc),
            "cos_const_mono": np.mean(cos_cm),
            "norm_task":  np.mean(norm_t),
            "norm_const": np.mean(norm_c),
            "norm_mono":  np.mean(norm_m),
        }
        history.append(rec)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"cos(task,mono)={rec['cos_task_mono']:+.4f}  "
                  f"cos(task,const)={rec['cos_task_const']:+.4f}  "
                  f"cos(const,mono)={rec['cos_const_mono']:+.4f} | "
                  f"||∇task||={rec['norm_task']:.4f}  "
                  f"||∇const||={rec['norm_const']:.4f}  "
                  f"||∇mono||={rec['norm_mono']:.6f}")

    with open("grad_conflict_results.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\nSaved to grad_conflict_results.json")

    # Summary
    early = history[:10]
    late  = history[-10:]
    print(f"\n--- Summary ---")
    print(f"Early (ep 1-10):  cos(task,mono)={np.mean([r['cos_task_mono'] for r in early]):+.4f}  "
          f"||∇mono||={np.mean([r['norm_mono'] for r in early]):.6f}")
    print(f"Late  (ep {args.epochs-9}-{args.epochs}):  cos(task,mono)={np.mean([r['cos_task_mono'] for r in late]):+.4f}  "
          f"||∇mono||={np.mean([r['norm_mono'] for r in late]):.6f}")
    print(f"\nIf cos(task,mono) < 0 early → gradient conflict confirmed")
    print(f"If ||∇mono|| → 0 late → hinge loss 'shuts off' confirmed")


if __name__ == "__main__":
    main()