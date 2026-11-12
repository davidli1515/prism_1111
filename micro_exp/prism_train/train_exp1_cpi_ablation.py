"""
Exp1 CPI-supervision ablation: is the V1 -> V2 gain from direct CPI supervision
or from anchoring all four stages (throughput conservation)?

Same data / split / recipe as exp1_v3_run2 [512,256] 183k (gem_full / gem_const):
  18k dataset, exp1 row split (seed 42, 15:1), AdamW lr 1e-3 wd 0.3, batch 50k,
  1521 epochs, MultiStepLR x0.5 at [42,58,75,92]% of steps, grad clip 1.0,
  model selection = best test MAPE (4-stage mean CPI), as in exp1.
  lambda_task = 0.1 whenever a CPI term is present, 1.0 otherwise (exp1 rule).

Variants (CPI term weight = 1.0, same total weight as L_const):
  v1             L_task
  v1_cpi_commit  0.1 L_task + |CPI_commit - CPI_gt| / CPI_gt
  v1_cpi_mean    0.1 L_task + |mean_s(CPI_s) - CPI_gt| / CPI_gt
  v2_const       0.1 L_task + L_const        (each stage anchored; = exp1 gem_const)
  v2_full        0.1 L_task + L_const + 0.01 L_mono   (= exp1 gem_full)

--seed changes model init / batch order only; the train/test split is fixed (seed 42).
Reports CPI MAPE from the 4-stage mean, the Commit stage alone, each stage, and the
cross-stage Base spread.

Usage:
    python train_exp1_cpi_ablation.py --variant v1_cpi_commit --seed 42 --gpu 0
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (GEM_INPUT_DIM, DROPOUT, BENCHMARK_NAMES, SEED,
                    CONCORDE_LR, CONCORDE_WEIGHT_DECAY, CONCORDE_EPOCHS,
                    LAMBDA_CONST, LAMBDA_MONO, BASE_INDICES)
from model import GEMModel, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, create_exp1_dataloaders

DATA_GEM = "/data/micro_traindata_packed/gem"
DATA_CON = "/data/micro_traindata_packed/concord"
HIDDEN = [512, 256]
BATCH = 50000
VARIANTS = ["v1", "v1_cpi_commit", "v1_cpi_mean", "v2_const", "v2_full"]
OUT_DEFAULT = "/scratch/gem5-TAO/micro_exp/exp1/results/cpi_ablation_1003"


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def stage_cpi(mp, w_min, eps=1e-8):
    """(B,4) per-stage CPI, same clamps as model.compute_cpi_from_mscs (used for eval)."""
    base = torch.clamp(mp[:, BASE_INDICES], min=0.01)
    return torch.clamp(1.0 / (base * w_min.unsqueeze(1) + eps), max=100.0)


def rel(p, t, eps=1e-8):
    return torch.abs(p - t) / (torch.abs(t) + eps)


def cpi_term(variant, mp, w_min, cpi_gt, eps=1e-8):
    """Training-time CPI term (no clamps, like loss_const)."""
    if variant == "v1_cpi_commit":
        c = 1.0 / (mp[:, BASE_INDICES[3]] * w_min + eps)
        return rel(c, cpi_gt).mean()
    if variant == "v1_cpi_mean":
        c = (1.0 / (mp[:, BASE_INDICES] * w_min.unsqueeze(1) + eps)).mean(1)
        return rel(c, cpi_gt).mean()
    if variant in ("v2_const", "v2_full"):
        return loss_const(mp, w_min, cpi_gt)
    return None


@torch.no_grad()
def evaluate(model, X, W, Y, M):
    model.eval()
    P = torch.cat([model(X[i:i + BATCH]) for i in range(0, len(X), BATCH)])
    C = stage_cpi(P, W)                                   # (N,4)
    base = P[:, BASE_INDICES]
    out = {
        "mape_mean4": rel(C.mean(1), Y).mean().item() * 100,
        "mape_commit": rel(C[:, 3], Y).mean().item() * 100,
        "mape_per_stage": [rel(C[:, s], Y).mean().item() * 100 for s in range(4)],
        "base_spread_rel": ((base.max(1).values - base.min(1).values) / base.mean(1)).mean().item(),
        "mscs_mse": ((P - M) ** 2).mean().item(),
    }
    return out, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=CONCORDE_EPOCHS)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.variant}_s{args.seed}"
    dev = torch.device(f"cuda:{args.gpu}")

    # Fixed split (seed 42) = exp1; data kept on GPU
    data = load_all_benchmarks(DATA_GEM, DATA_CON, BENCHMARK_NAMES)
    tl, vl, _, _, bm_idx = create_exp1_dataloaders(data, seed=SEED)
    del data
    g = lambda ds, k: ds.__dict__[k].to(dev)
    Xtr, Mtr, Ytr, Wtr = (g(tl.dataset, k) for k in ["gem_features", "mscs_target", "cpi_golden", "w_min"])
    Xte, Mte, Yte, Wte = (g(vl.dataset, k) for k in ["gem_features", "mscs_target", "cpi_golden", "w_min"])

    set_seed(args.seed)
    model = GEMModel(GEM_INPUT_DIM, HIDDEN, DROPOUT).to(dev)
    opt = optim.AdamW(model.parameters(), lr=CONCORDE_LR, weight_decay=CONCORDE_WEIGHT_DECAY)
    n = len(Xtr); spe = n // BATCH                          # drop_last=True as exp1
    total = args.epochs * spe
    milestones = [int(total * f) for f in [0.42, 0.58, 0.75, 0.92]]
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.5)
    has_cpi = args.variant != "v1"
    lam_task = 0.1 if has_cpi else 1.0
    print(f"[{tag}] train={n} test={len(Xte)} steps/ep={spe} milestones={milestones} lam_task={lam_task}")

    best, best_state, best_ep, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(spe):
            b = perm[i * BATCH:(i + 1) * BATCH]
            mp = model(Xtr[b])
            loss = lam_task * loss_task(mp, Mtr[b])
            c = cpi_term(args.variant, mp, Wtr[b], Ytr[b])
            if c is not None:
                loss = loss + LAMBDA_CONST * c
            if args.variant == "v2_full":
                loss = loss + LAMBDA_MONO * loss_mono(mp)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        ev, _ = evaluate(model, Xte, Wte, Yte, Mte)
        hist.append({"epoch": ep, **{k: v for k, v in ev.items() if k != "mape_per_stage"}})
        if ev["mape_mean4"] < best:
            best, best_ep = ev["mape_mean4"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 50 == 0 or ep == 1:
            print(f"[{tag}] ep {ep:4d} mean4={ev['mape_mean4']:.2f} commit={ev['mape_commit']:.2f} "
                  f"best={best:.2f}@{best_ep} | {time.time()-t0:.0f}s", flush=True)

    last, _ = evaluate(model, Xte, Wte, Yte, Mte)
    model.load_state_dict(best_state)
    sel, C = evaluate(model, Xte, Wte, Yte, Mte)
    per_bm = {}
    for name, (s, e) in bm_idx.items():
        if s < e:
            per_bm[name] = {"mean4": rel(C[s:e].mean(1), Yte[s:e]).mean().item() * 100,
                            "commit": rel(C[s:e, 3], Yte[s:e]).mean().item() * 100}
    res = {"variant": args.variant, "seed": args.seed, "split_seed": SEED, "hidden": HIDDEN,
           "lambda_task": lam_task, "best_epoch": best_ep, "selected": sel, "last_epoch": last,
           "per_benchmark": per_bm, "train_time_s": time.time() - t0, "history": hist}
    json.dump(res, open(os.path.join(args.out, f"{tag}.json"), "w"), indent=2)
    torch.save(best_state, os.path.join(args.out, f"{tag}.pt"))
    print(f"[{tag}] DONE selected: mean4={sel['mape_mean4']:.2f} commit={sel['mape_commit']:.2f} "
          f"stages={[round(x,2) for x in sel['mape_per_stage']]} base_spread={sel['base_spread_rel']:.4f}")


if __name__ == "__main__":
    main()
