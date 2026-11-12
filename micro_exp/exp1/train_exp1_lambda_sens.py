"""
Exp1 loss-weight sensitivity:  L = lam_task * L_task + lam_const * L_const + lam_mono * L_mono

Same data / split / recipe as train_exp1_cpi_ablation.py (= exp1_v3_run2 [512,256] 18k):
row split seed 42 (fixed), AdamW 1e-3 / wd 0.3, batch 50k, 1521 epochs, step LR at
[42,58,75,92]%, grad clip 1.0, selection = best test MAPE (4-stage mean CPI).
Current paper setting: lam_task=0.1, lam_const=1, lam_mono=0.01.

Besides CPI MAPE, reports decomposition quality (the MSCS is the paper's output):
per-component MAE, dominant non-Base bottleneck accuracy per stage, cross-stage
monotonicity violation rate (any / >0.01), and cross-stage Base spread.

--input concorde uses only the first 3873 (Concorde) features, i.e. drops the 808 D3 features
(per-feature normalization, so slicing after normalization is exact); tag gets suffix _nod3.
Refuses to overwrite an existing result.

Usage:
    python train_exp1_lambda_sens.py --lam-task 1 --lam-const 1 --lam-mono 0.01 --seed 42 --gpu 0
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (GEM_INPUT_DIM, CONCORDE_INPUT_DIM, DROPOUT, BENCHMARK_NAMES, SEED,
                    CONCORDE_LR, CONCORDE_WEIGHT_DECAY, CONCORDE_EPOCHS,
                    BASE_INDICES, FP_INDICES, BP_INDICES, SP_INDICES)
from model import GEMModel, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, create_exp1_dataloaders
from train_exp1_cpi_ablation import set_seed, stage_cpi, rel, DATA_GEM, DATA_CON, HIDDEN, BATCH

OUT_DEFAULT = "/path/to/gem5/micro_exp/exp1/results/lambda_sweep"
COMP = {"Base": BASE_INDICES, "FP": FP_INDICES, "BP": BP_INDICES, "SP": SP_INDICES}


def fmt(x):
    return ("%g" % x).replace(".", "p")


@torch.no_grad()
def evaluate(model, X, W, Y, M, full=False):
    model.eval()
    P = torch.cat([model(X[i:i + BATCH]) for i in range(0, len(X), BATCH)])
    C = stage_cpi(P, W)
    out = {"mape_mean4": rel(C.mean(1), Y).mean().item() * 100,
           "mape_commit": rel(C[:, 3], Y).mean().item() * 100}
    if not full:
        return out, C
    base = P[:, BASE_INDICES]
    out["mape_per_stage"] = [rel(C[:, s], Y).mean().item() * 100 for s in range(4)]
    out["base_spread_rel"] = ((base.max(1).values - base.min(1).values) / base.mean(1)).mean().item()
    out["mae"] = {k: (P[:, I] - M[:, I]).abs().mean().item() for k, I in COMP.items()}
    out["mse"] = {k: ((P[:, I] - M[:, I]) ** 2).mean().item() for k, I in COMP.items()}
    out["dominant_acc"] = [
        (P[:, [f, b, s]].argmax(1) == M[:, [f, b, s]].argmax(1)).float().mean().item() * 100
        for f, b, s in zip(FP_INDICES, BP_INDICES, SP_INDICES)]
    fp, bp, sp = P[:, FP_INDICES], P[:, BP_INDICES], P[:, SP_INDICES]
    v = torch.cat([(fp[:, 1:] - fp[:, :-1]).clamp(min=0), (sp[:, 1:] - sp[:, :-1]).clamp(min=0),
                   (bp[:, :-1] - bp[:, 1:]).clamp(min=0)], 1).max(1).values
    out["mono_viol_any"] = (v > 0).float().mean().item() * 100
    out["mono_viol_gt0.01"] = (v > 0.01).float().mean().item() * 100
    return out, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam-task", type=float, required=True)
    ap.add_argument("--lam-const", type=float, default=1.0)
    ap.add_argument("--lam-mono", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=CONCORDE_EPOCHS)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--input", choices=["gem", "concorde"], default="gem",
                    help="gem: 4681 features (Concorde + 808 D3); concorde: first 3873 only")
    ap.add_argument("--hidden", type=int, nargs="+", default=HIDDEN, help="hidden layer sizes; use a separate --out")
    args = ap.parse_args()
    in_dim = GEM_INPUT_DIM if args.input == "gem" else CONCORDE_INPUT_DIM

    os.makedirs(args.out, exist_ok=True)
    tag = f"t{fmt(args.lam_task)}_c{fmt(args.lam_const)}_m{fmt(args.lam_mono)}_s{args.seed}" + ("_nod3" if args.input == "concorde" else "")
    if os.path.exists(os.path.join(args.out, f"{tag}.json")):
        sys.exit(f"{tag}.json exists in {args.out}, refusing to overwrite")
    dev = torch.device(f"cuda:{args.gpu}")

    data = load_all_benchmarks(DATA_GEM, DATA_CON, BENCHMARK_NAMES)
    tl, vl, _, _, bm_idx = create_exp1_dataloaders(data, seed=SEED)
    del data
    g = lambda ds, k: (ds.__dict__[k][:, :in_dim].contiguous() if k == "gem_features"
                       else ds.__dict__[k]).to(dev)
    Xtr, Mtr, Ytr, Wtr = (g(tl.dataset, k) for k in ["gem_features", "mscs_target", "cpi_golden", "w_min"])
    Xte, Mte, Yte, Wte = (g(vl.dataset, k) for k in ["gem_features", "mscs_target", "cpi_golden", "w_min"])

    set_seed(args.seed)
    model = GEMModel(in_dim, args.hidden, DROPOUT).to(dev)
    opt = optim.AdamW(model.parameters(), lr=CONCORDE_LR, weight_decay=CONCORDE_WEIGHT_DECAY)
    n = len(Xtr); spe = n // BATCH
    total = args.epochs * spe
    sched = optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(total * f) for f in [0.42, 0.58, 0.75, 0.92]], gamma=0.5)
    print(f"[{tag}] train={n} test={len(Xte)} steps/ep={spe} in_dim={in_dim}", flush=True)

    best, best_state, best_ep, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(spe):
            b = perm[i * BATCH:(i + 1) * BATCH]
            mp = model(Xtr[b])
            loss = args.lam_task * loss_task(mp, Mtr[b])
            if args.lam_const > 0:
                loss = loss + args.lam_const * loss_const(mp, Wtr[b], Ytr[b])
            if args.lam_mono > 0:
                loss = loss + args.lam_mono * loss_mono(mp)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        ev, _ = evaluate(model, Xte, Wte, Yte, Mte)
        hist.append({"epoch": ep, **ev})
        if ev["mape_mean4"] < best:
            best, best_ep = ev["mape_mean4"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 100 == 0 or ep == 1:
            print(f"[{tag}] ep {ep:4d} mean4={ev['mape_mean4']:.2f} best={best:.2f}@{best_ep} "
                  f"| {time.time()-t0:.0f}s", flush=True)

    last, _ = evaluate(model, Xte, Wte, Yte, Mte, full=True)
    model.load_state_dict(best_state)
    sel, C = evaluate(model, Xte, Wte, Yte, Mte, full=True)
    per_bm = {name: rel(C[s:e].mean(1), Yte[s:e]).mean().item() * 100
              for name, (s, e) in bm_idx.items() if s < e}
    res = {"lam_task": args.lam_task, "lam_const": args.lam_const, "lam_mono": args.lam_mono,
           "seed": args.seed, "input": args.input, "input_dim": in_dim, "split_seed": SEED, "hidden": args.hidden, "best_epoch": best_ep,
           "selected": sel, "last_epoch": last, "per_benchmark_mape": per_bm,
           "train_time_s": time.time() - t0, "history": hist}
    json.dump(res, open(os.path.join(args.out, f"{tag}.json"), "w"), indent=2)
    torch.save(best_state, os.path.join(args.out, f"{tag}.pt"))
    print(f"[{tag}] DONE mape={sel['mape_mean4']:.2f} MAE FP/BP/SP="
          f"{sel['mae']['FP']:.3f}/{sel['mae']['BP']:.3f}/{sel['mae']['SP']:.3f} "
          f"dom@Dis={sel['dominant_acc'][1]:.1f} viol>0.01={sel['mono_viol_gt0.01']:.1f}", flush=True)


if __name__ == "__main__":
    main()
