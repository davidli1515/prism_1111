"""
Variant B of the CPI-supervision ablation: can cross-stage consistency be separated
from per-stage CPI supervision?

  B   0.1 L_task + |CPI_commit - CPI_gt|/CPI_gt + L_selfcons     (labeled samples only)

Only ONE stage is anchored to gem5 CPI; the other stages are tied to it by a
label-free self-consistency term on the same labeled samples. Two forms:
  var  : mean_s (CPI_s - mean CPI)^2               (= L_const^self used for unlabeled data)
  rel  : mean_s |CPI_s - mean CPI| / mean CPI      (scale-matched to L_const)
Compare with cpi_ablation: A = v1_cpi_commit (2.85% Commit, other stages 7-9%),
C = v2_const (all stages 2.87-2.88%).

Same data / split / recipe / evaluation as train_exp1_cpi_ablation.py.
Usage: python train_exp1_cpi_selfcons.py --form var --seed 42 --gpu 0
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (GEM_INPUT_DIM, DROPOUT, BENCHMARK_NAMES, SEED,
                    CONCORDE_LR, CONCORDE_WEIGHT_DECAY, CONCORDE_EPOCHS, BASE_INDICES)
from model import GEMModel, loss_task
from dataset import load_all_benchmarks, create_exp1_dataloaders
from train_exp1_cpi_ablation import set_seed, rel, evaluate, DATA_GEM, DATA_CON, HIDDEN, BATCH

OUT_DEFAULT = "/path/to/gem5/micro_exp/exp1/results/cpi_ablation_selfcons"


def selfcons(mp, w_min, form, eps=1e-8):
    c = 1.0 / (mp[:, BASE_INDICES] * w_min.unsqueeze(1) + eps)     # (B,4), no clamps (as loss_const)
    m = c.mean(1, keepdim=True)
    if form == "var":
        return ((c - m) ** 2).mean()
    return (torch.abs(c - m) / (m + eps)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", required=True, choices=["var", "rel"])
    ap.add_argument("--lam-self", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=CONCORDE_EPOCHS)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = f"B_{args.form}_l{('%g' % args.lam_self).replace('.', 'p')}_s{args.seed}"
    out_json = os.path.join(args.out, f"{tag}.json")
    if os.path.exists(out_json):
        sys.exit(f"{out_json} exists; refusing to overwrite")
    dev = torch.device(f"cuda:{args.gpu}")

    data = load_all_benchmarks(DATA_GEM, DATA_CON, BENCHMARK_NAMES)
    tl, vl, _, _, bm_idx = create_exp1_dataloaders(data, seed=SEED)
    del data
    g = lambda ds, k: ds.__dict__[k].to(dev)
    Xtr, Mtr, Ytr, Wtr = (g(tl.dataset, k) for k in ["gem_features", "mscs_target", "cpi_golden", "w_min"])
    Xte, Mte, Yte, Wte = (g(vl.dataset, k) for k in ["gem_features", "mscs_target", "cpi_golden", "w_min"])

    set_seed(args.seed)
    model = GEMModel(GEM_INPUT_DIM, HIDDEN, DROPOUT).to(dev)
    opt = optim.AdamW(model.parameters(), lr=CONCORDE_LR, weight_decay=CONCORDE_WEIGHT_DECAY)
    n = len(Xtr); spe = n // BATCH
    total = args.epochs * spe
    milestones = [int(total * f) for f in [0.42, 0.58, 0.75, 0.92]]
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.5)
    print(f"[{tag}] train={n} test={len(Xte)} steps/ep={spe}")

    best, best_state, best_ep, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(spe):
            b = perm[i * BATCH:(i + 1) * BATCH]
            mp = model(Xtr[b]); w = Wtr[b]
            anchor = rel(1.0 / (mp[:, BASE_INDICES[3]] * w + 1e-8), Ytr[b]).mean()
            loss = 0.1 * loss_task(mp, Mtr[b]) + anchor + args.lam_self * selfcons(mp, w, args.form)
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
    sel, _ = evaluate(model, Xte, Wte, Yte, Mte)
    res = {"variant": "B", "form": args.form, "lam_self": args.lam_self, "seed": args.seed,
           "split_seed": SEED, "best_epoch": best_ep, "selected": sel, "last_epoch": last,
           "train_time_s": time.time() - t0, "history": hist}
    json.dump(res, open(out_json, "w"), indent=2)
    torch.save(best_state, os.path.join(args.out, f"{tag}.pt"))
    print(f"[{tag}] DONE mean4={sel['mape_mean4']:.2f} commit={sel['mape_commit']:.2f} "
          f"stages={[round(x, 2) for x in sel['mape_per_stage']]} base_spread={sel['base_spread_rel']:.4f}")


if __name__ == "__main__":
    main()
