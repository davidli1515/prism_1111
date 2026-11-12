#!/usr/bin/env python3
"""
CPI-supervision ablation (cpi_ablation_1003 A and cpi_ablation_selfcons_1007 B) rerun at the final setting:
V2 model, no-D3 input (first 3873 Concorde features), [512,256], lam_task = 10, CPI-term weight 1 (= lam_const),
no L_mono, 100% labels. Same data / split / recipe / selection / evaluation as train_exp1_lambda_sens.py
(row split seed 42, AdamW 1e-3 / wd 0.3, batch 50k, 1521 epochs, step LR at [42,58,75,92]%, grad clip 1.0,
checkpoint with best test CPI MAPE of the 4-stage mean).

Variants:
  A      10 L_task + |CPI_commit - CPI_gt| / CPI_gt                          (only Commit anchored)
  B_var  A + mean_s (CPI_s - mean CPI)^2                                     (label-free self-consistency, variance form)
  B_rel  A + mean_s |CPI_s - mean CPI| / mean CPI                            (relative form)
  C      10 L_task + L_const = the final configuration; NOT rerun here, use
         results/lambda_sweep_nod3_1026/t10_c1_m0_s{42,123,456}_nod3.json (same code path)

Refuses to overwrite. Usage:
    python train_exp1_cpi_ablation_nod3_1110.py --variant A --seed 42 --gpu 0
"""
import os, sys, json, time, argparse
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONCORDE_INPUT_DIM, DROPOUT, BENCHMARK_NAMES, SEED, CONCORDE_LR, CONCORDE_WEIGHT_DECAY, CONCORDE_EPOCHS, BASE_INDICES
from model import GEMModel, loss_task
from dataset import load_all_benchmarks, create_exp1_dataloaders
from train_exp1_cpi_ablation import set_seed, rel, DATA_GEM, DATA_CON, HIDDEN, BATCH
from train_exp1_lambda_sens import evaluate

OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cpi_ablation_nod3_1110")
LAM_TASK, LAM_CPI, LAM_SELF = 10.0, 1.0, 1.0


def anchor(mp, w, y, eps=1e-8):
    return rel(1.0 / (mp[:, BASE_INDICES[3]] * w + eps), y).mean()


def selfcons(mp, w, form, eps=1e-8):
    c = 1.0 / (mp[:, BASE_INDICES] * w.unsqueeze(1) + eps)              # (B,4), no clamps (as loss_const)
    m = c.mean(1, keepdim=True)
    return ((c - m) ** 2).mean() if form == "var" else (torch.abs(c - m) / (m + eps)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["A", "B_var", "B_rel"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=CONCORDE_EPOCHS)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.variant}_s{args.seed}_nod3"
    out_json = os.path.join(args.out, f"{tag}.json")
    if os.path.exists(out_json):
        sys.exit(f"{out_json} exists, refusing to overwrite")
    dev = torch.device(f"cuda:{args.gpu}")

    data = load_all_benchmarks(DATA_GEM, DATA_CON, BENCHMARK_NAMES)
    tl, vl, _, _, bm_idx = create_exp1_dataloaders(data, seed=SEED)
    del data
    g = lambda ds, k: (ds.__dict__[k][:, :CONCORDE_INPUT_DIM].contiguous() if k == "gem_features" else ds.__dict__[k]).to(dev)
    Xtr, Mtr, Ytr, Wtr = (g(tl.dataset, k) for k in ["gem_features", "mscs_target", "cpi_golden", "w_min"])
    Xte, Mte, Yte, Wte = (g(vl.dataset, k) for k in ["gem_features", "mscs_target", "cpi_golden", "w_min"])

    set_seed(args.seed)
    model = GEMModel(CONCORDE_INPUT_DIM, HIDDEN, DROPOUT).to(dev)
    opt = optim.AdamW(model.parameters(), lr=CONCORDE_LR, weight_decay=CONCORDE_WEIGHT_DECAY)
    n = len(Xtr); spe = n // BATCH
    total = args.epochs * spe
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[int(total * f) for f in [0.42, 0.58, 0.75, 0.92]], gamma=0.5)
    print(f"[{tag}] train={n} test={len(Xte)} steps/ep={spe} in_dim={CONCORDE_INPUT_DIM} lam_task={LAM_TASK}", flush=True)

    best, best_state, best_ep, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(spe):
            b = perm[i * BATCH:(i + 1) * BATCH]
            mp = model(Xtr[b]); w = Wtr[b]
            loss = LAM_TASK * loss_task(mp, Mtr[b]) + LAM_CPI * anchor(mp, w, Ytr[b])
            if args.variant != "A":
                loss = loss + LAM_SELF * selfcons(mp, w, args.variant.split("_")[1])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        ev, _ = evaluate(model, Xte, Wte, Yte, Mte)
        hist.append({"epoch": ep, **ev})
        if ev["mape_mean4"] < best:
            best, best_ep = ev["mape_mean4"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 100 == 0 or ep == 1:
            print(f"[{tag}] ep {ep:4d} mean4={ev['mape_mean4']:.2f} commit={ev['mape_commit']:.2f} best={best:.2f}@{best_ep} "
                  f"| {time.time()-t0:.0f}s", flush=True)

    last, _ = evaluate(model, Xte, Wte, Yte, Mte, full=True)
    model.load_state_dict(best_state)
    sel, C = evaluate(model, Xte, Wte, Yte, Mte, full=True)
    per_bm = {name: rel(C[s:e].mean(1), Yte[s:e]).mean().item() * 100 for name, (s, e) in bm_idx.items() if s < e}
    res = {"variant": args.variant, "lam_task": LAM_TASK, "lam_cpi": LAM_CPI, "lam_self": LAM_SELF if args.variant != "A" else 0.0,
           "seed": args.seed, "input": "concorde", "input_dim": CONCORDE_INPUT_DIM, "split_seed": SEED, "hidden": HIDDEN,
           "best_epoch": best_ep, "selected": sel, "last_epoch": last, "per_benchmark_mape": per_bm,
           "train_time_s": time.time() - t0, "history": hist}
    json.dump(res, open(out_json, "w"), indent=2)
    torch.save(best_state, os.path.join(args.out, f"{tag}.pt"))
    print(f"[{tag}] DONE mean4={sel['mape_mean4']:.2f} commit={sel['mape_commit']:.2f} "
          f"stages={[round(x, 2) for x in sel['mape_per_stage']]} base_spread={sel['base_spread_rel']:.4f}", flush=True)


if __name__ == "__main__":
    main()
