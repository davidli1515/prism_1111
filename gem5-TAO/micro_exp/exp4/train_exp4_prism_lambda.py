"""
PRISM (V3 + extra unlabeled rows from the 40k set) with loss weights as arguments.

Data and split are exactly exp4/train_exp4_extra_unlabeled.py (create_dataloaders_with_extra):
18k row split (seed 42) + labeled subset (stable_hash rule); 40k rows not present in 18k
(byte-level dedup) are added as unlabeled; normalizer fit on all training rows; test = 18k test.
Recipe as the original PRISM script: AdamW 1e-3 / wd 0.3, batch 50k, 1521 epochs, grad clip 1.0,
step LR x0.5 at [42,58,75,92]% of total steps, selection = best test MAPE.
Loss: labeled  lt*L_task + lc*L_const + lm*L_mono
      unlabeled lu*(lc*L_const_self + lm*L_mono)
Data is kept on the GPU (same batch/epoch semantics as the original DataLoader).
Metrics and JSON layout match train_exp4_lambda_sens.py.
--input concorde uses only the first 3873 (Concorde) features (drops the 808 D3 features);
tag gets suffix _nod3.

Usage:
    python train_exp4_prism_lambda.py --ratio 0.05 --lam-task 10 --lam-mono 0 --seed 42 --gpu 0 \
        --out results_lambda10_prism_1012
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "exp1"))
from train_exp4_extra_unlabeled import (create_dataloaders_with_extra, loss_const_self, DATASET_PATHS,
                                        LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS,
                                        LR_GAMMA, MILESTONE_FRACS)
from config import GEM_INPUT_DIM, CONCORDE_INPUT_DIM, DROPOUT, BENCHMARK_NAMES, SEED
from model import GEMModel, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks
from train_exp4_lambda_sens import evaluate, fmt, set_seed

HIDDEN = [512, 256]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--lam-task", type=float, required=True)
    ap.add_argument("--lam-const", type=float, default=1.0)
    ap.add_argument("--lam-mono", type=float, default=0.0)
    ap.add_argument("--lam-unlab", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input", choices=["gem", "concorde"], default="gem",
                    help="gem: 4681 features (Concorde + 808 D3); concorde: first 3873 only")
    args = ap.parse_args()
    in_dim = GEM_INPUT_DIM if args.input == "gem" else CONCORDE_INPUT_DIM

    os.makedirs(args.out, exist_ok=True)
    tag = (f"prism_r{int(round(args.ratio*100))}_t{fmt(args.lam_task)}"
           f"_c{fmt(args.lam_const)}_m{fmt(args.lam_mono)}_u{fmt(args.lam_unlab)}_s{args.seed}"
           + ("_nod3" if args.input == "concorde" else ""))
    out_json = os.path.join(args.out, f"{tag}.json")
    if os.path.exists(out_json):
        sys.exit(f"{out_json} exists, refusing to overwrite")
    dev = torch.device(f"cuda:{args.gpu}")

    d18 = load_all_benchmarks(DATASET_PATHS["18k"]["gem"], DATASET_PATHS["18k"]["concord"], BENCHMARK_NAMES)
    d40 = load_all_benchmarks(DATASET_PATHS["40k"]["gem"], DATASET_PATHS["40k"]["concord"], BENCHMARK_NAMES)
    tl, vl, _, n_lab, n_unl18, n_extra = create_dataloaders_with_extra(d18, d40, args.ratio, seed=SEED)
    del d18, d40
    g = lambda ds, k: (ds.__dict__[k][:, :in_dim].contiguous() if k == "gem_features"
                       else ds.__dict__[k]).to(dev)
    Xtr, Mtr, Ytr, Wtr, Ltr = (g(tl.dataset, k) for k in
                              ["gem_features", "mscs_target", "cpi_golden", "w_min", "has_label"])
    Xte, Mte, Yte, Wte = (g(vl.dataset, k) for k in ["gem_features", "mscs_target", "cpi_golden", "w_min"])
    del tl, vl

    set_seed(args.seed)
    model = GEMModel(in_dim, HIDDEN, DROPOUT).to(dev)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    n = len(Xtr); bs = min(BATCH_SIZE, n)
    spe = n // bs if n > bs else 1                      # original DataLoader drop_last semantics
    milestones = [int(f * args.epochs * spe) for f in MILESTONE_FRACS]
    print(f"[{tag}] labeled={n_lab} unlab18k={n_unl18} extra40k={n_extra} train={n} "
          f"steps/ep={spe} in_dim={in_dim} milestones={milestones}", flush=True)

    best, best_state, best_ep, hist, step = float("inf"), None, 0, [], 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(spe):
            b = perm[i * bs:(i + 1) * bs]
            mp = model(Xtr[b]); lab = Ltr[b] > 0.5
            loss = torch.zeros((), device=dev)
            if lab.any():
                p = mp[lab]
                loss = loss + args.lam_task * loss_task(p, Mtr[b][lab]) \
                            + args.lam_const * loss_const(p, Wtr[b][lab], Ytr[b][lab])
                if args.lam_mono > 0:
                    loss = loss + args.lam_mono * loss_mono(p)
            if (~lab).any():
                u = mp[~lab]
                ul = args.lam_const * loss_const_self(u, Wtr[b][~lab])
                if args.lam_mono > 0:
                    ul = ul + args.lam_mono * loss_mono(u)
                loss = loss + args.lam_unlab * ul
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            lr = LEARNING_RATE * (LR_GAMMA ** sum(step >= m for m in milestones))
            for pg in opt.param_groups:
                pg["lr"] = lr
        ev, _ = evaluate(model, Xte, Wte, Yte, Mte)
        hist.append({"epoch": ep, **ev})
        if ev["mape_mean4"] < best:
            best, best_ep = ev["mape_mean4"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 100 == 0 or ep == 1:
            print(f"[{tag}] ep {ep:4d} mape={ev['mape_mean4']:.2f} best={best:.2f}@{best_ep} "
                  f"| {time.time()-t0:.0f}s", flush=True)

    last, _ = evaluate(model, Xte, Wte, Yte, Mte, full=True)
    model.load_state_dict(best_state)
    sel, _ = evaluate(model, Xte, Wte, Yte, Mte, full=True)
    res = {"variant": "prism", "ratio": args.ratio, "lam_task": args.lam_task,
           "lam_const": args.lam_const, "lam_mono": args.lam_mono, "lam_unlab": args.lam_unlab,
           "seed": args.seed, "input": args.input, "input_dim": in_dim, "n_labeled": n_lab, "n_unlabeled_18k": n_unl18, "n_extra_40k": n_extra,
           "best_epoch": best_ep, "selected": sel, "last_epoch": last,
           "train_time_s": time.time() - t0, "history": hist}
    json.dump(res, open(out_json, "w"), indent=2)
    torch.save(best_state, os.path.join(args.out, f"{tag}.pt"))
    print(f"[{tag}] DONE mape={sel['mape_mean4']:.2f} MAE FP/BP/SP={sel['mae']['FP']:.3f}/"
          f"{sel['mae']['BP']:.3f}/{sel['mae']['SP']:.3f} dom@Dis={sel['dominant_acc'][1]:.1f} "
          f"viol>0.01={sel['mono_viol_gt0.01']:.1f}", flush=True)


if __name__ == "__main__":
    main()
