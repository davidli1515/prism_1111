"""
Low-label loss-weight sensitivity (tests: does L_task alone learn structure with few labels;
does semi-supervised L_const_self + L_mono recover CPI AND decomposition/ranking; does a
large lam_task shrink the semi-supervised gain; is L_mono's value largest at low labels).

Same as exp4 (train_exp4_all.py) except loss weights are arguments and data is GPU-resident:
  18k data, exp4 row split (seed 42) + labeled subset (stable_hash rule), [512,256],
  AdamW 1e-3 / wd 0.3, batch min(50k, train), 1521 epochs, grad clip 1.0,
  FIXED_MILESTONES_18K (as exp4, all variants), selection = best test MAPE.
  V1: labeled rows only:   lt*L_task (as train_exp4_vanilla.py: milestones = MILESTONE_FRACS
      x total steps; run with --lam-task 1 --lam-const 0 --lam-mono 0)
  V2: labeled rows only:   lt*L_task + lc*L_const + lm*L_mono
  V3: + unlabeled rows:    lu*(lc*L_const_self + lm*L_mono)
--input concorde uses only the first 3873 (Concorde) features, i.e. drops the 808 D3 features
(per-feature normalization, so slicing after normalization is exact); tag gets suffix _nod3.

Reports (selected and last epoch): CPI MAPE, MSCS MAE/MSE per component, dominant
non-Base bottleneck accuracy and full FP/BP/SP ranking accuracy per stage, cross-stage
monotonicity violations, cross-stage Base spread.

Usage:
    python train_exp4_lambda_sens.py --variant v3 --ratio 0.05 --lam-task 1 --lam-mono 0.01 --gpu 0
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "exp1"))
from train_exp4_all import (create_exp4_dataloaders, loss_const_self, DATASET_PATHS,
                            LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS,
                            FIXED_MILESTONES_18K, LR_GAMMA, step_lr)
from train_exp4_vanilla import MILESTONE_FRACS
from config import GEM_INPUT_DIM, CONCORDE_INPUT_DIM, DROPOUT, BENCHMARK_NAMES, SEED, FP_INDICES, BP_INDICES, SP_INDICES
from model import GEMModel, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks
from train_exp1_lambda_sens import evaluate as evaluate_basic

OUT_DEFAULT = os.path.join(HERE, "results_lambda_lowlabel_1004")
HIDDEN = [512, 256]


def fmt(x):
    return ("%g" % x).replace(".", "p")


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate(model, X, W, Y, M, full=False):
    out, C = evaluate_basic(model, X, W, Y, M, full=full)
    if full:
        P = torch.cat([model(X[i:i + BATCH_SIZE]) for i in range(0, len(X), BATCH_SIZE)])
        out["rank_acc"] = [
            (P[:, [f, b, s]].argsort(1) == M[:, [f, b, s]].argsort(1)).all(1).float().mean().item() * 100
            for f, b, s in zip(FP_INDICES, BP_INDICES, SP_INDICES)]
    return out, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["v1", "v2", "v3"])
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--lam-task", type=float, required=True)
    ap.add_argument("--lam-const", type=float, default=1.0)
    ap.add_argument("--lam-mono", type=float, default=0.01)
    ap.add_argument("--lam-unlab", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--input", choices=["gem", "concorde"], default="gem",
                    help="gem: 4681 features (Concorde + 808 D3); concorde: first 3873 only")
    args = ap.parse_args()
    if args.variant == "v1":
        assert args.lam_const == 0 and args.lam_mono == 0, "V1 is L_task only"
    in_dim = GEM_INPUT_DIM if args.input == "gem" else CONCORDE_INPUT_DIM

    os.makedirs(args.out, exist_ok=True)
    tag = (f"{args.variant}_r{int(round(args.ratio*100))}_t{fmt(args.lam_task)}"
           f"_c{fmt(args.lam_const)}_m{fmt(args.lam_mono)}_u{fmt(args.lam_unlab)}_s{args.seed}"
           + ("_nod3" if args.input == "concorde" else ""))
    out_json = os.path.join(args.out, f"{tag}.json")
    if os.path.exists(out_json):
        sys.exit(f"{out_json} exists, refusing to overwrite")
    dev = torch.device(f"cuda:{args.gpu}")

    data = load_all_benchmarks(DATASET_PATHS["18k"]["gem"], DATASET_PATHS["18k"]["concord"], BENCHMARK_NAMES)
    tl, vl, _, _, n_lab, n_unl = create_exp4_dataloaders(
        data, args.ratio, labeled_only=(args.variant in ("v1", "v2")), seed=SEED)
    del data
    g = lambda ds, k: (ds.__dict__[k][:, :in_dim].contiguous() if k == "gem_features"
                       else ds.__dict__[k]).to(dev)
    Xtr, Mtr, Ytr, Wtr, Ltr = (g(tl.dataset, k) for k in
                              ["gem_features", "mscs_target", "cpi_golden", "w_min", "has_label"])
    Xte, Mte, Yte, Wte = (g(vl.dataset, k) for k in ["gem_features", "mscs_target", "cpi_golden", "w_min"])

    set_seed(args.seed)
    model = GEMModel(in_dim, HIDDEN, DROPOUT).to(dev)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    n = len(Xtr); bs = min(BATCH_SIZE, n)
    spe = n // bs if n > bs else 1                      # exp4 DataLoader drop_last semantics
    milestones = ([int(f * args.epochs * spe) for f in MILESTONE_FRACS] if args.variant == "v1"
                  else FIXED_MILESTONES_18K)
    print(f"[{tag}] labeled={n_lab} unlabeled={n_unl} train={n} batch={bs} steps/ep={spe} in_dim={in_dim} milestones={milestones}", flush=True)

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
                loss = loss + args.lam_task * loss_task(p, Mtr[b][lab])
                if args.lam_const > 0:
                    loss = loss + args.lam_const * loss_const(p, Wtr[b][lab], Ytr[b][lab])
                if args.lam_mono > 0:
                    loss = loss + args.lam_mono * loss_mono(p)
            if args.variant == "v3" and (~lab).any():
                u = mp[~lab]
                ul = args.lam_const * loss_const_self(u, Wtr[b][~lab])
                if args.lam_mono > 0:
                    ul = ul + args.lam_mono * loss_mono(u)
                loss = loss + args.lam_unlab * ul
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1; step_lr(opt, step, milestones)
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
    res = {"variant": args.variant, "ratio": args.ratio, "lam_task": args.lam_task,
           "lam_const": args.lam_const, "lam_mono": args.lam_mono, "lam_unlab": args.lam_unlab,
           "seed": args.seed, "input": args.input, "input_dim": in_dim, "n_labeled": n_lab, "n_unlabeled": n_unl, "best_epoch": best_ep,
           "selected": sel, "last_epoch": last, "train_time_s": time.time() - t0, "history": hist}
    json.dump(res, open(out_json, "w"), indent=2)
    torch.save(best_state, os.path.join(args.out, f"{tag}.pt"))
    print(f"[{tag}] DONE mape={sel['mape_mean4']:.2f} MAE FP/BP/SP={sel['mae']['FP']:.3f}/"
          f"{sel['mae']['BP']:.3f}/{sel['mae']['SP']:.3f} dom@Dis={sel['dominant_acc'][1]:.1f} "
          f"rank@Dis={sel['rank_acc'][1]:.1f} viol>0.01={sel['mono_viol_gt0.01']:.1f}", flush=True)


if __name__ == "__main__":
    main()
