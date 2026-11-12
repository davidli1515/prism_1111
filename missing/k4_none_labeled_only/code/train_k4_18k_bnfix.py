"""Concorde K heads + 18k unlabeled rows (agreement loss), same steps AND same labeled batches as C1,
with unlabeled rows kept out of BatchNorm (as exp2/train_ood_bnfix.py).

Per step: labeled batch exactly as C1 (bs = min(50k, n_lab), steps/epoch = n_lab // bs or 1), forwarded in
train mode (BN batch stats + running-stat update). Unlabeled rows: each epoch split into steps/epoch
chunks, forwarded in eval mode (BN uses running stats, no update; dropout is 0), loss = lam * agree_loss.
--lam 0 or --heads 1 gives C1-equivalent training (unlabeled forward contributes nothing).
LR x0.5 at FIXED_MILESTONES_18K (as C1); selection = best test CPI MAPE of the head mean.
Usage: python train_k4_18k_bnfix.py --heads 4 --lam 1 --ratio 0.05 --seed 42 --gpu 0 --out DIR
"""
import os, sys, json, time, argparse
import torch, torch.nn as nn, torch.optim as optim
sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp4"); sys.path.insert(1, "/scratch/gem5-TAO/micro_exp/exp1")
import train_exp4_concorde_multihead as M
from train_exp4_all import create_exp4_dataloaders, DATASET_PATHS, step_lr
from config import BENCHMARK_NAMES, SEED, CONCORDE_INPUT_DIM
from dataset import load_all_benchmarks

ap = argparse.ArgumentParser()
ap.add_argument("--heads", type=int, default=4); ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--ratio", type=float, required=True); ap.add_argument("--seed", type=int, default=SEED)
ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--epochs", type=int, default=M.NUM_EPOCHS)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = f"k{a.heads}_18kbnfix_r{int(round(a.ratio * 100))}_a{M.fmt(a.lam)}_s{a.seed}"
out_json = os.path.join(a.out, tag + ".json")
assert not os.path.exists(out_json), out_json
dev = torch.device(f"cuda:{a.gpu}")

d18 = load_all_benchmarks(DATASET_PATHS["18k"]["gem"], DATASET_PATHS["18k"]["concord"], BENCHMARK_NAMES)
tl, vl, _, _, n_lab, n_unl = create_exp4_dataloaders(d18, a.ratio, labeled_only=False, seed=SEED)
del d18
X, Y, L = tl.dataset.concord_features.to(dev), tl.dataset.cpi_golden.to(dev), tl.dataset.has_label.to(dev) > 0.5
Xl, Yl, Xu = X[L], Y[L], X[~L]
Xte, Yte = vl.dataset.concord_features.to(dev), vl.dataset.cpi_golden.to(dev)
del tl, vl, X, Y, L

M.set_seed(a.seed)
model = M.MultiHeadConcorde(CONCORDE_INPUT_DIM, M.HIDDEN, a.heads).to(dev)
opt = optim.AdamW(model.parameters(), lr=M.LEARNING_RATE, weight_decay=M.WEIGHT_DECAY)
nl, nu = len(Xl), len(Xu)
bs = min(M.BATCH_SIZE, nl); spe = nl // bs if nl > bs else 1          # = C1
ubs = -(-nu // spe)
ms = M.FIXED_MILESTONES_18K
bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
print(f"[{tag}] labeled={nl} unlabeled={nu} bs_lab={bs} bs_unlab={ubs} steps/ep={spe} milestones={ms}", flush=True)

best, best_state, best_ep, hist, step, t0 = float("inf"), None, 0, [], 0, time.time()
for ep in range(1, a.epochs + 1):
    model.train()
    pl, pu = torch.randperm(nl, device=dev), torch.randperm(nu, device=dev)
    l_lab = l_agr = 0.0
    for i in range(spe):
        b = pl[i * bs:(i + 1) * bs]
        loss = M.rel_loss(model(Xl[b]), Yl[b]); l_lab += loss.item()
        if a.lam > 0 and a.heads > 1:
            for m in bns: m.eval()                                     # BN: running stats, no update
            la = M.agree_loss(model(Xu[pu[i * ubs:(i + 1) * ubs]]))
            for m in bns: m.train()
            loss = loss + a.lam * la; l_agr += la.item()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1; step_lr(opt, step, ms)
    ev = M.evaluate(model, Xte, Yte)
    hist.append({"epoch": ep, "loss_lab": l_lab / spe, "loss_agree": l_agr / spe, **ev})
    if ev["mape"] < best:
        best, best_ep = ev["mape"], ep
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if ep % 100 == 0 or ep == 1:
        print(f"[{tag}] ep {ep:4d} mape={ev['mape']:.2f} best={best:.2f}@{best_ep} "
              f"spread={ev['head_spread_rel']:.3f}% L_agree={l_agr / spe:.2e}", flush=True)
last = M.evaluate(model, Xte, Yte); model.load_state_dict(best_state); sel = M.evaluate(model, Xte, Yte)
json.dump({"heads": a.heads, "lam_agree": a.lam, "ratio": a.ratio, "seed": a.seed, "n_labeled": nl, "n_unlabeled": nu,
           "bs_lab": bs, "bs_unlab": ubs, "steps": step, "milestones": ms, "best_epoch": best_ep, "selected": sel,
           "last_epoch": last, "history": hist}, open(out_json, "w"), indent=2)
print(f"[{tag}] DONE mape={sel['mape']:.2f} per-head={[round(x, 2) for x in sel['mape_per_head']]} "
      f"spread={sel['head_spread_rel']:.3f}% steps={step}", flush=True)
