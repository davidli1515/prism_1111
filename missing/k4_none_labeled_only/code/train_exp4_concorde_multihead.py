import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "exp1"))
from train_exp4_all import (create_exp4_dataloaders, DATASET_PATHS, LEARNING_RATE, WEIGHT_DECAY,
                            BATCH_SIZE, NUM_EPOCHS, FIXED_MILESTONES_18K, step_lr)
from train_exp4_extra_unlabeled import create_dataloaders_with_extra, MILESTONE_FRACS
from config import CONCORDE_INPUT_DIM, BENCHMARK_NAMES, SEED
from model import ConcordeModel
from dataset import load_all_benchmarks

OUT_DEFAULT = os.path.join(HERE, "results")
HIDDEN = [512, 256]


def fmt(x):
    return ("%g" % x).replace(".", "p")


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


class MultiHeadConcorde(ConcordeModel):
    """ConcordeModel with K scalar CPI heads; K=1 keeps the original head."""
    def __init__(self, input_dim, hidden_dims, heads):
        super().__init__(input_dim, hidden_dims, dropout=0.0)
        if heads > 1:
            self.head = nn.Linear(hidden_dims[-1], heads)

    def forward(self, x):
        return self.head(self.backbone(x))               # (batch, K)


def rel_loss(P, y, eps=1e-8):
    return (torch.abs(P - y[:, None]) / (torch.abs(y[:, None]) + eps)).mean()


def agree_loss(P):
    return ((P - P.mean(1, keepdim=True)) ** 2).mean()


@torch.no_grad()
def evaluate(model, X, Y):
    model.eval()
    P = torch.cat([model(X[i:i + BATCH_SIZE]) for i in range(0, len(X), BATCH_SIZE)])
    err = lambda p: (torch.abs(p - Y) / (torch.abs(Y) + 1e-8)).mean().item() * 100
    m = P.mean(1)
    return {"mape": err(m),
            "mape_per_head": [err(P[:, k]) for k in range(P.shape[1])],
            "head_spread_rel": ((P.max(1).values - P.min(1).values) / m.abs().clamp_min(1e-8)).mean().item() * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--unlab", choices=["none", "18k", "40k"], required=True)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--lam-agree", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = (f"k{args.heads}_{args.unlab}_r{int(round(args.ratio*100))}"
           f"_a{fmt(args.lam_agree if args.unlab != 'none' else 0)}_s{args.seed}")
    out_json = os.path.join(args.out, f"{tag}.json")
    if os.path.exists(out_json):
        sys.exit(f"{out_json} exists, refusing to overwrite")
    dev = torch.device(f"cuda:{args.gpu}")

    d18 = load_all_benchmarks(DATASET_PATHS["18k"]["gem"], DATASET_PATHS["18k"]["concord"], BENCHMARK_NAMES)
    if args.unlab == "40k":
        d40 = load_all_benchmarks(DATASET_PATHS["40k"]["gem"], DATASET_PATHS["40k"]["concord"], BENCHMARK_NAMES)
        tl, vl, _, n_lab, n_unl18, n_extra = create_dataloaders_with_extra(d18, d40, args.ratio, seed=SEED)
        del d40
        feat = lambda ds: ds.gem_features[:, :CONCORDE_INPUT_DIM].contiguous()   # per-feature norm: exact
    else:
        tl, vl, _, _, n_lab, n_unl18 = create_exp4_dataloaders(
            d18, args.ratio, labeled_only=(args.unlab == "none"), seed=SEED)
        n_extra = 0
        feat = lambda ds: ds.concord_features                                       # Concorde baseline path
    del d18
    Xtr, Ytr, Ltr = feat(tl.dataset).to(dev), tl.dataset.cpi_golden.to(dev), tl.dataset.has_label.to(dev)
    Xte, Yte = feat(vl.dataset).to(dev), vl.dataset.cpi_golden.to(dev)
    del tl, vl
    assert Xtr.shape[1] == CONCORDE_INPUT_DIM

    set_seed(args.seed)
    model = MultiHeadConcorde(CONCORDE_INPUT_DIM, HIDDEN, args.heads).to(dev)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    n = len(Xtr); bs = min(BATCH_SIZE, n)
    spe = n // bs if n > bs else 1                      # exp4 DataLoader drop_last semantics
    milestones = ([int(f * args.epochs * spe) for f in MILESTONE_FRACS] if args.unlab == "40k"
                  else FIXED_MILESTONES_18K)
    print(f"[{tag}] labeled={n_lab} unlab18k={n_unl18 if args.unlab != 'none' else 0} extra40k={n_extra} "
          f"train={n} steps/ep={spe} milestones={milestones}", flush=True)

    best, best_state, best_ep, hist, step = float("inf"), None, 0, [], 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        l_lab, l_agr = 0.0, 0.0
        for i in range(spe):
            b = perm[i * bs:(i + 1) * bs]
            P = model(Xtr[b]); lab = Ltr[b] > 0.5
            loss = torch.zeros((), device=dev)
            if lab.any():
                ll = rel_loss(P[lab], Ytr[b][lab]); loss = loss + ll; l_lab += ll.item()
            if args.unlab != "none" and args.heads > 1 and (~lab).any():
                la = agree_loss(P[~lab]); loss = loss + args.lam_agree * la; l_agr += la.item()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1; step_lr(opt, step, milestones)
        ev = evaluate(model, Xte, Yte)
        hist.append({"epoch": ep, "loss_lab": l_lab / spe, "loss_agree": l_agr / spe, **ev})
        if ev["mape"] < best:
            best, best_ep = ev["mape"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 100 == 0 or ep == 1:
            print(f"[{tag}] ep {ep:4d} mape={ev['mape']:.2f} best={best:.2f}@{best_ep} "
                  f"spread={ev['head_spread_rel']:.3f}% L_agree={l_agr/spe:.2e}",
                  flush=True)

    last = evaluate(model, Xte, Yte)
    model.load_state_dict(best_state)
    sel = evaluate(model, Xte, Yte)
    res = {"heads": args.heads, "unlab": args.unlab, "ratio": args.ratio, "lam_agree": args.lam_agree,
           "seed": args.seed, "input_dim": CONCORDE_INPUT_DIM, "n_labeled": n_lab,
           "n_unlabeled_18k": n_unl18 if args.unlab != "none" else 0, "n_extra_40k": n_extra,
           "milestones": milestones, "best_epoch": best_ep, "selected": sel, "last_epoch": last,
           "history": hist}
    json.dump(res, open(out_json, "w"), indent=2)
    torch.save(best_state, os.path.join(args.out, f"{tag}.pt"))
    print(f"[{tag}] DONE mape={sel['mape']:.2f} per-head={[round(x, 2) for x in sel['mape_per_head']]} "
          f"spread={sel['head_spread_rel']:.3f}%", flush=True)


if __name__ == "__main__":
    main()
