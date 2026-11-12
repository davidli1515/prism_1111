"""
Standard semi-supervised baseline: Concorde + Mean Teacher (Tarvainen & Valpola, NeurIPS 2017).

--setting a : Concorde baseline recipe (train_exp4_all.py --ablation concorde): 18k row split
              (seed 42), Concorde features, LR x0.5 at FIXED_MILESTONES_18K. ratio < 1: the
              remaining 18k train rows are unlabeled (create_exp4_dataloaders, as V3);
              ratio 1.0: labeled rows only = the 2.74% Concorde run.
--setting b : PRISM's data (create_dataloaders_with_extra, as train_exp4_prism_lambda.py):
              labeled subset + 267k extra 40k rows (byte-deduped vs 18k) as unlabeled; input =
              first 3873 gem features (per-feature normalisation, so slicing is exact); LR x0.5 at
              [42,58,75,92]% of total steps, as PRISM.
Both: AdamW 1e-3 / wd 0.3, batch min(50k, train), 1521 epochs, grad clip 1.0, test = 18k test set,
selection = best test CPI MAPE of the teacher (student also reported).
Model: Concorde backbone ([512,256], MLPBlock = Linear-BN-ReLU, dropout 0) + scalar head.
Teacher = EMA of the student weights (BN buffers copied from the student).
Loss per batch:
  labeled rows : Concorde relative loss |s(x) - cpi| / cpi                 (s = student on clean x)
  all rows     : w(t) * lam_cons * mean((s(x + sigma*e) - t(x + sigma*e'))^2) / c^2
                 (input-perturbation consistency; c = mean labeled training CPI, so the term is
                  scale-free like Concorde's loss; w(t) = exp(-5 (1 - t/T_r)^2) sigmoid ramp-up
                  over the first --rampup fraction of steps, as in Mean Teacher)

Usage:
    python train_exp4_concorde_mt.py --setting a --lam-cons 1 --sigma 0.1 --seed 42 --gpu 0 --out DIR
"""
import os, sys, json, time, copy, argparse
import numpy as np
import torch
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

HIDDEN = [512, 256]


def fmt(x):
    return ("%g" % x).replace(".", "p")


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def rel_loss(p, y, eps=1e-8):
    return (torch.abs(p - y) / (torch.abs(y) + eps)).mean()


@torch.no_grad()
def ema_update(teacher, student, decay):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(decay).add_(ps, alpha=1 - decay)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.copy_(bs)


@torch.no_grad()
def evaluate(model, X, Y):
    model.eval()
    P = torch.cat([model(X[i:i + BATCH_SIZE]) for i in range(0, len(X), BATCH_SIZE)])
    return (torch.abs(P - Y) / (torch.abs(Y) + 1e-8)).mean().item() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["a", "b"], required=True)
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--lam-cons", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.1, help="input noise std (z-scored features)")
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--rampup", type=float, default=0.1, help="fraction of steps for consistency ramp-up")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = (f"mt{args.setting}_r{int(round(args.ratio*100))}_l{fmt(args.lam_cons)}_n{fmt(args.sigma)}"
           f"_e{fmt(args.ema)}_s{args.seed}")
    out_json = os.path.join(args.out, f"{tag}.json")
    if os.path.exists(out_json):
        sys.exit(f"{out_json} exists, refusing to overwrite")
    dev = torch.device(f"cuda:{args.gpu}")

    d18 = load_all_benchmarks(DATASET_PATHS["18k"]["gem"], DATASET_PATHS["18k"]["concord"], BENCHMARK_NAMES)
    if args.setting == "b":
        d40 = load_all_benchmarks(DATASET_PATHS["40k"]["gem"], DATASET_PATHS["40k"]["concord"], BENCHMARK_NAMES)
        tl, vl, _, n_lab, n_unl18, n_extra = create_dataloaders_with_extra(d18, d40, args.ratio, seed=SEED)
        del d40
        feat = lambda ds: ds.gem_features[:, :CONCORDE_INPUT_DIM].contiguous()
    else:
        tl, vl, _, _, n_lab, n_unl18 = create_exp4_dataloaders(
            d18, args.ratio, labeled_only=(args.ratio >= 1.0), seed=SEED)
        n_extra = 0
        feat = lambda ds: ds.concord_features
    del d18
    Xtr, Ytr, Ltr = feat(tl.dataset).to(dev), tl.dataset.cpi_golden.to(dev), tl.dataset.has_label.to(dev)
    Xte, Yte = feat(vl.dataset).to(dev), vl.dataset.cpi_golden.to(dev)
    del tl, vl
    assert Xtr.shape[1] == CONCORDE_INPUT_DIM
    c2 = Ytr[Ltr > 0.5].mean().item() ** 2

    set_seed(args.seed)
    student = ConcordeModel(CONCORDE_INPUT_DIM, HIDDEN, dropout=0.0).to(dev)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    n = len(Xtr); bs = min(BATCH_SIZE, n)
    spe = n // bs if n > bs else 1                      # exp4 DataLoader drop_last semantics
    total = args.epochs * spe
    milestones = ([int(f * total) for f in MILESTONE_FRACS] if args.setting == "b"
                  else FIXED_MILESTONES_18K)
    t_ramp = max(1, int(args.rampup * total))
    print(f"[{tag}] labeled={n_lab} unlab18k={n_unl18} extra40k={n_extra} train={n} "
          f"steps/ep={spe} total={total} milestones={milestones} ramp={t_ramp} c={c2**0.5:.3f}", flush=True)

    best, best_state, best_ep, hist, step = float("inf"), None, 0, [], 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        student.train(); teacher.train()
        perm = torch.randperm(n, device=dev)
        l_lab, l_con = 0.0, 0.0
        for i in range(spe):
            b = perm[i * bs:(i + 1) * bs]
            x = Xtr[b]; lab = Ltr[b] > 0.5
            w = float(np.exp(-5.0 * (1.0 - min(step, t_ramp) / t_ramp) ** 2))
            s_clean = student(x)
            s_noisy = student(x + args.sigma * torch.randn_like(x))
            with torch.no_grad():
                t_noisy = teacher(x + args.sigma * torch.randn_like(x))
            lc = ((s_noisy - t_noisy) ** 2).mean() / c2
            loss = w * args.lam_cons * lc; l_con += lc.item()
            if lab.any():
                ll = rel_loss(s_clean[lab], Ytr[b][lab]); loss = loss + ll; l_lab += ll.item()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            step += 1; step_lr(opt, step, milestones)
            ema_update(teacher, student, args.ema)
        mt, ms = evaluate(teacher, Xte, Yte), evaluate(student, Xte, Yte)
        hist.append({"epoch": ep, "loss_lab": l_lab / spe, "loss_cons": l_con / spe,
                     "mape_teacher": mt, "mape_student": ms})
        if mt < best:
            best, best_ep = mt, ep
            best_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
        if ep % 100 == 0 or ep == 1:
            print(f"[{tag}] ep {ep:4d} teacher={mt:.2f} student={ms:.2f} best={best:.2f}@{best_ep} "
                  f"L_cons={l_con/spe:.2e} | {time.time()-t0:.0f}s", flush=True)

    last_t, last_s = evaluate(teacher, Xte, Yte), evaluate(student, Xte, Yte)
    res = {"method": "concorde_mean_teacher", "setting": args.setting, "ratio": args.ratio,
           "lam_cons": args.lam_cons, "sigma": args.sigma, "ema": args.ema, "rampup": args.rampup,
           "seed": args.seed, "input_dim": CONCORDE_INPUT_DIM, "n_labeled": n_lab,
           "n_unlabeled_18k": n_unl18, "n_extra_40k": n_extra, "milestones": milestones,
           "best_epoch": best_ep, "selected": {"mape": best},
           "last_epoch": {"mape_teacher": last_t, "mape_student": last_s},
           "train_time_s": time.time() - t0, "history": hist}
    json.dump(res, open(out_json, "w"), indent=2)
    print(f"[{tag}] DONE teacher mape={best:.2f}@{best_ep} last teacher/student={last_t:.2f}/{last_s:.2f}",
          flush=True)


if __name__ == "__main__":
    main()
