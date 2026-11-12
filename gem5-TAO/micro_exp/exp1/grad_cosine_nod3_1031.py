"""
Re-measure the gradient directions of Fig. 6 under the final experimental setting.

Training is identical to train_exp1_lambda_sens.py --input concorde (V2, 100% labels, 3873 Concorde
features, 18k row split seed 42, [512,256], dropout 0.1, AdamW 1e-3 / wd 0.3, batch 50k, 1521 epochs,
step LR at [42,58,75,92]%, grad clip 1.0, selection = best test CPI MAPE). The measurement runs under
a forked RNG and restores the BN buffers, so it does not change the training trajectory: a run must
reproduce the matching results/lambda_sweep_nod3_1026 run (collect_grad_cosine_nod3_1031.py checks it).

L_task, L_const and L_mono gradients are always measured, whichever terms are trained. Measured after
each of the first 100 epochs, then every 10 epochs, plus at initialization (epoch 0), on all full
training batches of one extra shuffle, in train mode (dropout, BN), on the same forward pass:
  cos_task_mono, cos_task_const, cos_const_mono   mean per-batch cosine of the raw gradients
  cos_task_hinge    cosine with the gradient of the violated part of L_mono: hinge = relu instead of
                    softplus. Softplus keeps widening margins of satisfied constraints, the hinge only
                    acts on actual violations. None if no training row violates monotonicity.
  norm_task, norm_const, norm_mono, norm_total    norm_total = ||sum of lam_i * grad_i|| (trained terms)
  l_task, l_const, l_mono                         training-batch loss values
  test                                            evaluate(full=True) on the test set
No .pt is saved. Refuses to overwrite an existing result.

Usage:
    python grad_cosine_nod3_1031.py --lam-task 10 --lam-const 1 --lam-mono 0 --seed 42 --gpu 0
"""
import os, sys, json, time, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (CONCORDE_INPUT_DIM, DROPOUT, BENCHMARK_NAMES, SEED, CONCORDE_LR, CONCORDE_WEIGHT_DECAY,
                    CONCORDE_EPOCHS, FP_INDICES, BP_INDICES, SP_INDICES)
from model import GEMModel, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, create_exp1_dataloaders
from train_exp1_cpi_ablation import set_seed, DATA_GEM, DATA_CON, HIDDEN, BATCH
from train_exp1_lambda_sens import evaluate, fmt

OUT_DEFAULT = "/scratch/gem5-TAO/micro_exp/exp1/results/grad_cosine_nod3_1031"
DENSE_EPOCHS = 100
SPARSE_EVERY = 10


def is_measured(ep, last):
    return ep <= DENSE_EPOCHS or ep % SPARSE_EVERY == 0 or ep == last


def loss_hinge(mp):
    """L_mono with relu instead of softplus: zero gradient where monotonicity holds."""
    fp, sp, bp = mp[:, FP_INDICES], mp[:, SP_INDICES], mp[:, BP_INDICES]
    return (F.relu(fp[:, 1:] - fp[:, :-1]).sum(1).mean() + F.relu(sp[:, 1:] - sp[:, :-1]).sum(1).mean()
            + F.relu(bp[:, :-1] - bp[:, 1:]).sum(1).mean())


def flat_grad(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for g, p in zip(gs, params)])


def cosine(a, b):
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()


def measure(model, X, M, Y, W, lam_task, lam_const, lam_mono, batch=BATCH):
    """Gradient directions on one shuffle of X; leaves RNG, BN buffers and .grad untouched."""
    params = list(model.parameters())
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    acc = defaultdict(list)
    with torch.random.fork_rng(devices=[X.device.index] if X.device.type == "cuda" else []):
        model.train()
        perm = torch.randperm(len(X), device=X.device)
        for i in range(len(X) // batch):
            b = perm[i * batch:(i + 1) * batch]
            mp = model(X[b])
            lt, lc, lm, lh = loss_task(mp, M[b]), loss_const(mp, W[b], Y[b]), loss_mono(mp), loss_hinge(mp)
            gt, gc, gm = flat_grad(lt, params), flat_grad(lc, params), flat_grad(lm, params)
            acc["cos_task_mono"].append(cosine(gt, gm))
            acc["cos_task_const"].append(cosine(gt, gc))
            acc["cos_const_mono"].append(cosine(gc, gm))
            if lh.item() > 0:
                acc["cos_task_hinge"].append(cosine(gt, flat_grad(lh, params)))
            acc["norm_task"].append(gt.norm().item())
            acc["norm_const"].append(gc.norm().item())
            acc["norm_mono"].append(gm.norm().item())
            acc["norm_total"].append((lam_task * gt + lam_const * gc + lam_mono * gm).norm().item())
            acc["l_task"].append(lt.item())
            acc["l_const"].append(lc.item())
            acc["l_mono"].append(lm.item())
    with torch.no_grad():
        for k, v in model.named_buffers():
            v.copy_(buffers[k])
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out.setdefault("cos_task_hinge", None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam-task", type=float, required=True)
    ap.add_argument("--lam-const", type=float, required=True)
    ap.add_argument("--lam-mono", type=float, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=CONCORDE_EPOCHS)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    in_dim = CONCORDE_INPUT_DIM

    os.makedirs(args.out, exist_ok=True)
    tag = f"t{fmt(args.lam_task)}_c{fmt(args.lam_const)}_m{fmt(args.lam_mono)}_s{args.seed}_nod3"
    if os.path.exists(os.path.join(args.out, f"{tag}.json")):
        sys.exit(f"{tag}.json exists in {args.out}, refusing to overwrite")
    dev = torch.device(f"cuda:{args.gpu}")

    data = load_all_benchmarks(DATA_GEM, DATA_CON, BENCHMARK_NAMES)
    tl, vl, _, _, _ = create_exp1_dataloaders(data, seed=SEED)
    del data
    g = lambda ds, k: (ds.__dict__[k][:, :in_dim].contiguous() if k == "gem_features"
                       else ds.__dict__[k]).to(dev)
    Xtr, Mtr, Ytr, Wtr = (g(tl.dataset, k) for k in ["gem_features", "mscs_target", "cpi_golden", "w_min"])
    Xte, Mte, Yte, Wte = (g(vl.dataset, k) for k in ["gem_features", "mscs_target", "cpi_golden", "w_min"])

    set_seed(args.seed)
    model = GEMModel(in_dim, HIDDEN, DROPOUT).to(dev)
    opt = optim.AdamW(model.parameters(), lr=CONCORDE_LR, weight_decay=CONCORDE_WEIGHT_DECAY)
    n = len(Xtr); spe = n // BATCH
    total = args.epochs * spe
    sched = optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(total * f) for f in [0.42, 0.58, 0.75, 0.92]], gamma=0.5)
    print(f"[{tag}] train={n} test={len(Xte)} steps/ep={spe} in_dim={in_dim}", flush=True)

    lams = (args.lam_task, args.lam_const, args.lam_mono)

    def record(ep):
        ev, _ = evaluate(model, Xte, Wte, Yte, Mte, full=True)
        return {"epoch": ep, **measure(model, Xtr, Mtr, Ytr, Wtr, *lams), "test": ev}

    grads = [record(0)]
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
        if is_measured(ep, args.epochs):
            grads.append(record(ep))
        if ep % 100 == 0 or ep == 1:
            print(f"[{tag}] ep {ep:4d} mean4={ev['mape_mean4']:.2f} best={best:.2f}@{best_ep} "
                  f"cos(task,mono)={grads[-1]['cos_task_mono']:+.3f} "
                  f"cos(task,const)={grads[-1]['cos_task_const']:+.3f} | {time.time()-t0:.0f}s", flush=True)

    last, _ = evaluate(model, Xte, Wte, Yte, Mte, full=True)
    model.load_state_dict(best_state)
    sel, _ = evaluate(model, Xte, Wte, Yte, Mte, full=True)
    res = {"lam_task": args.lam_task, "lam_const": args.lam_const, "lam_mono": args.lam_mono,
           "seed": args.seed, "input": "concorde", "input_dim": in_dim, "split_seed": SEED, "hidden": HIDDEN,
           "best_epoch": best_ep, "selected": sel, "last_epoch": last,
           "train_time_s": time.time() - t0, "grads": grads, "history": hist}
    json.dump(res, open(os.path.join(args.out, f"{tag}.json"), "w"), indent=2)
    print(f"[{tag}] DONE mape={sel['mape_mean4']:.2f}@{best_ep} "
          f"mean cos(task,mono)={np.mean([h['cos_task_mono'] for h in grads]):+.3f}", flush=True)


if __name__ == "__main__":
    main()
